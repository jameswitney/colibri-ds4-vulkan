/* M3b-1 L2 attention parity for the dsv4 Vulkan tier (TEST.md §2 L2).
 * Template: tests/test_dsv4_sparse_attn_batch_cuda.c.
 *
 * The GPU attention core (sparse_attn.comp over the persistent device KV:
 * window ring + compressed pool + sigmoid sink, bf16 probability/value) is
 * tested against the EXACT CPU function the engine calls on the decode path —
 * coli_v4_sparse_attention_ref (the REAL engine unit, linked as the oracle):
 *   - cached variant (dsv4_cuda_sparse_attn_batch_cached): the engine's
 *     decode block at deepseek_v4.c:2151 (ring seeded with the historic
 *     window rows, the current chunk riding along as its own buffer);
 *   - indexed variant (_cached_idx): the indexer-selected compressed subset,
 *     the same ordinals the decode block passes for indexer layers;
 *   - ring wrap: positions crossing the window boundary (the engine reseeds
 *     the ring on discontinuity — the test covers the slot arithmetic).
 *
 * Skips (exit 0) without a Vulkan ICD. Deterministic seed. Threshold: 2e-3
 * abs on the bf16-rounded outputs (fp32 reduction order on the scores) — the
 * same bar as the CUDA test.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "../backend_vulkan.h"
#include "../backend_vulkan_dsv4.h"

/* The REAL engine CPU reference (deepseek_v4.c COLI_V4_UNIT_SPARSE_ATTENTION,
 * linked without the GPU-tier defines so the pure-CPU path is what links). */
extern int coli_v4_sparse_attention_ref(float *output, const float *queries,
                                        const float *kv, const float *sinks,
                                        const int *indices, int heads,
                                        int head_dimension, int kv_count,
                                        int topk, float softmax_scale);

static float frand(unsigned *seed) {
    *seed = *seed * 1664525u + 1013904223u;
    return ((*seed >> 8) & 0xffff) / 65536.0f - 0.5f;
}

/* One shared case runner for both variants. sel==NULL runs the plain cached
 * kernel (compressed prefix); else the idx kernel with sel ordinals. */
static int run_case(int use_idx, int topk) {
    enum { HEADS = 64, DIM = 512, WINDOW = 128, T = 16, START = 200, COMP = 60 };
    unsigned seed = 24680;
    int first = START - WINDOW + 1;
    int pre = START - first;             /* historic rows seeded into the ring */
    int comp_base = START + T;           /* compressed pool rows (absolute pos) */
    float *q = malloc((size_t)T * HEADS * DIM * sizeof(float));
    float *vals = malloc((size_t)(comp_base + COMP) * DIM * sizeof(float));
    float *sinks = malloc(HEADS * sizeof(float));
    int *meta = malloc((size_t)T * 3 * sizeof(int));
    float *got = malloc((size_t)T * HEADS * DIM * sizeof(float));
    float *want = malloc((size_t)HEADS * DIM * sizeof(float));
    if (!q || !vals || !sinks || !meta || !got || !want) return 2;
    for (size_t i = 0; i < (size_t)T * HEADS * DIM; i++) q[i] = frand(&seed);
    for (size_t i = 0; i < (size_t)(comp_base + COMP) * DIM; i++) vals[i] = frand(&seed);
    for (int h = 0; h < HEADS; h++) sinks[h] = frand(&seed) * 4.0f;
    int *sel = malloc((size_t)T * topk * sizeof(*sel));
    int *pool = malloc((size_t)COMP * sizeof(*pool));
    if (!sel || !pool) return 2;
    for (int t = 0; t < T; t++) {
        int pos = START + t;
        int wfirst = pos - WINDOW + 1;
        if (wfirst < 0) wfirst = 0;
        meta[3 * t] = wfirst - first;
        meta[3 * t + 1] = pos - wfirst + 1;
        int cn = COMP * (t + 1) / T;     /* growing prefix */
        meta[3 * t + 2] = use_idx ? (cn < topk ? cn : topk) : cn;
        if (use_idx) {                   /* partial shuffle: unsorted subset */
            for (int i = 0; i < cn; i++) pool[i] = i;
            int sn = meta[3 * t + 2];
            for (int i = 0; i < sn; i++) {
                int j = i + (int)((frand(&seed) + 0.5f) * (cn - i));
                if (j >= cn) j = cn - 1;
                int tmp = pool[i]; pool[i] = pool[j]; pool[j] = tmp;
                sel[(size_t)t * topk + i] = pool[i];
            }
        }
    }
    float scale = 1.0f / sqrtf((float)DIM);
    enum { LAYER = 7 };
    /* ring: positions [first, START) at slot (pos % window); chunk: absolute
     * rows [START, START+T); pool: absolute rows [comp_base, comp_base+COMP) */
    if (!coli_vk_dsv4_kv_ring_append(LAYER, vals + (size_t)first * DIM, first,
                                     pre, WINDOW, DIM) ||
        !coli_vk_dsv4_kv_comp_append(LAYER, vals + (size_t)comp_base * DIM,
                                     0, COMP, DIM)) {
        fprintf(stderr, "vk-attn: kv seed refused\n");
        return 3;
    }
    int ok = use_idx
        ? coli_vk_dsv4_attn_cached_idx(LAYER, got, q, vals + (size_t)START * DIM,
                                       START, sinks, meta, sel, topk, first,
                                       COMP, HEADS, DIM, T, scale)
        : coli_vk_dsv4_attn_cached(LAYER, got, q, vals + (size_t)START * DIM,
                                   START, sinks, meta, first, COMP, HEADS, DIM,
                                   T, scale);
    if (!ok) {
        fprintf(stderr, "vk-attn: attention refused (%s)\n", use_idx ? "idx" : "plain");
        return 4;
    }
    double worst = 0.0;
    for (int t = 0; t < T; t++) {
        int pos = START + t;
        int wfirst = pos - WINDOW + 1;
        if (wfirst < 0) wfirst = 0;
        int cn = COMP * (t + 1) / T;
        int sn = use_idx ? (cn < topk ? cn : topk) : cn;
        /* The ordered-slab ref: window rows in order, then the selected rows. */
        int wn = pos - wfirst + 1;
        float *kv = malloc((size_t)(wn + sn) * DIM * sizeof(*kv));
        for (int i = 0; i < wn; i++)
            memcpy(kv + (size_t)i * DIM, vals + (size_t)(wfirst + i) * DIM,
                   (size_t)DIM * sizeof(*kv));
        for (int i = 0; i < sn; i++)
            memcpy(kv + (size_t)(wn + i) * DIM,
                   vals + (size_t)(comp_base + (use_idx ? sel[(size_t)t * topk + i] : i)) * DIM,
                   (size_t)DIM * sizeof(*kv));
        int *idx = malloc((size_t)(wn + sn) * sizeof(*idx));
        for (int i = 0; i < wn + sn; i++) idx[i] = i;
        if (coli_v4_sparse_attention_ref(want, q + (size_t)t * HEADS * DIM, kv,
                                         sinks, idx, HEADS, DIM, wn + sn,
                                         wn + sn, scale)) {
            fprintf(stderr, "ref refused t=%d\n", t);
            return 5;
        }
        for (size_t i = 0; i < (size_t)HEADS * DIM; i++) {
            double diff = fabs((double)want[i] - got[(size_t)t * HEADS * DIM + i]);
            if (diff > worst) worst = diff;
            if (diff > 2e-3) {
                fprintf(stderr, "vk-attn mismatch (%s) t=%d i=%zu want=%g got=%g\n",
                        use_idx ? "idx" : "plain", t, i, want[i],
                        got[(size_t)t * HEADS * DIM + i]);
                return 6;
            }
        }
        free(idx); free(kv);
    }
    printf("vk-attn %s (heads=%d dim=%d window=%d comp<=%d): max |diff| = %g\n",
           use_idx ? "cached idx" : "cached", HEADS, DIM, WINDOW, COMP, worst);
    free(pool); free(sel); free(want); free(got); free(meta); free(sinks);
    free(vals); free(q);
    return 0;
}

/* Ring wrap: positions crossing the window boundary (slot arithmetic). The
 * engine's sync reseeds the ring from [wfirst, pos) on discontinuity — the
 * test seeds exactly that range (which wraps the ring once) and attends at
 * the wrapped position with the full window. */
static int run_wrap(void) {
    enum { HEADS = 8, DIM = 128, WINDOW = 32, POS = 40 };
    unsigned seed = 97531;
    int wfirst = POS - WINDOW + 1;       /* 9 */
    float *q = malloc((size_t)HEADS * DIM * sizeof(float));
    float *vals = malloc((size_t)(POS + 1) * DIM * sizeof(float));
    float *sinks = malloc(HEADS * sizeof(float));
    float *got = malloc((size_t)HEADS * DIM * sizeof(float));
    float *want = malloc((size_t)HEADS * DIM * sizeof(float));
    if (!q || !vals || !sinks || !got || !want) return 2;
    for (int i = 0; i < HEADS * DIM; i++) q[i] = frand(&seed);
    for (size_t i = 0; i < (size_t)(POS + 1) * DIM; i++) vals[i] = frand(&seed);
    for (int h = 0; h < HEADS; h++) sinks[h] = frand(&seed) * 4.0f;
    enum { LAYER = 3 };
    /* seed [wfirst, POS) = 31 rows: slots 9..31 then 0..7 (one wrap) */
    if (!coli_vk_dsv4_kv_ring_append(LAYER, vals + (size_t)wfirst * DIM,
                                     wfirst, POS - wfirst, WINDOW, DIM)) {
        fprintf(stderr, "vk-attn wrap: ring refused\n");
        return 3;
    }
    int meta[3] = {0, WINDOW, 0};        /* wn=32 rows: [wfirst..POS] */
    float scale = 1.0f / sqrtf((float)DIM);
    /* chunk = the last row itself (absolute pos POS); ring holds [wfirst, POS) */
    if (!coli_vk_dsv4_attn_cached(LAYER, got, q, vals + (size_t)POS * DIM, POS,
                                  sinks, meta, wfirst, 0, HEADS, DIM, 1, scale)) {
        fprintf(stderr, "vk-attn wrap: attention refused\n");
        return 4;
    }
    /* CPU ref over the ordered slab [wfirst..pos] */
    float *kv = malloc((size_t)WINDOW * DIM * sizeof(*kv));
    int *idx = malloc((size_t)WINDOW * sizeof(*idx));
    for (int i = 0; i < WINDOW; i++) {
        memcpy(kv + (size_t)i * DIM, vals + (size_t)(wfirst + i) * DIM,
               (size_t)DIM * sizeof(*kv));
        idx[i] = i;
    }
    if (coli_v4_sparse_attention_ref(want, q, kv, sinks, idx, HEADS, DIM,
                                     WINDOW, WINDOW, scale))
        return 5;
    double worst = 0.0;
    for (size_t i = 0; i < (size_t)HEADS * DIM; i++) {
        double diff = fabs((double)want[i] - got[i]);
        if (diff > worst) worst = diff;
        if (diff > 2e-3) {
            fprintf(stderr, "vk-attn wrap mismatch i=%zu want=%g got=%g\n", i,
                    want[i], got[i]);
            return 6;
        }
    }
    printf("vk-attn ring wrap (window=%d, %d rows appended): max |diff| = %g\n",
           WINDOW, POS - wfirst, worst);
    free(idx); free(kv); free(want); free(got); free(sinks); free(vals); free(q);
    return 0;
}

int main(void) {
    if (!coli_vk_init("shaders/qmatmul.spv") || !coli_vk_available()) {
        fprintf(stderr, "vk-attn: no Vulkan device — skipped\n");
        return 0;
    }
    srand(21);
    int bad = run_case(0, 0);      /* plain cached (growing compressed prefix) */
    bad |= run_case(1, 24);        /* indexed subset (topk=24) */
    bad |= run_wrap();
    coli_vk_dsv4_kv_free_all();
    coli_vk_shutdown();
    if (bad) { fprintf(stderr, "vk-attn: FAIL\n"); return 1; }
    printf("vk-attn: OK\n");
    return 0;
}
