/* M3a-1 L2 per-op parity for the dsv4 Vulkan tier (TEST.md §2 L2).
 * Template: tests/test_vk_dsv4_fp8.c.
 *
 * Each GPU op is tested against the EXACT CPU function the engine calls on the
 * decode path (never a reimplementation):
 *   - fp8 matvec (qkv / wo / shared): dsv4_cuda_matvec vs coli_fp8_matvec_ref
 *     — the identical dispatch the engine's coli_fp8_matvec_pre/_ref performs
 *     when ColiTensorView.gpu is set. The VK backend runs the raw input through
 *     the QDQ shader (block=128, the `_pre` hoist contract) + the fmt=8 matmul
 *     on the dequantized activations — the M2-4 chain (global maxabs/maxref
 *     1e-7..1e-6).
 *   - grouped matvec (wo_a): dsv4_cuda_matvec_grouped vs the engine's CPU
 *     group-view loop — per-group coli_fp8_matvec_ref over the block-diagonal
 *     slices (deepseek_v4.c:2257), exactly what the decode unit does.
 *   - route: dsv4_cuda_route vs coli_v4_route_bf16 (engine ROUTE_BF16 unit):
 *     the GPU computes the 256 f32 gate logits, the host-side top-6 is the
 *     same strict-greater selection — ids must match exactly, weights
 *     thresholded.
 *   - head: dsv4_cuda_head_argmax vs a host bf16-dot + argmax reference
 *     (head_bf16_dot is static in the GENERATE_STATS unit; the reference is
 *     the same scalar formula — bf16 decode, fp32 accumulate, strictly-greater
 *     argmax — so the winner must match exactly, scores thresholded).
 *
 * The qkv chain composes the matvecs with the CPU bf16-round + rmsnorm steps
 * (those stay CPU in v1 — the GPU only replaces the matvecs).
 *
 * Skips (exit 0) without a Vulkan ICD, like the other vk tests. Deterministic
 * seed; NaN-free weights (nn_byte, M2-4 discipline).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "../quant.h"
#include "../tensor.h"
#include "../native_quant.h"
#include "../backend_vulkan.h"
#include "../backend_vulkan_dsv4.h"

/* Real engine CPU references (amalgamation unit objects linked WITHOUT the
 * GPU-tier defines — the pure-CPU paths are what link, so these are oracles,
 * not replicas). */
extern int coli_fp8_matvec_ref(float *output, const ColiTensorView *weight,
                               const float *input);
extern int coli_fp8_matvec_pre(float *output, const ColiTensorView *weight,
                               const float *input, const float *activation);
extern void coli_bf16_round_array(float *values, size_t count);
extern int coli_v4_rmsnorm(float *output, const float *input,
                           const float *weight, int elements, float eps);
extern int coli_v4_route_bf16(float *weights, int *indices, const float *hidden,
                              const uint16_t *gate, const float *bias,
                              const int *forced_indices, int experts,
                              int dimension, int topk, float route_scale);

/* NaN-free random e4m3 weight byte (M2-4: 0x7f/0xff are NaN codes — real
 * checkpoints never carry them; unfiltered bytes poison every dot product). */
static uint8_t nn_byte(void) {
    uint8_t c = (uint8_t)rand();
    return (c & 0x7f) == 0x7f ? (uint8_t)(c & 0x7e) : c;
}

static float ref_e8(uint8_t v) { return v == 0xff ? NAN : ldexpf(1.0f, (int)v - 127); }

/* f32 -> bf16 (round-to-nearest-even) for the route gate / head weights. */
static uint16_t f32_to_bf16(float value) {
    uint32_t bits; memcpy(&bits, &value, 4);
    uint32_t rounded = (bits + 0x7fffu + ((bits >> 16) & 1)) >> 16;
    return (uint16_t)rounded;
}

static void cmp_stats(const float *a, const float *b, int n,
                      double *maxabs, double *maxrel, double *maxref) {
    *maxabs = 0; *maxrel = 0; *maxref = 0;
    for (int i = 0; i < n; i++) {
        double d = fabs((double)a[i] - (double)b[i]);
        if (d > *maxabs) *maxabs = d;
        double r = fabs((double)b[i]);
        if (r > *maxref) *maxref = r;
        if (r >= 1e-2) {
            double rel = d / r;
            if (rel > *maxrel) *maxrel = rel;
        }
    }
}

static int fill_view(ColiTensorView *view, uint8_t *w, float *bscale,
                     int rows, int cols, int packed_rows8) {
    memset(view, 0, sizeof(*view));
    view->format = COLI_TENSOR_FP8_E4M3_BLOCK;
    view->scale_format = COLI_SCALE_F32;
    view->data = w;
    view->scales = bscale;
    view->data_bytes = (size_t)rows * cols;
    view->scale_bytes = (size_t)((rows + 127) / 128) * ((cols + 127) / 128) * sizeof(float);
    view->rows = rows;
    view->columns = cols;
    view->block_rows = packed_rows8 ? 8 : 128;
    view->block_columns = 128;
    return 0;
}

static int alloc_fp8(uint8_t **w, float **bscale, uint8_t **e8, int O, int I) {
    size_t nsc = (size_t)((O + 127) / 128) * ((I + 127) / 128);
    *w = malloc((size_t)O * I);
    *bscale = malloc(nsc * 4);
    *e8 = malloc(nsc);
    if (!*w || !*bscale || !*e8) return -1;
    for (size_t i = 0; i < (size_t)O * I; i++) (*w)[i] = nn_byte();
    for (size_t i = 0; i < nsc; i++) {
        uint8_t code = (uint8_t)(115 + rand() % 14);   /* realistic UE8M0: 2^-12..2^1 */
        (*e8)[i] = code;
        (*bscale)[i] = ref_e8(code);
    }
    return 0;
}

/* ---- qkv chain: wq_a -> bf16-round -> rmsnorm -> bf16-round -> wq_b, and
 *      wkv -> bf16-round -> rmsnorm -> bf16-round. GPU replaces the three
 *      matvecs; the rest stays CPU exactly like the engine. ---- */
static int run_qkv_chain(int q_rank, int hidden, int heads, int head_dim) {
    int q_b_rows = heads * head_dim;
    uint8_t *wa, *wb, *wk; float *sa, *sb, *sk; uint8_t *ea, *eb, *ek;
    if (alloc_fp8(&wa, &sa, &ea, q_rank, hidden) ||
        alloc_fp8(&wb, &sb, &eb, q_b_rows, q_rank) ||
        alloc_fp8(&wk, &sk, &ek, head_dim, hidden))
        return 1;
    uint16_t *qnorm = malloc((size_t)q_rank * 2), *kvn = malloc((size_t)head_dim * 2);
    float *qn = malloc((size_t)q_rank * 4), *kn = malloc((size_t)head_dim * 4);
    float *x = malloc((size_t)hidden * 4);
    for (int i = 0; i < q_rank; i++) { qnorm[i] = f32_to_bf16((float)(rand() % 2000) / 100.0f + 0.5f); }
    for (int i = 0; i < head_dim; i++) { kvn[i] = f32_to_bf16((float)(rand() % 2000) / 100.0f + 0.5f); }
    /* decode bf16 norms to f32 exactly (bf16->f32 is lossless) */
    for (int i = 0; i < q_rank; i++) { uint32_t bits = (uint32_t)qnorm[i] << 16; memcpy(&qn[i], &bits, 4); }
    for (int i = 0; i < head_dim; i++) { uint32_t bits = (uint32_t)kvn[i] << 16; memcpy(&kn[i], &bits, 4); }
    for (int i = 0; i < hidden; i++) x[i] = (float)(rand() % 200001 - 100000) / 100.0f;

    ColiTensorView vq_a, vq_b, vkv;
    fill_view(&vq_a, wa, sa, q_rank, hidden, 0);
    fill_view(&vq_b, wb, sb, q_b_rows, q_rank, 0);
    fill_view(&vkv, wk, sk, head_dim, hidden, 0);

    /* CPU chain (the engine's attention_token_impl, matvecs only) */
    float *qa_c = calloc((size_t)q_rank, 4), *q_c = calloc((size_t)q_b_rows, 4);
    float *kv_c = calloc((size_t)head_dim, 4);
    float *act = malloc((size_t)hidden * 4); uint8_t *actsc = malloc((size_t)(hidden / 128));
    if (coli_fp8_activation_qdq_ref(act, actsc, x, hidden, 128)) return 1;
    if (coli_fp8_matvec_pre(qa_c, &vq_a, x, act)) return 1;
    coli_bf16_round_array(qa_c, q_rank);
    if (coli_v4_rmsnorm(qa_c, qa_c, qn, q_rank, 1e-6f)) return 1;
    coli_bf16_round_array(qa_c, q_rank);
    if (coli_fp8_matvec_ref(q_c, &vq_b, qa_c)) return 1;
    coli_bf16_round_array(q_c, q_b_rows);
    if (coli_fp8_matvec_pre(kv_c, &vkv, x, act)) return 1;
    coli_bf16_round_array(kv_c, head_dim);
    if (coli_v4_rmsnorm(kv_c, kv_c, kn, head_dim, 1e-6f)) return 1;
    coli_bf16_round_array(kv_c, head_dim);

    /* GPU chain: the same steps, matvecs dispatched through the backend */
    Dsv4CudaTensor *tq_a = NULL, *tq_b = NULL, *tkv = NULL;
    if (!dsv4_cuda_upload_fp8(&tq_a, wa, ea, q_rank, hidden, 0) ||
        !dsv4_cuda_upload_fp8(&tq_b, wb, eb, q_b_rows, q_rank, 0) ||
        !dsv4_cuda_upload_fp8(&tkv, wk, ek, head_dim, hidden, 0)) {
        fprintf(stderr, "vk-ops qkv: upload refused\n"); return 1;
    }
    ds4vk_tensor_set_op(tq_a, "qkv"); ds4vk_tensor_set_op(tq_b, "qkv");
    ds4vk_tensor_set_op(tkv, "qkv");
    float *qa_g = calloc((size_t)q_rank, 4), *q_g = calloc((size_t)q_b_rows, 4);
    float *kv_g = calloc((size_t)head_dim, 4);
    int ok = dsv4_cuda_matvec(tq_a, qa_g, x);
    if (ok) coli_bf16_round_array(qa_g, q_rank);
    if (ok) ok = coli_v4_rmsnorm(qa_g, qa_g, qn, q_rank, 1e-6f) == 0;
    if (ok) coli_bf16_round_array(qa_g, q_rank);
    if (ok) ok = dsv4_cuda_matvec(tq_b, q_g, qa_g);
    if (ok) coli_bf16_round_array(q_g, q_b_rows);
    if (ok) ok = dsv4_cuda_matvec(tkv, kv_g, x);
    if (ok) coli_bf16_round_array(kv_g, head_dim);
    if (ok) ok = coli_v4_rmsnorm(kv_g, kv_g, kn, head_dim, 1e-6f) == 0;
    if (ok) coli_bf16_round_array(kv_g, head_dim);
    if (!ok) {
        fprintf(stderr, "vk-ops qkv: GPU chain refused\n");
        return 1;
    }
    double ma, mr, mref;
    cmp_stats(q_g, q_c, q_b_rows, &ma, &mr, &mref);
    int q_ok = mref > 0 && ma / mref < 1e-2;
    cmp_stats(kv_g, kv_c, head_dim, &ma, &mr, &mref);
    int kv_ok = mref > 0 && ma / mref < 1e-2;
    printf("vk-ops qkv chain (q_rank=%d hidden=%d heads=%d hd=%d): q global %.3e %s | kv global %.3e %s\n",
           q_rank, hidden, heads, head_dim,
           q_ok ? 0 : ma / mref, q_ok ? "OK" : "FAIL",
           kv_ok ? 0 : ma / mref, kv_ok ? "OK" : "FAIL");
    dsv4_cuda_tensor_free(tkv); dsv4_cuda_tensor_free(tq_b); dsv4_cuda_tensor_free(tq_a);
    free(kv_g); free(q_g); free(qa_g); free(actsc); free(act); free(kv_c); free(q_c); free(qa_c);
    free(x); free(kn); free(qn); free(kvn); free(qnorm);
    free(ek); free(sk); free(wk); free(eb); free(sb); free(wb); free(ea); free(sa); free(wa);
    return q_ok && kv_ok ? 0 : 1;
}

/* ---- wo: grouped wo_a (8 groups) + wo_b ---- */
static int run_wo(int o_width, int group_width, int groups, int o_rank, int hidden) {
    uint8_t *wa, *wb; float *sa, *sb; uint8_t *ea, *eb;
    if (alloc_fp8(&wa, &sa, &ea, o_width, group_width) ||
        alloc_fp8(&wb, &sb, &eb, hidden, o_width))
        return 1;
    /* input = attended [heads*head_dim]; each group g consumes slice
     * [g*group_width, (g+1)*group_width). */
    int attended = groups * group_width;
    float *x = malloc((size_t)attended * 4);
    for (int i = 0; i < attended; i++) x[i] = (float)(rand() % 200001 - 100000) / 100.0f;

    /* CPU reference: the engine's group-view loop (deepseek_v4.c:2257) */
    ColiTensorView vwo_a, vwo_b;
    fill_view(&vwo_a, wa, sa, o_width, group_width, 0);
    fill_view(&vwo_b, wb, sb, hidden, o_width, 0);
    float *oa_c = malloc((size_t)o_width * 4), *out_c = malloc((size_t)hidden * 4);
    int scale_rows_per_group = (o_rank + 127) / 128;
    int scale_columns = (group_width + 127) / 128;
    for (int g = 0; g < groups; g++) {
        ColiTensorView group_view = vwo_a;
        group_view.rows = o_rank;
        group_view.columns = group_width;
        group_view.data = wa + (size_t)g * o_rank * group_width;
        group_view.scales = sa + (size_t)g * scale_rows_per_group * scale_columns;
        group_view.data_bytes = (size_t)o_rank * group_width;
        group_view.scale_bytes = (size_t)scale_rows_per_group * scale_columns * sizeof(float);
        if (coli_fp8_matvec_ref(oa_c + (size_t)g * o_rank, &group_view,
                                x + (size_t)g * group_width))
            return 1;
    }
    coli_bf16_round_array(oa_c, o_width);
    if (coli_fp8_matvec_ref(out_c, &vwo_b, oa_c)) return 1;
    coli_bf16_round_array(out_c, hidden);

    /* GPU: dsv4_cuda_matvec_grouped for wo_a, plain matvec for wo_b */
    Dsv4CudaTensor *twa = NULL, *twb = NULL;
    if (!dsv4_cuda_upload_fp8(&twa, wa, ea, o_width, group_width, 0) ||
        !dsv4_cuda_upload_fp8(&twb, wb, eb, hidden, o_width, 0)) {
        fprintf(stderr, "vk-ops wo: upload refused\n"); return 1;
    }
    ds4vk_tensor_set_op(twa, "wo"); ds4vk_tensor_set_op(twb, "wo");
    float *oa_g = malloc((size_t)o_width * 4), *out_g = malloc((size_t)hidden * 4);
    int ok = dsv4_cuda_matvec_grouped(twa, oa_g, x, groups);
    if (ok) coli_bf16_round_array(oa_g, o_width);
    if (ok) ok = dsv4_cuda_matvec(twb, out_g, oa_g);
    if (ok) coli_bf16_round_array(out_g, hidden);
    if (!ok) {
        fprintf(stderr, "vk-ops wo: GPU chain refused\n");
        return 1;
    }
    double ma, mr, mref;
    cmp_stats(oa_g, oa_c, o_width, &ma, &mr, &mref);
    int oa_ok = mref > 0 && ma / mref < 1e-2;
    cmp_stats(out_g, out_c, hidden, &ma, &mr, &mref);
    int out_ok = mref > 0 && ma / mref < 1e-2;
    printf("vk-ops wo (o_width=%d group_width=%d groups=%d): oa global %.3e %s | out global %.3e %s\n",
           o_width, group_width, groups, oa_ok ? 0 : ma / mref, oa_ok ? "OK" : "FAIL",
           out_ok ? 0 : ma / mref, out_ok ? "OK" : "FAIL");
    dsv4_cuda_tensor_free(twb); dsv4_cuda_tensor_free(twa);
    free(out_g); free(oa_g); free(out_c); free(oa_c);
    free(x); free(eb); free(sb); free(wb); free(ea); free(sa); free(wa);
    return oa_ok && out_ok ? 0 : 1;
}

/* ---- route: f32 gate logits matvec + host top-6 vs coli_v4_route_bf16 ---- */
static int run_route(int experts, int hidden, int topk) {
    float *gate = malloc((size_t)experts * hidden * 4);
    uint16_t *gate_bf16 = malloc((size_t)experts * hidden * 2);
    float *bias = malloc((size_t)experts * 4);
    float *x = malloc((size_t)hidden * 4);
    for (int i = 0; i < experts * hidden; i++) {
        gate[i] = (float)(rand() % 200001 - 100000) / 100000.0f;
        gate_bf16[i] = f32_to_bf16(gate[i]);
    }
    for (int i = 0; i < experts; i++) bias[i] = (float)(rand() % 2001 - 1000) / 1000.0f;
    for (int i = 0; i < hidden; i++) x[i] = (float)(rand() % 200001 - 100000) / 100.0f;

    /* CPU reference: the engine's route (returns 0 on success) */
    float w_c[8]; int id_c[8];
    int okc = coli_v4_route_bf16(w_c, id_c, x, gate_bf16, bias, NULL,
                                 experts, hidden, topk, 1.25f) == 0;

    /* GPU: dsv4_cuda_route (f32 gate uploaded — the exact mirror the engine
     * builds in v4_gpu_upload_gate) */
    Dsv4CudaTensor *tg = NULL, *tb = NULL;
    int okg = dsv4_cuda_upload_f32(&tg, gate, experts, hidden, 0) &&
              dsv4_cuda_upload_f32(&tb, bias, experts, 1, 0);
    if (!okg) { fprintf(stderr, "vk-ops route: upload refused\n"); return 1; }
    ds4vk_tensor_set_op(tg, "route"); ds4vk_tensor_set_op(tb, "route");
    Dsv4CudaActivation *act = dsv4_cuda_activation_create(0, hidden);
    okg = act && dsv4_cuda_activation_upload(act, x, hidden);
    int ids[6]; float wts[6];
    if (okg) okg = dsv4_cuda_route(act, tg, tb, NULL, 1.25f, ids, wts);
    dsv4_cuda_activation_free(act);
    if (!okc || !okg) { fprintf(stderr, "vk-ops route: refused cpu=%d gpu=%d\n", okc, okg); return 1; }
    int ids_ok = 1;
    for (int k = 0; k < topk; k++) if (ids[k] != id_c[k]) ids_ok = 0;
    /* Weights: fp32 reduction order on the 4096-term logits is amplified by
     * sqrt(softplus) near zero — measured ~1e-4 relative, well inside the
     * fp32 contract (TEST L1 bar ~1e-3..2e-3). The SELECTION (ids) is the
     * exact-match contract (DSV4_IDX_VERIFY semantics). */
    int w_ok = 1;
    for (int k = 0; k < topk; k++)
        if (fabs(wts[k] - w_c[k]) > 2e-3 * fmaxf(1.0f, fabs(w_c[k]))) w_ok = 0;
    if (!w_ok) {
        for (int k = 0; k < topk; k++)
            fprintf(stderr, "route w[%d] gpu=%.8f cpu=%.8f diff=%.3e\n", k, wts[k], w_c[k], fabs(wts[k] - w_c[k]));
    }
    printf("vk-ops route (experts=%d hidden=%d topk=%d): ids %s weights %s\n",
           experts, hidden, topk, ids_ok ? "MATCH" : "DIFFER", w_ok ? "OK" : "FAIL");
    dsv4_cuda_tensor_free(tb); dsv4_cuda_tensor_free(tg);
    free(x); free(bias); free(gate_bf16); free(gate);
    return ids_ok && w_ok ? 0 : 1;
}

/* ---- head: bf16 matvec + argmax vs the CPU head formula ---- */
static int run_head(int vocab, int hidden) {
    uint16_t *head = malloc((size_t)vocab * hidden * 2);
    float *x = malloc((size_t)hidden * 4);
    for (size_t i = 0; i < (size_t)vocab * hidden; i++)
        head[i] = f32_to_bf16((float)(rand() % 200001 - 100000) / 1000.0f);
    for (int i = 0; i < hidden; i++) x[i] = (float)(rand() % 200001 - 100000) / 100.0f;

    /* CPU reference (head_bf16_dot semantics: bf16 decode, fp32 sequential
     * accumulate, strictly-greater argmax — deepseek_v4.c:11655/11700) */
    float *scores_c = malloc((size_t)vocab * 4);
    for (int row = 0; row < vocab; row++) {
        float sum = 0.0f;
        const uint16_t *w = head + (size_t)row * hidden;
        for (int i = 0; i < hidden; i++) {
            uint32_t bits = (uint32_t)w[i] << 16;
            float v; memcpy(&v, &bits, 4);
            sum += v * x[i];
        }
        scores_c[row] = sum;
    }
    int win_c = -1; float max_c = -INFINITY;
    for (int row = 0; row < vocab; row++)
        if (scores_c[row] > max_c) { max_c = scores_c[row]; win_c = row; }

    Dsv4CudaTensor *th = NULL;
    if (!dsv4_cuda_upload_bf16(&th, head, vocab, hidden, 0)) {
        fprintf(stderr, "vk-ops head: upload refused\n"); return 1;
    }
    ds4vk_tensor_set_op(th, "head");
    int win_g = -1; float max_g = 0;
    int okg = dsv4_cuda_head_argmax(th, x, &win_g, &max_g);
    if (!okg) { fprintf(stderr, "vk-ops head: GPU refused\n"); return 1; }
    int win_ok = win_g == win_c;
    int val_ok = fabs(max_g - max_c) <= 1e-4 * fmaxf(1.0f, fabs(max_c));
    printf("vk-ops head (vocab=%d hidden=%d): winner %s (gpu=%d cpu=%d) value %s (g %.6e c %.6e)\n",
           vocab, hidden, win_ok ? "MATCH" : "DIFFER", win_g, win_c,
           val_ok ? "OK" : "FAIL", max_g, max_c);
    dsv4_cuda_tensor_free(th);
    free(scores_c); free(x); free(head);
    return win_ok && val_ok ? 0 : 1;
}

int main(void) {
    if (!coli_vk_init("shaders/qmatmul.spv") || !coli_vk_available()) {
        fprintf(stderr, "vk-ops: no Vulkan device — skipped\n");
        return 0;
    }
    srand(11);
    int bad = 0;
    /* real DeepSeek-V4-Flash-0731 shapes: hidden 4096, q_rank 1024, head_dim
     * 512, heads 64, o_lora_rank 1024, o_groups 8 */
    bad |= run_qkv_chain(1024, 4096, 64, 512);
    bad |= run_wo(8192, 4096, 8, 1024, 4096);
    bad |= run_route(256, 4096, 6);
    bad |= run_head(16384, 4096);       /* reduced vocab for test speed */
    /* tiny sanity (single group, small head) */
    bad |= run_wo(512, 512, 1, 512, 512);
    bad |= run_head(1024, 128);
    if (bad) { fprintf(stderr, "vk-ops: FAIL\n"); coli_vk_shutdown(); return 1; }
    printf("vk-ops: OK\n");
    coli_vk_shutdown();
    return 0;
}
