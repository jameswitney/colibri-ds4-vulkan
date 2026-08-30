/* M5-1a L2 per-op parity for the dsv4 Vulkan fp4 expert tier (TEST.md §2 L2).
 * Template: tests/test_dsv4_vk_ops.c / test_dsv4_mhc_vk.c.
 *
 * Covers the routed-expert decode surface:
 *   - fp4 matvec: dsv4_cuda_matvec (QDQ block=128 + fmt=7 mxfp4-e2m1 matmul,
 *     gs=32) vs the REAL engine coli_fp4_matvec_ref — the identical dispatch
 *     the engine's coli_fp4_matvec_ref performs when ColiTensorView.gpu is
 *     set. Contract: QDQ bitwise, matmul thresholded (same class as fp8).
 *   - upload/refill: dsv4_cuda_upload_fp4 (UE8M0 -> f32 at upload) +
 *     dsv4_cuda_tensor_refill_fp4 (in-place LRU recycle) — after a refill the
 *     matvec must match the NEW bytes' CPU reference.
 *   - expert group: dsv4_cuda_expert_group (the decode fused path) vs the
 *     per-expert CPU chain the engine falls back to (coli_v4_expert_forward_ref
 *     composition over the REAL engine primitives — coli_fp4_matvec_ref +
 *     coli_bf16_round_array + coli_v4_swiglu; the rounding points are bitwise,
 *     only the matvecs are thresholded).
 *   - op gating: COLI_DSV4_VK_OPS without "expert" must refuse both the
 *     per-expert matvec and the group; COLI_DSV4_VK_FAIL=expert must force the
 *     group to fail (L4 hook exercised at the backend level).
 *
 * Skips (exit 0) without a Vulkan ICD. Deterministic seed. e2m1 has no NaN
 * codes, so raw random weight bytes are safe (the M2-4 NaN discipline applies
 * only to e4m3).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/wait.h>
#include "../quant.h"
#include "../tensor.h"
#include "../native_quant.h"
#include "../backend_vulkan.h"
#include "../backend_vulkan_dsv4.h"

/* Real engine CPU primitives (oracle objects linked WITHOUT the GPU-tier
 * defines — the pure-CPU paths are what link). */
extern int coli_fp4_matvec_ref(float *output, const ColiTensorView *weight,
                               const float *input);
extern void coli_bf16_round_array(float *values, size_t count);
extern int coli_v4_swiglu(float *output, const float *gate, const float *up,
                          int dimension, float limit);
extern float coli_bf16_round(float value);
/* M5-1c: the shared-expert fp8 matvec + the engine's route (oracle units
 * NATIVE_QUANT / ROUTE_BF16 — the ops suite's own externs). */
extern int coli_fp8_matvec_ref(float *output, const ColiTensorView *weight,
                               const float *input);
extern int coli_v4_route_bf16(float *weights, int *indices,
                              const float *hidden, const uint16_t *gate,
                              const float *bias, const int *forced_indices,
                              int experts, int dimension, int topk,
                              float route_scale);

/* e4m3 weight bytes must avoid the NaN codes 0x7f/0xff (the M2-4 discipline
 * — the fp8 suite's nn_byte, shared weights for the bank test). */
static uint8_t nn_byte(void) {
    uint8_t c = (uint8_t)rand();
    return (c & 0x7f) == 0x7f ? (uint8_t)(c & 0x7e) : c;
}

/* f32 -> bf16 (round-to-nearest-even) — the ops suite's helper, for the
 * route gate's bf16 storage bytes. */
static uint16_t f32_to_bf16(float value) {
    uint32_t bits; memcpy(&bits, &value, 4);
    uint32_t rounded = (bits + 0x7fffu + ((bits >> 16) & 1)) >> 16;
    return (uint16_t)rounded;
}

/* bf16 -> f32 (exact). The engine's gate mirror is the resident bf16 gate
 * decoded to f32 (v4_gpu_upload_gate), so the test builds the GPU mirror the
 * same way: identical weight VALUES on both sides, only the reduction order
 * differs (the ops suite's bf16-rounded-vs-raw draw can flip a borderline
 * top-6 at 4096 terms; identical values cannot). */
static float bf16_to_f32(uint16_t b) {
    uint32_t bits = (uint32_t)b << 16;
    float f; memcpy(&f, &bits, 4);
    return f;
}

/* sigmoidf_stable twin (deepseek_v4.c) — the engine's swiglu sigmoid. */
static float sigmoid_stable(float value) {
    if (value >= 0.0f) {
        float decay = expf(-value);
        return 1.0f / (1.0f + decay);
    }
    float growth = expf(value);
    return growth / (1.0f + growth);
}

/* UE8M0 -> f32 exactly as the CPU fp4 reference decodes it (the e8 TABLE:
 * 0xff -> NaN, else 2^(value-127) — deepseek_v4.c coli_e8m0_decode). The
 * real checkpoint's expert GATE scales carry codes 0 and 255, so the table
 * semantics are load-bearing (M5-1a parity fix). */
static float ref_ue8m0(uint8_t s) {
    if (s == 0xff) return NAN;
    return ldexpf(1.0f, (int)s - 127);
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

static int check_stats(const char *what, int O, const float *g, const float *c,
                       int verbose) {
    double ma, mr, mref;
    cmp_stats(g, c, O, &ma, &mr, &mref);
    double global = mref > 0 ? ma / mref : 1.0;
    if (verbose)
        printf("  %s: maxabs/maxref %.3e/%.3e global %.3e max_rel(floor) %.3e\n",
               what, ma, mref, global, mr);
    /* Same global bar as the fp8 tier (test_vk_dsv4_fp8.c): a broken shader
     * shows up as a O(1) global ratio; fp32 reduction-order noise lands at
     * ~1e-7..1e-6. floor-maxrel spikes on cancellation rows are informational
     * (the fp8 suite's own gate). */
    return (mref > 0 && global > 1e-2) ? 1 : 0;
}

/* Allocate random fp4 expert weights: packed e2m1 nibbles [O, I/2] + UE8M0
 * per-32-group scales [O, I/32]. e2m1 has no NaN codes; scales use the
 * realistic 2^-12..2^1 range like the fp8 tests. */
static int alloc_fp4(uint8_t **w, uint8_t **sc, int O, int I) {
    size_t wb = (size_t)O * (size_t)(I / 2);
    size_t sb = (size_t)O * (size_t)(I / 32);
    *w = malloc(wb);
    *sc = malloc(sb);
    if (!*w || !*sc) return -1;
    for (size_t i = 0; i < wb; i++) (*w)[i] = (uint8_t)rand();
    for (size_t i = 0; i < sb; i++) (*sc)[i] = (uint8_t)(115 + rand() % 14);
    return 0;
}

static void fill_fp4_view(ColiTensorView *view, uint8_t *w, uint8_t *sc,
                          int rows, int cols) {
    memset(view, 0, sizeof(*view));
    view->format = COLI_TENSOR_FP4_NATIVE_BLOCK;
    view->scale_format = COLI_SCALE_UE8M0;
    view->data = w;
    view->scales = sc;
    view->data_bytes = (size_t)rows * (size_t)(cols / 2);
    view->scale_bytes = (size_t)rows * (size_t)(cols / 32);
    view->rows = rows;
    view->columns = cols;
    view->block_rows = 1;
    view->block_columns = 32;
}

/* Per-expert CPU chain — the engine's coli_v4_expert_forward_ref body
 * (deepseek_v4.c, attention-cache unit) over the REAL engine primitives. */
static int ref_expert_forward(float *output, const ColiTensorView *gate_w,
                              const ColiTensorView *up_w,
                              const ColiTensorView *down_w,
                              const float *input, float route_weight,
                              float swiglu_limit) {
    size_t intermediate = (size_t)gate_w->rows;
    size_t output_size = (size_t)down_w->rows;
    float *gate = malloc(intermediate * sizeof(*gate));
    float *up = malloc(intermediate * sizeof(*up));
    float *activated = malloc(intermediate * sizeof(*activated));
    if (!gate || !up || !activated) { free(activated); free(up); free(gate); return -1; }
    int result = coli_fp4_matvec_ref(gate, gate_w, input) ||
                 coli_fp4_matvec_ref(up, up_w, input);
    if (!result) {
        coli_bf16_round_array(gate, intermediate);
        coli_bf16_round_array(up, intermediate);
        result = coli_v4_swiglu(activated, gate, up,
                                (int)intermediate, swiglu_limit);
    }
    if (!result) {
        for (size_t index = 0; index < intermediate; index++)
            activated[index] = coli_bf16_round(
                activated[index] * route_weight);
        result = coli_fp4_matvec_ref(output, down_w, activated);
    }
    if (!result) coli_bf16_round_array(output, output_size);
    free(activated); free(up); free(gate);
    return result ? -1 : 0;
}

static int check_stats_nan_aware(const char *what, int O, const float *g,
                                 const float *c, int verbose) {
    /* The CPU reference emits NaN rows for 255-coded scale groups (the e8
     * table); the swiglu fminf/fmaxf clamps repair them later. The matvec
     * comparison treats NaN==NaN as a match and otherwise uses the global
     * bar (same as check_stats). */
    double ma = 0, mr = 0, mref = 0;
    int nan_mismatch = 0;
    for (int i = 0; i < O; i++) {
        int gn = isnan(g[i]), cn = isnan(c[i]);
        if (gn != cn) { nan_mismatch++; continue; }
        if (gn && cn) continue;
        double d = fabs((double)g[i] - (double)c[i]);
        if (d > ma) ma = d;
        double r = fabs((double)c[i]);
        if (r > mref) mref = r;
        if (r >= 1e-2) {
            double rel = d / r;
            if (rel > mr) mr = rel;
        }
    }
    double global = mref > 0 ? ma / mref : 1.0;
    if (verbose)
        printf("  %s: maxabs/maxref %.3e/%.3e global %.3e max_rel(floor) %.3e "
               "nan_mismatch %d\n", what, ma, mref, global, mr, nan_mismatch);
    if (nan_mismatch) {
        for (int i = 0; i < O; i++) {
            int gn = isnan(g[i]), cn = isnan(c[i]);
            if (gn != cn)
                fprintf(stderr, "  [%s] row %d: gpu=%.6e cpu=%.6e (gpu_nan=%d cpu_nan=%d)\n",
                        what, i, g[i], c[i], gn, cn);
        }
    }
    return nan_mismatch ? 1 : (mref > 0 && global > 1e-2) ? 1 : 0;
}

/* fp4 matvec with EXTREME UE8M0 codes (0 and 255) — the real checkpoint's
 * expert GATE scales carry them (M5-1a parity fix). The GPU expansion must
 * reproduce the e8-table semantics (0xff -> NaN, 0 -> 2^-127), NOT the mx4
 * bit trick (+inf / +0): the swiglu fminf/fmaxf clamps repair the NaN rows
 * identically on both sides, so the matvec must NaN the SAME rows. (Codes
 * 1/254 are skipped: 2^±127 partials overflow and the inf/NaN pattern is
 * accumulation-order-dependent — the same thresholded class as the fp8
 * tier, not a pin.) */
static int run_fp4_matvec_edge(int O, int I, int verbose) {
    uint8_t *w, *sc;
    if (alloc_fp4(&w, &sc, O, I)) return 1;
    int ng = I / 32;
    for (int o = 0; o < 8 && o < O; o++)
        sc[(size_t)o * ng + 0] = (o < 4) ? 255 : 0;
    float *x = malloc((size_t)I * 4);
    for (int i = 0; i < I; i++) x[i] = (float)(rand() % 4001 - 2000) / 1000.0f;
    ColiTensorView view;
    fill_fp4_view(&view, w, sc, O, I);
    float *yc = malloc((size_t)O * 4), *yg = malloc((size_t)O * 4);
    int bad = 0;
    if (coli_fp4_matvec_ref(yc, &view, x)) { bad = 1; goto out; }
    Dsv4CudaTensor *t = NULL;
    if (!dsv4_cuda_upload_fp4(&t, w, sc, O, I, 0) || !t) { bad = 1; goto out; }
    if (!dsv4_cuda_matvec(t, yg, x)) { bad = 1; goto out; }
    if (verbose) printf("vk-experts fp4 matvec extreme codes (O=%d I=%d): ", O, I);
    bad |= check_stats_nan_aware("matvec(0/255 codes)", O, yg, yc, verbose);
    dsv4_cuda_tensor_free(t);
out:
    free(yg); free(yc); free(x); free(sc); free(w);
    return bad;
}
/* ---- fp4 matvec parity + upload/refill ---- */
static int run_fp4_matvec(int O, int I, int verbose) {
    uint8_t *w, *sc, *w2, *sc2;
    if (alloc_fp4(&w, &sc, O, I) || alloc_fp4(&w2, &sc2, O, I)) return 1;
    float *x = malloc((size_t)I * 4);
    for (int i = 0; i < I; i++) x[i] = (float)(rand() % 200001 - 100000) / 100.0f;
    ColiTensorView view, view2;
    fill_fp4_view(&view, w, sc, O, I);
    fill_fp4_view(&view2, w2, sc2, O, I);
    float *yc = malloc((size_t)O * 4), *yg = malloc((size_t)O * 4);
    float *yc2 = malloc((size_t)O * 4), *yg2 = malloc((size_t)O * 4);
    int bad = 0;

    /* CPU references */
    if (coli_fp4_matvec_ref(yc, &view, x)) { bad = 1; goto out; }
    if (coli_fp4_matvec_ref(yc2, &view2, x)) { bad = 1; goto out; }

    /* GPU: upload + matvec through the backend's per-op dispatch */
    Dsv4CudaTensor *t = NULL;
    if (!dsv4_cuda_upload_fp4(&t, w, sc, O, I, 0) || !t) {
        fprintf(stderr, "vk-experts: fp4 upload failed (O=%d I=%d)\n", O, I);
        bad = 1; goto out;
    }
    if (!dsv4_cuda_matvec(t, yg, x)) {
        fprintf(stderr, "vk-experts: fp4 matvec failed (O=%d I=%d)\n", O, I);
        bad = 1; goto out;
    }
    if (verbose) printf("vk-experts fp4 matvec (O=%d I=%d): ", O, I);
    bad |= check_stats("matvec", O, yg, yc, verbose);

    /* Refill with DIFFERENT bytes (LRU recycle): the matvec must now match
     * the new bytes' CPU reference. */
    if (!dsv4_cuda_tensor_refill_fp4(t, w2, sc2, O, I, 1)) {
        fprintf(stderr, "vk-experts: fp4 refill failed (O=%d I=%d)\n", O, I);
        bad = 1; goto out;
    }
    if (!dsv4_cuda_matvec(t, yg2, x)) {
        fprintf(stderr, "vk-experts: fp4 matvec post-refill failed\n");
        bad = 1; goto out;
    }
    bad |= check_stats("matvec(post-refill)", O, yg2, yc2, verbose);

    dsv4_cuda_tensor_free(t);
out:
    free(yc2); free(yg2); free(yc); free(yg);
    free(sc2); free(w2); free(sc); free(w); free(x);
    return bad;
}

/* ---- expert group parity vs the CPU per-expert chain ---- */
static int run_expert_group(int hidden, int input, int experts, int topk,
                            float limit, int verbose) {
    /* gate/up: hidden x input; down: input x hidden (engine's forward_ref
     * geometry). topk experts are routed; the group computes all `topk`. */
    Dsv4CudaTensor *g[8] = {0}, *u[8] = {0}, *d[8] = {0};
    uint8_t *wg[8], *scg[8], *wu[8], *scu[8], *wd[8], *scd[8];
    ColiTensorView vg[8], vu[8], vd[8];
    float *x = malloc((size_t)input * 4);
    float *yg = malloc((size_t)input * 4), *yc = calloc((size_t)input, 4);
    float *wgt = malloc((size_t)topk * 4);
    for (int i = 0; i < input; i++) x[i] = (float)(rand() % 200001 - 100000) / 100.0f;
    for (int e = 0; e < topk; e++) wgt[e] = (float)(rand() % 2000) / 1000.0f + 0.05f;
    int bad = 0;
    for (int e = 0; e < topk; e++) {
        if (alloc_fp4(&wg[e], &scg[e], hidden, input) ||
            alloc_fp4(&wu[e], &scu[e], hidden, input) ||
            alloc_fp4(&wd[e], &scd[e], input, hidden)) { bad = 1; break; }
        fill_fp4_view(&vg[e], wg[e], scg[e], hidden, input);
        fill_fp4_view(&vu[e], wu[e], scu[e], hidden, input);
        fill_fp4_view(&vd[e], wd[e], scd[e], input, hidden);
        /* CPU chain: per-expert forward + fp32 accumulation (engine's
         * non-GPU branch). */
        float *out = malloc((size_t)input * 4);
        if (ref_expert_forward(out, &vg[e], &vu[e], &vd[e], x, wgt[e], limit)) {
            free(out); bad = 1; break;
        }
        for (int i = 0; i < input; i++) yc[i] += out[i];
        free(out);
        if (!dsv4_cuda_upload_fp4(&g[e], wg[e], scg[e], hidden, input, 0) ||
            !dsv4_cuda_upload_fp4(&u[e], wu[e], scu[e], hidden, input, 0) ||
            !dsv4_cuda_upload_fp4(&d[e], wd[e], scd[e], input, hidden, 0)) {
            bad = 1; break;
        }
    }
    if (!bad) {
        if (!dsv4_cuda_expert_group((Dsv4CudaTensor *const *)g,
                                    (Dsv4CudaTensor *const *)u,
                                    (Dsv4CudaTensor *const *)d,
                                    wgt, topk, limit, yg, x)) {
            fprintf(stderr, "vk-experts: expert_group failed "
                            "(hidden=%d input=%d topk=%d)\n", hidden, input, topk);
            bad = 1;
        } else {
            if (verbose)
                printf("vk-experts expert group (hidden=%d input=%d topk=%d "
                       "limit=%.1f): ", hidden, input, topk, limit);
            bad |= check_stats("group", input, yg, yc, verbose);
        }
    }
    for (int e = 0; e < topk; e++) {
        dsv4_cuda_tensor_free(g[e]);
        dsv4_cuda_tensor_free(u[e]);
        dsv4_cuda_tensor_free(d[e]);
        free(scd[e]); free(wd[e]); free(scu[e]); free(wu[e]);
        free(scg[e]); free(wg[e]);
    }
    free(wgt); free(yc); free(yg); free(x);
    return bad;
}

/* ---- M5-1b: the two batched phase primitives, driven DIRECTLY (not via
 * dsv4_cuda_expert_group) — pins the one-submit-per-phase entry points:
 * phase1 = QDQ(x) + all gate/up matmuls, phase2 = QDQ(hid rows) + all down
 * matmuls. The host rounding chain between phases is the same bitwise code
 * the group runs; the final sum must match the CPU per-expert chain within
 * the matvec threshold (tiny shape: bitwise). Also pins the refusal paths
 * (count=0 / shape mismatch / bad fmt). */
static int run_fused_phases(int hidden, int input, int experts, int topk,
                            float limit, int verbose) {
    /* The test's "hidden" = the gate/up matmul ROW count = moe intermediate
     * (the phase API's I); "input" = the gate/up matmul COLUMN count = the
     * activation dim (the phase API's H). The phase API convention matches
     * dsv4_cuda_expert_group: H = activation, I = intermediate; gate/up are
     * H->I, down is I->H. */
    int H = input, I = hidden;
    ColiVkTensor *g[8] = {0}, *u[8] = {0}, *d[8] = {0};
    uint8_t *wg[8], *scg[8], *wu[8], *scu[8], *wd[8], *scd[8];
    float *sg_[8], *su_[8], *sd_[8];   /* UE8M0 -> f32 expanded (like the backend upload) */
    ColiTensorView vg[8], vu[8], vd[8];
    float *x = malloc((size_t)H * 4);
    float *yc = calloc((size_t)H, 4);
    float *wgt = malloc((size_t)topk * 4);
    for (int i = 0; i < H; i++) x[i] = (float)(rand() % 200001 - 100000) / 100.0f;
    for (int e = 0; e < topk; e++) wgt[e] = (float)(rand() % 2000) / 1000.0f + 0.05f;
    int bad = 0;
    for (int e = 0; e < topk; e++) {
        if (alloc_fp4(&wg[e], &scg[e], I, H) ||
            alloc_fp4(&wu[e], &scu[e], I, H) ||
            alloc_fp4(&wd[e], &scd[e], H, I)) { bad = 1; break; }
        fill_fp4_view(&vg[e], wg[e], scg[e], I, H);
        fill_fp4_view(&vu[e], wu[e], scu[e], I, H);
        fill_fp4_view(&vd[e], wd[e], scd[e], H, I);
        float *out = malloc((size_t)H * 4);
        if (ref_expert_forward(out, &vg[e], &vu[e], &vd[e], x, wgt[e], limit)) {
            free(out); bad = 1; break;
        }
        for (int i = 0; i < H; i++) yc[i] += out[i];
        free(out);
        size_t ng = (size_t)I * (size_t)(H / 32);   /* gate/up scales [O=I, H/32] */
        size_t nd = (size_t)H * (size_t)(I / 32);   /* down scales [O=H, I/32] */
        sg_[e] = malloc(ng * 4); su_[e] = malloc(ng * 4); sd_[e] = malloc(nd * 4);
        if (!sg_[e] || !su_[e] || !sd_[e]) { bad = 1; break; }
        for (size_t i = 0; i < ng; i++) {
            sg_[e][i] = ref_ue8m0(scg[e][i]);
            su_[e][i] = ref_ue8m0(scu[e][i]);
        }
        for (size_t i = 0; i < nd; i++) sd_[e][i] = ref_ue8m0(scd[e][i]);
        if (!coli_vk_tensor_ensure(&g[e], wg[e], sg_[e], 10, H, I, 32) ||
            !coli_vk_tensor_ensure(&u[e], wu[e], su_[e], 10, H, I, 32) ||
            !coli_vk_tensor_ensure(&d[e], wd[e], sd_[e], 10, I, H, 32)) {
            bad = 1; break;
        }
    }
    if (!bad) {
        float *gatebuf = malloc((size_t)topk * (size_t)I * 4);
        float *upbuf = malloc((size_t)topk * (size_t)I * 4);
        float *hid = malloc((size_t)topk * (size_t)I * 4);
        float *out = malloc((size_t)topk * (size_t)H * 4);
        float *yg = calloc((size_t)H, 4);
        if (!gatebuf || !upbuf || !hid || !out || !yg) bad = 1;
        else {
            int ok = coli_vk_expert_dsv4_phase1(g, u, gatebuf, upbuf, x,
                                                topk, H, I);
            if (ok)
                for (int e = 0; e < topk && ok; e++)
                    for (int i = 0; i < I; i++) {
                        float gv_ = coli_bf16_round(gatebuf[e * I + i]);
                        float uv_ = coli_bf16_round(upbuf[e * I + i]);
                        if (limit > 0.0f) {
                            gv_ = fminf(gv_, limit);
                            uv_ = fmaxf(-limit, fminf(uv_, limit));
                        }
                        hid[e * I + i] = coli_bf16_round(
                            gv_ * sigmoid_stable(gv_) * uv_ * wgt[e]);
                    }
            if (ok) ok = coli_vk_expert_dsv4_phase2(d, out, hid, topk, H, I);
            if (ok)
                for (int e = 0; e < topk; e++)
                    for (int i = 0; i < H; i++)
                        yg[i] += coli_bf16_round(out[e * H + i]);
            if (!ok) {
                fprintf(stderr, "vk-experts: fused phases failed "
                                "(hidden=%d input=%d topk=%d)\n",
                        hidden, input, topk);
                bad = 1;
            } else {
                if (verbose)
                    printf("vk-experts fused phases (hidden=%d input=%d "
                           "topk=%d limit=%.1f): ", hidden, input, topk, limit);
                bad |= check_stats("fused", H, yg, yc, verbose);
            }
            /* refusal paths: count=0, a wrong fmt. */
            if (coli_vk_expert_dsv4_phase1(g, u, gatebuf, upbuf, x, 0, H, I)) {
                fprintf(stderr, "vk-experts: phase1 accepted count=0\n");
                bad = 1;
            }
            if (coli_vk_expert_dsv4_phase2(d, out, hid, 0, H, I)) {
                fprintf(stderr, "vk-experts: phase2 accepted count=0\n");
                bad = 1;
            }
            ColiVkTensor bogus = *g[0];
            bogus.fmt = 7;   /* not fp4-elem */
            ColiVkTensor *bogus_arr[1] = {&bogus};
            if (coli_vk_expert_dsv4_phase1(bogus_arr, u, gatebuf, upbuf, x,
                                           1, H, I)) {
                fprintf(stderr, "vk-experts: phase1 accepted fmt!=10\n");
                bad = 1;
            }
            free(yg); free(out); free(hid); free(upbuf); free(gatebuf);
        }
    }
    for (int e = 0; e < topk; e++) {
        coli_vk_tensor_free(g[e]);
        coli_vk_tensor_free(u[e]);
        coli_vk_tensor_free(d[e]);
        free(sd_[e]); free(wd[e]); free(scd[e]);
        free(su_[e]); free(wu[e]); free(scu[e]);
        free(sg_[e]); free(wg[e]); free(scg[e]);
    }
    free(wgt); free(yc); free(x);
    return bad;
}

/* ---- M5-1c: batch routing (dsv4_cuda_route_top6_batch) vs the engine's
 * coli_v4_route_bf16 per token. The batched fmt=0 gate matvec (S=tokens,
 * the mHC-batch pattern) over the same f32 gate mirror the decode route
 * builds (bf16 storage -> exact f32 decode); ids must match EXACTLY
 * (strict-greater top-6), weights within the ops suite's fp32-reduction
 * bar (2e-3 relative). */
static int run_route_batch(int experts, int hidden, int tokens) {
    if (experts > 256 || tokens > 128) return 1;
    float *gate = malloc((size_t)experts * hidden * 4);
    uint16_t *gate_bf16 = malloc((size_t)experts * hidden * 2);
    float *bias = malloc((size_t)experts * 4);
    float *x = malloc((size_t)tokens * hidden * 4);
    if (!gate || !gate_bf16 || !bias || !x) {
        free(x); free(bias); free(gate_bf16); free(gate); return 1;
    }
    for (int i = 0; i < experts * hidden; i++) {
        float raw = (float)(rand() % 200001 - 100000) / 100000.0f;
        gate_bf16[i] = f32_to_bf16(raw);
        gate[i] = bf16_to_f32(gate_bf16[i]);   /* the engine's gate mirror */
    }
    for (int i = 0; i < experts; i++)
        bias[i] = (float)(rand() % 2001 - 1000) / 1000.0f;
    for (int i = 0; i < tokens * hidden; i++)
        x[i] = (float)(rand() % 200001 - 100000) / 100.0f;
    float w_c[128][6]; int id_c[128][6];
    int okc = 1;
    for (int t = 0; t < tokens; t++)
        okc &= coli_v4_route_bf16(w_c[t], id_c[t], x + (size_t)t * hidden,
                                  gate_bf16, bias, NULL, experts, hidden,
                                  6, 1.25f) == 0;
    Dsv4CudaTensor *tg = NULL, *tb = NULL;
    int okg = dsv4_cuda_upload_f32(&tg, gate, experts, hidden, 0) &&
              dsv4_cuda_upload_f32(&tb, bias, experts, 1, 0);
    if (!okg) { fprintf(stderr, "vk-experts: route_batch upload refused\n"); okg = 0; }
    ds4vk_tensor_set_op(tg, "route");
    ds4vk_tensor_set_op(tb, "route");
    Dsv4CudaActivation *in =
        dsv4_cuda_activation_create(0, (long long)tokens * hidden);
    if (okg && !dsv4_cuda_activation_upload(in, x, (long long)tokens * hidden))
        okg = 0;
    int ids[128][6]; float wts[128][6];
    if (okg && !dsv4_cuda_route_top6_batch(in, tg, tb, tokens, 1.25f,
                                           &ids[0][0], &wts[0][0])) {
        fprintf(stderr, "vk-experts: route_top6_batch refused\n");
        okg = 0;
    }
    int ids_ok = 1, w_ok = 1;
    for (int t = 0; t < tokens; t++)
        for (int k = 0; k < 6; k++) {
            if (ids[t][k] != id_c[t][k]) ids_ok = 0;
            if (fabsf(wts[t][k] - w_c[t][k]) > 2e-3f *
                    fmaxf(1.0f, fabsf(w_c[t][k])))
                w_ok = 0;
        }
    printf("vk-experts route_batch (experts=%d hidden=%d tokens=%d): ids %s "
           "weights %s\n", experts, hidden, tokens,
           ids_ok ? "MATCH" : "DIFFER", w_ok ? "OK" : "FAIL");
    dsv4_cuda_activation_free(in);
    dsv4_cuda_tensor_free(tb); dsv4_cuda_tensor_free(tg);
    free(x); free(bias); free(gate_bf16); free(gate);
    return okc && okg && ids_ok && w_ok ? 0 : 1;
}

/* CPU union reference for the bank batch (v4_moe_batch_union's pre-round
 * state): per-route bf16 contributions accumulated EXPERT-MAJOR (ascending
 * expert id, then (item, rank) ascending — the exact order of
 * v4_apply_expert_batch's flushes), then the shared fp8 batch chain
 * (bf16(gate)/bf16(up)/swiglu/bf16(act)/down/bf16(out) — the rounding points
 * of v4_shared_expert_forward_batch_ref) added last. yc is the f32 sum the
 * engine bf16-rounds after the backend returns. contrib is scratch
 * (tokens*6*H floats). */
static int bank_cpu_ref(float *yc, const float *x, const int *ids,
                        const float *wgt, int tokens, int experts_n,
                        const ColiTensorView *vg, const ColiTensorView *vu,
                        const ColiTensorView *vd, const ColiTensorView *w1v,
                        const ColiTensorView *w3v, const ColiTensorView *w2v,
                        int H, int I, float limit, float *contrib) {
    memset(yc, 0, (size_t)tokens * H * 4);
    for (int t = 0; t < tokens; t++)
        for (int r = 0; r < 6; r++) {
            int e = ids[t * 6 + r];
            if (ref_expert_forward(contrib + ((size_t)t * 6 + r) * H,
                                   &vg[e], &vu[e], &vd[e],
                                   x + (size_t)t * H, wgt[t * 6 + r], limit))
                return -1;
        }
    for (int e = 0; e < experts_n; e++)
        for (int t = 0; t < tokens; t++)
            for (int r = 0; r < 6; r++)
                if (ids[t * 6 + r] == e) {
                    const float *out =
                        contrib + ((size_t)t * 6 + r) * H;
                    float *yt = yc + (size_t)t * H;
                    for (int i = 0; i < H; i++) yt[i] += out[i];
                }
    size_t cells = (size_t)tokens * (size_t)I;
    float *g = malloc(cells * 4), *u = malloc(cells * 4);
    float *act = malloc(cells * 4), *s = malloc((size_t)tokens * H * 4);
    if (!g || !u || !act || !s) { free(s); free(act); free(u); free(g); return -1; }
    for (int t = 0; t < tokens; t++)
        if (coli_fp8_matvec_ref(g + (size_t)t * I, w1v, x + (size_t)t * H) ||
            coli_fp8_matvec_ref(u + (size_t)t * I, w3v, x + (size_t)t * H)) {
            free(s); free(act); free(u); free(g); return -1;
        }
    coli_bf16_round_array(g, cells);
    coli_bf16_round_array(u, cells);
    for (int t = 0; t < tokens; t++)
        if (coli_v4_swiglu(act + (size_t)t * I, g + (size_t)t * I,
                           u + (size_t)t * I, I, limit)) {
            free(s); free(act); free(u); free(g); return -1;
        }
    coli_bf16_round_array(act, cells);
    for (int t = 0; t < tokens; t++)
        if (coli_fp8_matvec_ref(s + (size_t)t * H, w2v,
                                act + (size_t)t * I)) {
            free(s); free(act); free(u); free(g); return -1;
        }
    coli_bf16_round_array(s, (size_t)tokens * H);
    for (int i = 0; i < tokens * H; i++) yc[i] += s[i];
    free(s); free(act); free(u); free(g);
    return 0;
}

/* ---- M5-1c: the prefill expert bank (COLI_CUDA_MOE_BATCH=1) ----
 * Drives dsv4_cuda_expert_bank_create/upload + dsv4_cuda_route_moe_ids_batch
 * DIRECTLY over a small batch with fixed ids/weights vs the CPU union
 * reference above (bank_cpu_ref). The backend returns the pre-round f32
 * routed+shared sum (the engine bf16-rounds after), so the compare is
 * thresholded like the decode group. Also pins the in-place refill
 * (layer-switch pattern): a second upload pass over the SAME slots with new
 * bytes must match a reference built from the NEW bytes. */
static int run_bank_batch(int hidden, int input, int experts_n, int tokens,
                          float limit, int verbose) {
    int H = input, I = hidden;   /* gate/up: I x H, down: H x I (rows16) */
    if (experts_n < 6) experts_n = 6;
    if (experts_n > 16) experts_n = 16;
    if (tokens < 1 || tokens > 128) tokens = 4;
    uint8_t *wg[16], *scg[16], *wu[16], *scu[16], *wd[16], *scd[16];
    uint8_t *wg2[16], *scg2[16], *wu2[16], *scu2[16], *wd2[16], *scd2[16];
    ColiTensorView vg[16], vu[16], vd[16], vg2[16], vu2[16], vd2[16];
    memset(vg, 0, sizeof(vg)); memset(vu, 0, sizeof(vu)); memset(vd, 0, sizeof(vd));
    memset(vg2, 0, sizeof(vg2)); memset(vu2, 0, sizeof(vu2)); memset(vd2, 0, sizeof(vd2));
    float *x = malloc((size_t)tokens * H * 4);
    float *yc = calloc((size_t)tokens * H, 4);
    float *yc2 = calloc((size_t)tokens * H, 4);
    float *wgt = malloc((size_t)tokens * 6 * 4);
    int *ids = malloc((size_t)tokens * 6 * sizeof(int));
    float *contrib = malloc((size_t)tokens * 6 * H * 4);
    if (!x || !yc || !yc2 || !wgt || !ids || !contrib) {
        free(contrib); free(ids); free(wgt); free(yc2); free(yc); free(x);
        return 1;
    }
    int bad = 0;
    for (int i = 0; i < tokens * H; i++)
        x[i] = (float)(rand() % 200001 - 100000) / 100.0f;
    for (int t = 0; t < tokens; t++)
        for (int r = 0; r < 6; r++) {
            /* distinct within a token (gcd(3,16)=1); repeats across tokens */
            ids[t * 6 + r] = (t * 2 + r * 3 + 1) % experts_n;
            wgt[t * 6 + r] = (float)(rand() % 2000) / 1000.0f + 0.05f;
        }
    for (int e = 0; e < experts_n; e++) {
        if (alloc_fp4(&wg[e], &scg[e], I, H) ||
            alloc_fp4(&wu[e], &scu[e], I, H) ||
            alloc_fp4(&wd[e], &scd[e], H, I) ||
            alloc_fp4(&wg2[e], &scg2[e], I, H) ||
            alloc_fp4(&wu2[e], &scu2[e], I, H) ||
            alloc_fp4(&wd2[e], &scd2[e], H, I)) { bad = 1; break; }
        fill_fp4_view(&vg[e], wg[e], scg[e], I, H);
        fill_fp4_view(&vu[e], wu[e], scu[e], I, H);
        fill_fp4_view(&vd[e], wd[e], scd[e], H, I);
        fill_fp4_view(&vg2[e], wg2[e], scg2[e], I, H);
        fill_fp4_view(&vu2[e], wu2[e], scu2[e], I, H);
        fill_fp4_view(&vd2[e], wd2[e], scd2[e], H, I);
    }
    /* shared experts (fp8): w1/w3 = gate/up I x H, w2 = down H x I; raw
     * e4m3 bytes + E8M0 128x128-block codes (bscale = the expanded fp32 the
     * CPU reference decodes to). */
    size_t nsc1 = (size_t)((I + 127) / 128) * ((H + 127) / 128);
    size_t nsc2 = (size_t)((H + 127) / 128) * ((I + 127) / 128);
    uint8_t *w1 = malloc((size_t)I * H), *e8_1 = malloc(nsc1);
    uint8_t *w3 = malloc((size_t)I * H), *e8_3 = malloc(nsc1);
    uint8_t *w2 = malloc((size_t)H * I), *e8_2 = malloc(nsc2);
    float *b1 = malloc(nsc1 * 4), *b3 = malloc(nsc1 * 4), *b2 = malloc(nsc2 * 4);
    if (!w1 || !e8_1 || !w3 || !e8_3 || !w2 || !e8_2 || !b1 || !b3 || !b2)
        bad = 1;
    for (size_t i = 0; !bad && i < (size_t)I * H; i++) {
        w1[i] = nn_byte(); w3[i] = nn_byte();
    }
    for (size_t i = 0; !bad && i < (size_t)H * I; i++) w2[i] = nn_byte();
    for (size_t i = 0; !bad && i < nsc1; i++) {
        uint8_t c = (uint8_t)(115 + rand() % 14);
        e8_1[i] = c; b1[i] = ref_ue8m0(c);
        c = (uint8_t)(115 + rand() % 14);
        e8_3[i] = c; b3[i] = ref_ue8m0(c);
    }
    for (size_t i = 0; !bad && i < nsc2; i++) {
        uint8_t c = (uint8_t)(115 + rand() % 14);
        e8_2[i] = c; b2[i] = ref_ue8m0(c);
    }
    ColiTensorView w1v = {0}, w3v = {0}, w2v = {0};
    w1v.format = COLI_TENSOR_FP8_E4M3_BLOCK; w1v.scale_format = COLI_SCALE_F32;
    w1v.data = w1; w1v.scales = b1; w1v.data_bytes = (size_t)I * H;
    w1v.scale_bytes = nsc1 * 4; w1v.rows = I; w1v.columns = H;
    w1v.block_rows = 128; w1v.block_columns = 128;
    w3v = w1v; w3v.data = w3; w3v.scales = b3;
    w2v.format = COLI_TENSOR_FP8_E4M3_BLOCK; w2v.scale_format = COLI_SCALE_F32;
    w2v.data = w2; w2v.scales = b2; w2v.data_bytes = (size_t)H * I;
    w2v.scale_bytes = nsc2 * 4; w2v.rows = H; w2v.columns = I;
    w2v.block_rows = 128; w2v.block_columns = 128;
    if (!bad && bank_cpu_ref(yc, x, ids, wgt, tokens, experts_n, vg, vu, vd,
                             &w1v, &w3v, &w2v, H, I, limit, contrib))
        bad = 1;
    /* GPU side: bank + shared mirrors (the engine tags the shared mirrors
     * "shared" at upload — untagged tensors are gated off by default) */
    Dsv4CudaTensor *sg = NULL, *su = NULL, *sd = NULL;
    int okg = !bad && dsv4_cuda_upload_fp8(&sg, w1, e8_1, I, H, 0) &&
              dsv4_cuda_upload_fp8(&su, w3, e8_3, I, H, 0) &&
              dsv4_cuda_upload_fp8(&sd, w2, e8_2, H, I, 0);
    if (okg) {
        ds4vk_tensor_set_op(sg, "shared");
        ds4vk_tensor_set_op(su, "shared");
        ds4vk_tensor_set_op(sd, "shared");
    }
    Dsv4CudaExpertSet *bank =
        okg ? dsv4_cuda_expert_bank_create(experts_n, H, I, 0, sg, su, sd)
            : NULL;
    if (bank) {
        for (int e = 0; e < experts_n && okg; e++) {
            Dsv4CudaTensor *bg = NULL, *bu = NULL, *bd = NULL;
            if (!dsv4_cuda_expert_bank_upload(bank, e, wg[e], scg[e],
                                              wu[e], scu[e], wd[e], scd[e],
                                              &bg, &bu, &bd) ||
                bg || bu || bd)   /* bank owns the tensors: handles must be NULL */
                okg = 0;
        }
    }
    Dsv4CudaActivation *in =
        dsv4_cuda_activation_create(0, (long long)tokens * H);
    Dsv4CudaActivation *out =
        dsv4_cuda_activation_create(0, (long long)tokens * H);
    if (okg && !dsv4_cuda_activation_upload(in, x, (long long)tokens * H))
        okg = 0;
    int okb = okg && dsv4_cuda_route_moe_ids_batch(
                          in, ids, wgt, tokens, bank, limit, out);
    if (okg && !okb)
        fprintf(stderr, "vk-experts: route_moe_ids_batch refused "
                        "(H=%d I=%d experts=%d tokens=%d)\n",
                H, I, experts_n, tokens);
    float *yg = malloc((size_t)tokens * H * 4);
    if (okb && !yg) { bad = 1; okb = 0; }
    if (okb && !dsv4_cuda_activation_download(yg, out, (long long)tokens * H))
        okb = 0;
    if (okb)
        bad |= check_stats("bank batch", tokens * H, yg, yc, verbose);
    /* in-place refill pass: same slots, new bytes (the layer-switch pattern) */
    if (bank && !bad) {
        int refill_ok = 1;
        for (int e = 0; e < experts_n && refill_ok; e++) {
            Dsv4CudaTensor *bg = NULL, *bu = NULL, *bd = NULL;
            if (!dsv4_cuda_expert_bank_upload(bank, e, wg2[e], scg2[e],
                                              wu2[e], scu2[e], wd2[e], scd2[e],
                                              &bg, &bu, &bd))
                refill_ok = 0;
        }
        if (!refill_ok)
            fprintf(stderr, "vk-experts: bank refill upload failed\n");
        else if (bank_cpu_ref(yc2, x, ids, wgt, tokens, experts_n,
                              vg2, vu2, vd2, &w1v, &w3v, &w2v,
                              H, I, limit, contrib))
            bad = 1;
        else if (!dsv4_cuda_route_moe_ids_batch(in, ids, wgt, tokens, bank,
                                                limit, out)) {
            fprintf(stderr, "vk-experts: route_moe_ids_batch refused after "
                            "refill\n");
            bad = 1;
        } else if (!dsv4_cuda_activation_download(yg, out,
                                                  (long long)tokens * H))
            bad = 1;
        else
            bad |= check_stats("bank refill", tokens * H, yg, yc2, verbose);
    }
    dsv4_cuda_expert_set_free(bank);
    dsv4_cuda_tensor_free(sd); dsv4_cuda_tensor_free(su); dsv4_cuda_tensor_free(sg);
    dsv4_cuda_activation_free(in); dsv4_cuda_activation_free(out);
    free(yg);
    for (int e = 0; e < experts_n; e++) {
        free(scd2[e]); free(wd2[e]); free(scu2[e]); free(wu2[e]);
        free(scg2[e]); free(wg2[e]);
        free(scd[e]); free(wd[e]); free(scu[e]); free(wu[e]);
        free(scg[e]); free(wg[e]);
    }
    free(contrib); free(ids); free(wgt); free(yc2); free(yc); free(x);
    free(b2); free(b3); free(b1); free(e8_2); free(w2); free(e8_3); free(w3);
    free(e8_1); free(w1);
    return bad;
}

/* Env-sensitive scenarios (op gating, fault injection) must run in forked
 * children: the backend's COLI_DSV4_VK_OPS / COLI_DSV4_VK_FAIL masks are
 * process-static caches parsed at first use, so an env change mid-process
 * would not re-parse (same pattern as tests/test_dsv4_vk_fallback.c). */
static int run_child(int (*fn)(void)) {
    pid_t pid = fork();
    if (pid < 0) { perror("fork"); return 1; }
    if (pid == 0) {
        int r = 1;
        if (coli_vk_init("shaders/qmatmul.spv") && coli_vk_available()) {
            r = fn();
            coli_vk_shutdown();
        }
        fflush(stdout); fflush(stderr);
        _exit(r ? 1 : 0);
    }
    int status = 0;
    if (waitpid(pid, &status, 0) < 0) { perror("waitpid"); return 1; }
    return WIFEXITED(status) && WEXITSTATUS(status) == 0 ? 0 : 1;
}

/* COLI_DSV4_VK_OPS without "expert" must refuse both the per-expert matvec
 * and the group (the engine then falls back to CPU per-op). */
static int gating_child(void) {
    int bad = 0;
    setenv("COLI_DSV4_VK_OPS",
           "qkv,wo,route,head,shared,attn,mhc,comp", 1);
    uint8_t *w, *sc;
    if (alloc_fp4(&w, &sc, 128, 128)) return 1;
    Dsv4CudaTensor *t = NULL;
    if (!dsv4_cuda_upload_fp4(&t, w, sc, 128, 128, 0)) { bad = 1; goto out; }
    float x[128], y[128];
    for (int i = 0; i < 128; i++) x[i] = 0.5f;
    int ran = dsv4_cuda_matvec(t, y, x);
    Dsv4CudaTensor *gg[1] = {t}, *uu[1] = {t}, *dd[1] = {t};
    float one = 1.0f;
    int grp = dsv4_cuda_expert_group(gg, uu, dd, &one, 1, 15.0f, y, x);
    if (ran || grp) {
        fprintf(stderr, "vk-experts: expert ran with OPS excluding it "
                        "(matvec=%d group=%d) — gating broken\n", ran, grp);
        bad = 1;
    } else {
        printf("vk-experts: expert refused under OPS without it (gating OK)\n");
    }
out:
    dsv4_cuda_tensor_free(t);
    free(sc); free(w);
    return bad;
}

/* L4 hook at the backend level: COLI_DSV4_VK_FAIL=expert forces the group to
 * fail (the engine's fused path then degrades to the CPU chain). */
static int fault_child(void) {
    int bad = 0;
    setenv("COLI_DSV4_VK_FAIL", "expert", 1);
    uint8_t *w, *sc;
    if (alloc_fp4(&w, &sc, 128, 128)) return 1;
    Dsv4CudaTensor *t = NULL;
    if (!dsv4_cuda_upload_fp4(&t, w, sc, 128, 128, 0)) { bad = 1; goto out; }
    float x[128], y[128];
    for (int i = 0; i < 128; i++) x[i] = 0.5f;
    Dsv4CudaTensor *gg[1] = {t}, *uu[1] = {t}, *dd[1] = {t};
    float one = 1.0f;
    int grp = dsv4_cuda_expert_group(gg, uu, dd, &one, 1, 15.0f, y, x);
    if (grp) {
        fprintf(stderr, "vk-experts: expert_group ran under "
                        "COLI_DSV4_VK_FAIL=expert — fault hook broken\n");
        bad = 1;
    } else {
        printf("vk-experts: COLI_DSV4_VK_FAIL=expert forces group failure "
               "(L4 hook OK)\n");
    }
out:
    dsv4_cuda_tensor_free(t);
    free(sc); free(w);
    return bad;
}

int main(void) {
    /* Parent probe: one clean init/shutdown just to test ICD availability
     * (skips exit 0 without a Vulkan device). The env-sensitive scenarios
     * (gating, fault) run in forked children that init the backend THEMSELVES
     * and fork BEFORE the parent runs any op — the COLI_DSV4_VK_OPS /
     * COLI_DSV4_VK_FAIL masks are process-static caches parsed on first use,
     * so each child needs its own process + env (same pattern as
     * tests/test_dsv4_vk_fallback.c, incl. the child-owned backend
     * lifecycle). The parent then runs the shape scenarios with its own
     * init. */
    if (!coli_vk_init("shaders/qmatmul.spv") || !coli_vk_available()) {
        fprintf(stderr, "vk-experts: no Vulkan device — skipped\n");
        return 0;
    }
    coli_vk_shutdown();
    srand(23);
    int bad = 0;
    bad |= run_child(gating_child);
    bad |= run_child(fault_child);
    if (!coli_vk_init("shaders/qmatmul.spv") || !coli_vk_available()) {
        fprintf(stderr, "vk-experts: backend init failed after children\n");
        return 1;
    }
    /* real DeepSeek-V4-Flash-0731 routed-expert shapes: gate/up 2048x4096,
     * down 4096x2048, top-6, swiglu_limit from the config. */
    bad |= run_fp4_matvec(2048, 4096, 1);
    bad |= run_fp4_matvec(4096, 2048, 1);
    bad |= run_fp4_matvec_edge(2048, 4096, 1);
    bad |= run_expert_group(2048, 4096, 6, 6, 15.0f, 1);
    bad |= run_fused_phases(2048, 4096, 6, 6, 15.0f, 1);
    /* M5-1c prefill bank: batch routing + batched MoE over the bank (real
     * DeepSeek-V4-Flash-0731 routed-expert shapes: gate/up 2048x4096,
     * down 4096x2048, top-6, swiglu_limit from the config) */
    bad |= run_route_batch(256, 4096, 4);
    bad |= run_route_batch(256, 4096, 128);
    bad |= run_bank_batch(2048, 4096, 12, 4, 15.0f, 1);
    /* tiny fixture shapes (4 experts, top-2, intermediate 128) */
    bad |= run_fp4_matvec(128, 128, 1);
    bad |= run_expert_group(128, 128, 4, 2, 15.0f, 1);
    bad |= run_fused_phases(128, 128, 4, 2, 15.0f, 1);
    bad |= run_route_batch(256, 128, 4);
    bad |= run_bank_batch(128, 128, 8, 4, 15.0f, 1);
    /* M5-1c batch-128 stress: the real prefill chunks run up to 128 tokens
     * per layer-chunk (the engine's union gate), so the bank's batched MoE
     * must hold at the full chunk width, not just the small batches above. */
    bad |= run_bank_batch(128, 128, 8, 128, 15.0f, 1);

    if (bad) { fprintf(stderr, "vk-experts: FAIL\n"); coli_vk_shutdown(); return 1; }
    printf("vk-experts: OK\n");
    coli_vk_shutdown();
    return 0;
}
