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
    /* tiny fixture shapes (4 experts, top-2, intermediate 128) */
    bad |= run_fp4_matvec(128, 128, 1);
    bad |= run_expert_group(128, 128, 4, 2, 15.0f, 1);

    if (bad) { fprintf(stderr, "vk-experts: FAIL\n"); coli_vk_shutdown(); return 1; }
    printf("vk-experts: OK\n");
    coli_vk_shutdown();
    return 0;
}
