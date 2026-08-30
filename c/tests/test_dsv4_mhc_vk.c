/* M3c-1 L2 per-op parity for the dsv4 Vulkan tier (TEST.md §2 L2): mHC
 * pre/post (sinkhorn).
 * Template: tests/test_dsv4_vk_ops.c.
 *
 * Each GPU op is tested against the EXACT CPU function the engine calls on
 * the decode path:
 *   - dsv4_cuda_mhc_pre_norm vs normalized_hc_pre (deepseek_v4.c block unit):
 *     coli_v4_hc_pre (fn matvec + RMS inverse + sinkhorn pre/post/comb),
 *     coli_bf16_round_array(reduced), coli_v4_rmsnorm with the branch norm,
 *     coli_bf16_round_array(normalized). The VK backend runs ONLY the fn
 *     matvec on the GPU (the N x M*H f32 matmul, thresholded vs the CPU's
 *     sequential fp32 — the same contract as the CUDA tier's split
 *     accumulation); the RMS inverse, sinkhorn iterations, pre*residual sum,
 *     bf16 rounding points and norm rmsnorm are host-side sequential fp32 —
 *     BITWISE vs the CPU reference.
 *   - dsv4_cuda_mhc_post vs coli_v4_hc_post + coli_bf16_round_array: the
 *     post[j]*x[h] + comb[i*M+j]*residual[i*H+h] sum in the CPU's order,
 *     bf16-rounded at the same point.
 *
 * Oracles = the REAL engine units (COLI_V4_UNIT_MATH for hc_pre/hc_post/
 * rmsnorm, COLI_V4_UNIT_NATIVE_QUANT for bf16 round) linked without the
 * GPU-tier defines. The test sets COLI_DSV4_VK_OPS=mhc up front, so the mHC
 * group must be tagged and enabled for the ops to run at all — a broken tag
 * fails the test; a non-mhc op (route) must be refused under the same env.
 *
 * Skips (exit 0) without a Vulkan ICD, like the other vk tests.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "../backend_vulkan.h"
#include "../backend_vulkan_dsv4.h"

/* Real engine CPU references (amalgamation unit objects linked WITHOUT the
 * GPU-tier defines — the pure-CPU paths are what link, so these are oracles,
 * not replicas). */
extern int coli_v4_hc_pre(float *output, float *post, float *comb,
                          const float *input, const float *hc_fn,
                          const float scale[3], const float *base, int hc,
                          int dimension, int iterations, float norm_eps,
                          float hc_eps);
extern int coli_v4_hc_post(float *output, const float *branch,
                           const float *residual, const float *post,
                           const float *comb, int hc, int dimension);
extern int coli_v4_rmsnorm(float *output, const float *input,
                           const float *weight, int elements, float eps);
extern void coli_bf16_round_array(float *values, size_t count);

/* f32 -> bf16 (round-to-nearest-even) for the synthetic bf16 compressor
 * weights (the same helper the ops test uses). */
static uint16_t f32_to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    uint32_t rounded = (bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16;
    return (uint16_t)rounded;
}

/* The batch mHC ABI hardcodes the model config: hc_sinkhorn_iters = 20. */
enum { MHC_BATCH_ITERS = 20 };

static void cmp_stats(const float *a, const float *b, int n,
                      double *maxabs, double *maxref) {
    *maxabs = 0; *maxref = 0;
    for (int i = 0; i < n; i++) {
        double d = fabs((double)a[i] - (double)b[i]);
        if (d > *maxabs) *maxabs = d;
        double r = fabs((double)b[i]);
        if (r > *maxref) *maxref = r;
    }
}

/* One mHC pre (norm) + post round-trip at (M, H): GPU vs the engine CPU
 * chain. Returns 0 on pass. */
static int run_mhc(int M, int H, int seed_shift) {
    int N = 2 * M + M * M, MH = M * H;
    float *fn = malloc((size_t)N * MH * sizeof(*fn));
    float *scale = malloc(3 * sizeof(*scale));
    float *base = malloc((size_t)N * sizeof(*base));
    float *norm = malloc((size_t)H * sizeof(*norm));
    float *residual = malloc((size_t)MH * sizeof(*residual));
    float *branch = malloc((size_t)H * sizeof(*branch));
    if (!fn || !scale || !base || !norm || !residual || !branch) return 1;
    /* synthetic weights: small, non-degenerate (sigmoid/softmax in range) */
    for (int i = 0; i < N * MH; i++)
        fn[i] = (float)(rand() % 2001 - 1000) / 100000.0f;   /* ~1e-2 */
    scale[0] = (float)(rand() % 2000 + 100) / 100.0f;        /* ~1..20 */
    scale[1] = (float)(rand() % 2000 + 100) / 100.0f;
    scale[2] = (float)(rand() % 2000 + 100) / 100.0f;
    for (int i = 0; i < N; i++) base[i] = (float)(rand() % 2001 - 1000) / 100.0f;
    for (int i = 0; i < H; i++) norm[i] = (float)(rand() % 2000 + 100) / 1000.0f;
    for (int i = 0; i < MH; i++) residual[i] = (float)(rand() % 20001 - 10000) / 1000.0f;
    for (int i = 0; i < H; i++) branch[i] = (float)(rand() % 20001 - 10000) / 1000.0f;
    float rms_eps = 1e-6f, hc_eps = 1e-6f;
    int sink_iters = 3;

    /* ---- CPU oracle: the engine's normalized_hc_pre chain + hc_post ---- */
    float *reduced_c = malloc((size_t)H * sizeof(*reduced_c));
    float *normalized_c = malloc((size_t)H * sizeof(*normalized_c));
    float *post_c = malloc((size_t)M * sizeof(*post_c));
    float *comb_c = malloc((size_t)M * M * sizeof(*comb_c));
    float *out_c = malloc((size_t)MH * sizeof(*out_c));
    if (!reduced_c || !normalized_c || !post_c || !comb_c || !out_c) return 1;
    if (coli_v4_hc_pre(reduced_c, post_c, comb_c, residual, fn, scale, base,
                       M, H, sink_iters, rms_eps, hc_eps))
        return 1;
    coli_bf16_round_array(reduced_c, H);
    if (coli_v4_rmsnorm(normalized_c, reduced_c, norm, H, rms_eps)) return 1;
    coli_bf16_round_array(normalized_c, H);
    if (coli_v4_hc_post(out_c, branch, residual, post_c, comb_c, M, H)) return 1;
    coli_bf16_round_array(out_c, MH);

    /* ---- GPU: fn/scale/base/norm f32 mirrors + the two ABI entries ---- */
    Dsv4CudaTensor *tfn = NULL, *tsc = NULL, *tbs = NULL, *tnm = NULL;
    if (!dsv4_cuda_upload_f32(&tfn, fn, N, MH, 0) ||
        !dsv4_cuda_upload_f32(&tsc, scale, 3, 1, 0) ||
        !dsv4_cuda_upload_f32(&tbs, base, N, 1, 0) ||
        !dsv4_cuda_upload_f32(&tnm, norm, H, 1, 0)) {
        fprintf(stderr, "vk-mhc: upload refused\n"); return 1;
    }
    ds4vk_tensor_set_op(tfn, "mhc"); ds4vk_tensor_set_op(tsc, "mhc");
    ds4vk_tensor_set_op(tbs, "mhc"); ds4vk_tensor_set_op(tnm, "mhc");
    Dsv4CudaActivation *r = dsv4_cuda_activation_create(0, MH);
    Dsv4CudaActivation *state = dsv4_cuda_activation_create(0, M + M * M + M);
    Dsv4CudaActivation *input = dsv4_cuda_activation_create(0, H);
    Dsv4CudaActivation *x = dsv4_cuda_activation_create(0, H);
    Dsv4CudaActivation *out = dsv4_cuda_activation_create(0, MH);
    if (!r || !state || !input || !x || !out) return 1;
    float *state_host = malloc((size_t)(M + M * M + M) * sizeof(*state_host));
    float *normalized_g = malloc((size_t)H * sizeof(*normalized_g));
    float *out_g = malloc((size_t)MH * sizeof(*out_g));
    if (!state_host || !normalized_g || !out_g) return 1;
    int ok = dsv4_cuda_activation_upload(r, residual, MH) &&
             dsv4_cuda_activation_upload(x, branch, H) &&
             dsv4_cuda_mhc_pre_norm(r, tfn, tsc, tbs, tnm, M, H, rms_eps,
                                    hc_eps, hc_eps, 2.0f, sink_iters,
                                    rms_eps, state, input) &&
             dsv4_cuda_activation_download(state_host, state,
                                           M + M * M + M) &&
             dsv4_cuda_activation_download(normalized_g, input, H) &&
             dsv4_cuda_mhc_post(x, r, state, M, H, out) &&
             dsv4_cuda_activation_download(out_g, out, MH) &&
             dsv4_cuda_activation_sync(out);
    if (!ok) {
        fprintf(stderr, "vk-mhc (M=%d H=%d): GPU chain refused\n", M, H);
        return 1;
    }
    double ma, mr, np_ma, np_mr, o_ma, o_mr;
    cmp_stats(state_host, post_c, M, &ma, &mr);
    int post_ok = mr > 0 && ma / mr < 1e-3;
    cmp_stats(state_host + M, comb_c, M * M, &ma, &mr);
    int comb_ok = mr > 0 && ma / mr < 1e-3;
    cmp_stats(normalized_g, normalized_c, H, &np_ma, &np_mr);
    /* bf16-rounding-point flips: the GPU fn matvec is fp32 tree-order, so a
     * reduced value within ~1e-5 of a bf16 boundary can round to the
     * neighbour ulp — same class as the qkv chain's 1e-2 gate. */
    int norm_ok = np_mr > 0 && np_ma / np_mr < 1e-2;
    cmp_stats(out_g, out_c, MH, &o_ma, &o_mr);
    int out_ok = o_mr > 0 && o_ma / o_mr < 1e-3;
    printf("vk-mhc (M=%d H=%d): post %.3e %s | comb %.3e %s | normalized %.3e %s | out %.3e %s\n",
           M, H,
           post_ok ? 0.0 : ma / mr, post_ok ? "OK" : "FAIL",
           comb_ok ? 0.0 : ma / mr, comb_ok ? "OK" : "FAIL",
           norm_ok ? 0.0 : np_ma / np_mr, norm_ok ? "OK" : "FAIL",
           out_ok ? 0.0 : o_ma / o_mr, out_ok ? "OK" : "FAIL");
    free(out_g); free(normalized_g); free(state_host);
    dsv4_cuda_activation_free(out); dsv4_cuda_activation_free(x);
    dsv4_cuda_activation_free(input); dsv4_cuda_activation_free(state);
    dsv4_cuda_activation_free(r);
    dsv4_cuda_tensor_free(tnm); dsv4_cuda_tensor_free(tbs);
    dsv4_cuda_tensor_free(tsc); dsv4_cuda_tensor_free(tfn);
    free(out_c); free(comb_c); free(post_c); free(normalized_c); free(reduced_c);
    free(branch); free(residual); free(norm); free(base); free(scale); free(fn);
    (void)seed_shift;
    return post_ok && comb_ok && norm_ok && out_ok ? 0 : 1;
}

/* M4-1: the batched prefill mHC — dsv4_cuda_mhc_pre_norm_batch +
 * dsv4_cuda_mhc_post_batch over T tokens vs the engine's per-token
 * normalized_hc_pre / hc_post chain (the prefill block's CPU loop). The
 * backend runs ONE fn matvec over all tokens, then the per-token host-side
 * chain (RMS inverse, sinkhorn, pre*residual, bf16 rounds, branch-norm
 * rmsnorm) with the CUDA batch state layout [post T*M][comb T*M*M]; the
 * oracle concatenates the per-token CPU results in the same layout. The ABI
 * hardcodes the model config (eps 1e-6, iters 20, post_mult 2) — the oracle
 * uses the same values. Returns 0 on pass. */
static int run_mhc_batch(int M, int H, int T, int seed_shift) {
    int N = 2 * M + M * M, MH = M * H;
    float *fn = malloc((size_t)N * MH * sizeof(*fn));
    float *scale = malloc(3 * sizeof(*scale));
    float *base = malloc((size_t)N * sizeof(*base));
    float *norm = malloc((size_t)H * sizeof(*norm));
    float *residual = malloc((size_t)T * MH * sizeof(*residual));
    float *branch = malloc((size_t)T * H * sizeof(*branch));
    if (!fn || !scale || !base || !norm || !residual || !branch) return 1;
    srand(17 + seed_shift);
    for (int i = 0; i < N * MH; i++)
        fn[i] = (float)(rand() % 2001 - 1000) / 100000.0f;
    scale[0] = (float)(rand() % 2000 + 100) / 100.0f;
    scale[1] = (float)(rand() % 2000 + 100) / 100.0f;
    scale[2] = (float)(rand() % 2000 + 100) / 100.0f;
    for (int i = 0; i < N; i++) base[i] = (float)(rand() % 2001 - 1000) / 100.0f;
    for (int i = 0; i < H; i++) norm[i] = (float)(rand() % 2000 + 100) / 1000.0f;
    for (int i = 0; i < T * MH; i++)
        residual[i] = (float)(rand() % 20001 - 10000) / 1000.0f;
    for (int i = 0; i < T * H; i++)
        branch[i] = (float)(rand() % 20001 - 10000) / 1000.0f;

    /* CPU oracle: the per-token chain, concatenated in the batch layout. */
    float *post_c = malloc((size_t)T * M * sizeof(*post_c));
    float *comb_c = malloc((size_t)T * M * M * sizeof(*comb_c));
    float *norm_c = malloc((size_t)T * H * sizeof(*norm_c));
    float *out_c = malloc((size_t)T * MH * sizeof(*out_c));
    float *reduced = malloc((size_t)H * sizeof(*reduced));
    float *p1 = malloc((size_t)M * sizeof(*p1));
    float *c1 = malloc((size_t)M * M * sizeof(*c1));
    float *o1 = malloc((size_t)MH * sizeof(*o1));
    if (!post_c || !comb_c || !norm_c || !out_c || !reduced || !p1 || !c1 || !o1)
        return 1;
    for (int t = 0; t < T; t++) {
        const float *res = residual + (size_t)t * MH;
        if (coli_v4_hc_pre(reduced, p1, c1, res, fn, scale, base, M, H,
                           MHC_BATCH_ITERS, 1e-6f, 1e-6f))
            return 1;
        coli_bf16_round_array(reduced, H);
        if (coli_v4_rmsnorm(norm_c + (size_t)t * H, reduced, norm, H, 1e-6f))
            return 1;
        coli_bf16_round_array(norm_c + (size_t)t * H, H);
        memcpy(post_c + (size_t)t * M, p1, (size_t)M * sizeof(*p1));
        memcpy(comb_c + (size_t)t * M * M, c1, (size_t)M * M * sizeof(*c1));
        if (coli_v4_hc_post(o1, branch + (size_t)t * H, res, p1, c1, M, H))
            return 1;
        coli_bf16_round_array(o1, MH);
        memcpy(out_c + (size_t)t * MH, o1, (size_t)MH * sizeof(*o1));
    }

    Dsv4CudaTensor *tfn = NULL, *tsc = NULL, *tbs = NULL, *tnm = NULL;
    if (!dsv4_cuda_upload_f32(&tfn, fn, N, MH, 0) ||
        !dsv4_cuda_upload_f32(&tsc, scale, 3, 1, 0) ||
        !dsv4_cuda_upload_f32(&tbs, base, N, 1, 0) ||
        !dsv4_cuda_upload_f32(&tnm, norm, H, 1, 0)) {
        fprintf(stderr, "vk-mhc-batch: upload refused\n"); return 1;
    }
    ds4vk_tensor_set_op(tfn, "mhc"); ds4vk_tensor_set_op(tsc, "mhc");
    ds4vk_tensor_set_op(tbs, "mhc"); ds4vk_tensor_set_op(tnm, "mhc");
    Dsv4CudaActivation *r = dsv4_cuda_activation_create(0, T * MH);
    Dsv4CudaActivation *state = dsv4_cuda_activation_create(0, T * (M + M * M));
    Dsv4CudaActivation *input = dsv4_cuda_activation_create(0, T * H);
    Dsv4CudaActivation *x = dsv4_cuda_activation_create(0, T * H);
    Dsv4CudaActivation *out = dsv4_cuda_activation_create(0, T * MH);
    if (!r || !state || !input || !x || !out) return 1;
    float *state_host = malloc((size_t)T * (M + M * M) * sizeof(*state_host));
    float *norm_g = malloc((size_t)T * H * sizeof(*norm_g));
    float *out_g = malloc((size_t)T * MH * sizeof(*out_g));
    if (!state_host || !norm_g || !out_g) return 1;
    int ok = dsv4_cuda_activation_upload(r, residual, T * MH) &&
             dsv4_cuda_activation_upload(x, branch, T * H) &&
             dsv4_cuda_mhc_pre_norm_batch(r, tfn, tsc, tbs, tnm, T, H, state,
                                          input) &&
             dsv4_cuda_activation_download(state_host, state, T * (M + M * M)) &&
             dsv4_cuda_activation_download(norm_g, input, T * H) &&
             dsv4_cuda_mhc_post_batch(x, r, state, T, H, out) &&
             dsv4_cuda_activation_download(out_g, out, T * MH) &&
             dsv4_cuda_activation_sync(out);
    if (!ok) {
        fprintf(stderr, "vk-mhc-batch (M=%d H=%d T=%d): GPU chain refused\n",
                M, H, T);
        return 1;
    }
    double ma, mr;
    cmp_stats(state_host, post_c, T * M, &ma, &mr);
    int post_ok = mr > 0 && ma / mr < 1e-3;
    cmp_stats(state_host + T * M, comb_c, T * M * M, &ma, &mr);
    int comb_ok = mr > 0 && ma / mr < 1e-3;
    cmp_stats(norm_g, norm_c, T * H, &ma, &mr);
    int norm_ok = mr > 0 && ma / mr < 1e-2;
    cmp_stats(out_g, out_c, T * MH, &ma, &mr);
    /* the fn matvec is fp32-tree vs the CPU's sequential fp32; the sigmoid/
     * softmax chain amplifies ~1e-7 mix diffs to ~1e-5, and a value within a
     * bf16 ulp of a rounding boundary flips the rounded output — the same
     * bf16-ulp-flip class the M3c single-token case documents (threshold
     * 1e-2, the qkv chain's gate). */
    int out_ok = mr > 0 && ma / mr < 1e-2;
    printf("vk-mhc-batch (M=%d H=%d T=%d): post %.3e %s | comb %.3e %s | "
           "normalized %.3e %s | out %.3e %s\n", M, H, T,
           post_ok ? 0.0 : ma / mr, post_ok ? "OK" : "FAIL",
           comb_ok ? 0.0 : ma / mr, comb_ok ? "OK" : "FAIL",
           norm_ok ? 0.0 : ma / mr, norm_ok ? "OK" : "FAIL",
           out_ok ? 0.0 : ma / mr, out_ok ? "OK" : "FAIL");
    free(out_g); free(norm_g); free(state_host);
    dsv4_cuda_activation_free(out); dsv4_cuda_activation_free(x);
    dsv4_cuda_activation_free(input); dsv4_cuda_activation_free(state);
    dsv4_cuda_activation_free(r);
    dsv4_cuda_tensor_free(tnm); dsv4_cuda_tensor_free(tbs);
    dsv4_cuda_tensor_free(tsc); dsv4_cuda_tensor_free(tfn);
    free(o1); free(c1); free(p1); free(reduced); free(out_c); free(norm_c);
    free(comb_c); free(post_c);
    free(branch); free(residual); free(norm); free(base); free(scale); free(fn);
    return post_ok && comb_ok && norm_ok && out_ok ? 0 : 1;
}

/* M4-1: the plain batched pre (dsv4_cuda_mhc_pre_batch, no norm rmsnorm) and
 * the fused post -> next pre_norm (dsv4_cuda_mhc_post_pre_norm_batch). Tiny
 * shape only (the engine uses the _norm pre on the real model; these entries
 * are ABI completeness). Returns 0 on pass. */
static int run_mhc_batch_plain_fused(int M, int H, int T, int seed_shift) {
    int N = 2 * M + M * M, MH = M * H;
    float *fn = malloc((size_t)N * MH * sizeof(*fn));
    float *scale = malloc(3 * sizeof(*scale));
    float *base = malloc((size_t)N * sizeof(*base));
    float *norm = malloc((size_t)H * sizeof(*norm));
    float *residual = malloc((size_t)T * MH * sizeof(*residual));
    float *branch = malloc((size_t)T * H * sizeof(*branch));
    if (!fn || !scale || !base || !norm || !residual || !branch) return 1;
    srand(23 + seed_shift);
    for (int i = 0; i < N * MH; i++)
        fn[i] = (float)(rand() % 2001 - 1000) / 100000.0f;
    for (int i = 0; i < 3; i++) scale[i] = (float)(rand() % 2000 + 100) / 100.0f;
    for (int i = 0; i < N; i++) base[i] = (float)(rand() % 2001 - 1000) / 100.0f;
    for (int i = 0; i < H; i++) norm[i] = (float)(rand() % 2000 + 100) / 1000.0f;
    for (int i = 0; i < T * MH; i++)
        residual[i] = (float)(rand() % 20001 - 10000) / 1000.0f;
    for (int i = 0; i < T * H; i++)
        branch[i] = (float)(rand() % 20001 - 10000) / 1000.0f;

    /* CPU: hc_pre (no rmsnorm) -> bf16; then hc_post -> bf16; then the next
     * normalized_hc_pre over the post output (the fused contract). The
     * fused GPU call OVERWRITES state with the SECOND pre's post/comb, so
     * the oracle keeps both generations. */
    float *reduced_c = malloc((size_t)T * H * sizeof(*reduced_c));
    float *post_c = malloc((size_t)T * M * sizeof(*post_c));
    float *comb_c = malloc((size_t)T * M * M * sizeof(*comb_c));
    float *post2_c = malloc((size_t)T * M * sizeof(*post2_c));
    float *comb2_c = malloc((size_t)T * M * M * sizeof(*comb2_c));
    float *out_c = malloc((size_t)T * MH * sizeof(*out_c));
    float *norm_c = malloc((size_t)T * H * sizeof(*norm_c));
    float *p1 = malloc((size_t)M * sizeof(*p1));
    float *c1 = malloc((size_t)M * M * sizeof(*c1));
    float *r1 = malloc((size_t)H * sizeof(*r1));
    float *o1 = malloc((size_t)MH * sizeof(*o1));
    if (!reduced_c || !post_c || !comb_c || !post2_c || !comb2_c || !out_c ||
        !norm_c || !p1 || !c1 || !r1 || !o1) return 1;
    for (int t = 0; t < T; t++) {
        const float *res = residual + (size_t)t * MH;
        if (coli_v4_hc_pre(r1, p1, c1, res, fn, scale, base, M, H,
                           MHC_BATCH_ITERS, 1e-6f, 1e-6f))
            return 1;
        coli_bf16_round_array(r1, H);
        memcpy(reduced_c + (size_t)t * H, r1, (size_t)H * sizeof(*r1));
        memcpy(post_c + (size_t)t * M, p1, (size_t)M * sizeof(*p1));
        memcpy(comb_c + (size_t)t * M * M, c1, (size_t)M * M * sizeof(*c1));
        if (coli_v4_hc_post(o1, branch + (size_t)t * H, res, p1, c1, M, H))
            return 1;
        coli_bf16_round_array(o1, MH);
        memcpy(out_c + (size_t)t * MH, o1, (size_t)MH * sizeof(*o1));
        /* fused: the next pre_norm runs over the post output. */
        if (coli_v4_hc_pre(r1, p1, c1, o1, fn, scale, base, M, H,
                           MHC_BATCH_ITERS, 1e-6f, 1e-6f))
            return 1;
        coli_bf16_round_array(r1, H);
        if (coli_v4_rmsnorm(norm_c + (size_t)t * H, r1, norm, H, 1e-6f))
            return 1;
        coli_bf16_round_array(norm_c + (size_t)t * H, H);
        memcpy(post2_c + (size_t)t * M, p1, (size_t)M * sizeof(*p1));
        memcpy(comb2_c + (size_t)t * M * M, c1, (size_t)M * M * sizeof(*c1));
    }

    Dsv4CudaTensor *tfn = NULL, *tsc = NULL, *tbs = NULL, *tnm = NULL;
    if (!dsv4_cuda_upload_f32(&tfn, fn, N, MH, 0) ||
        !dsv4_cuda_upload_f32(&tsc, scale, 3, 1, 0) ||
        !dsv4_cuda_upload_f32(&tbs, base, N, 1, 0) ||
        !dsv4_cuda_upload_f32(&tnm, norm, H, 1, 0)) return 1;
    ds4vk_tensor_set_op(tfn, "mhc"); ds4vk_tensor_set_op(tsc, "mhc");
    ds4vk_tensor_set_op(tbs, "mhc"); ds4vk_tensor_set_op(tnm, "mhc");
    Dsv4CudaActivation *r = dsv4_cuda_activation_create(0, T * MH);
    Dsv4CudaActivation *state = dsv4_cuda_activation_create(0, T * (M + M * M));
    Dsv4CudaActivation *input = dsv4_cuda_activation_create(0, T * H);
    Dsv4CudaActivation *x = dsv4_cuda_activation_create(0, T * H);
    Dsv4CudaActivation *out = dsv4_cuda_activation_create(0, T * MH);
    if (!r || !state || !input || !x || !out) return 1;
    float *state_host = malloc((size_t)T * (M + M * M) * sizeof(*state_host));
    float *red_g = malloc((size_t)T * H * sizeof(*red_g));
    float *norm_g = malloc((size_t)T * H * sizeof(*norm_g));
    float *out_g = malloc((size_t)T * MH * sizeof(*out_g));
    if (!state_host || !red_g || !norm_g || !out_g) return 1;
    int ok = dsv4_cuda_activation_upload(r, residual, T * MH) &&
             dsv4_cuda_activation_upload(x, branch, T * H) &&
             dsv4_cuda_mhc_pre_batch(r, tfn, tsc, tbs, T, H, state, input) &&
             dsv4_cuda_activation_download(red_g, input, T * H) &&
             dsv4_cuda_mhc_post_pre_norm_batch(x, r, state, T, H, out, tfn,
                                               tsc, tbs, tnm, input) &&
             dsv4_cuda_activation_download(state_host, state, T * (M + M * M)) &&
             dsv4_cuda_activation_download(norm_g, input, T * H) &&
             dsv4_cuda_activation_download(out_g, out, T * MH) &&
             dsv4_cuda_activation_sync(out);
    if (!ok) {
        fprintf(stderr, "vk-mhc-batch-plain (M=%d H=%d T=%d): refused\n",
                M, H, T);
        return 1;
    }
    double ma, mr;
    cmp_stats(red_g, reduced_c, T * H, &ma, &mr);
    int red_ok = mr > 0 && ma / mr < 1e-3;
    cmp_stats(out_g, out_c, T * MH, &ma, &mr);
    int out_ok = mr > 0 && ma / mr < 1e-3;
    cmp_stats(norm_g, norm_c, T * H, &ma, &mr);
    int norm_ok = mr > 0 && ma / mr < 1e-2;
    cmp_stats(state_host, post2_c, T * M, &ma, &mr);
    int post_ok = mr > 0 && ma / mr < 1e-3;
    cmp_stats(state_host + T * M, comb2_c, T * M * M, &ma, &mr);
    int comb_ok = mr > 0 && ma / mr < 1e-3;
    printf("vk-mhc-batch-plain/fused (M=%d H=%d T=%d): reduced %.3e %s | "
           "out %.3e %s | normalized %.3e %s | post %.3e %s | comb %.3e %s\n",
           M, H, T,
           red_ok ? 0.0 : ma / mr, red_ok ? "OK" : "FAIL",
           out_ok ? 0.0 : ma / mr, out_ok ? "OK" : "FAIL",
           norm_ok ? 0.0 : ma / mr, norm_ok ? "OK" : "FAIL",
           post_ok ? 0.0 : ma / mr, post_ok ? "OK" : "FAIL",
           comb_ok ? 0.0 : ma / mr, comb_ok ? "OK" : "FAIL");
    free(out_g); free(norm_g); free(red_g); free(state_host);
    dsv4_cuda_activation_free(out); dsv4_cuda_activation_free(x);
    dsv4_cuda_activation_free(input); dsv4_cuda_activation_free(state);
    dsv4_cuda_activation_free(r);
    dsv4_cuda_tensor_free(tnm); dsv4_cuda_tensor_free(tbs);
    dsv4_cuda_tensor_free(tsc); dsv4_cuda_tensor_free(tfn);
    free(o1); free(r1); free(c1); free(p1); free(norm_c); free(out_c);
    free(comb2_c); free(post2_c); free(comb_c); free(post_c); free(reduced_c);
    free(branch); free(residual); free(norm); free(base); free(scale); free(fn);
    return red_ok && out_ok && norm_ok && post_ok && comb_ok ? 0 : 1;
}

/* M4-1: the compressor projection matmul (dsv4_cuda_matmul_bf16_batch) —
 * bf16 weights, fp32 activations, S tokens. Oracle = the scalar formula the
 * engine's compressor_step computes inline: decode bf16 to f32, fp32 dot in
 * the CPU's sequential order (the same reference basis the head matvec's L2
 * case documents). Thresholded (fp32 tree vs sequential). Returns 0 on pass. */
static int run_comp_batch(int rows, int hidden, int T, int seed_shift) {
    uint16_t *w = malloc((size_t)rows * hidden * sizeof(*w));
    float *x = malloc((size_t)T * hidden * sizeof(*x));
    float *y_g = malloc((size_t)T * rows * sizeof(*y_g));
    float *y_c = malloc((size_t)T * rows * sizeof(*y_c));
    if (!w || !x || !y_g || !y_c) return 1;
    srand(29 + seed_shift);
    for (int i = 0; i < rows * hidden; i++)
        w[i] = f32_to_bf16((float)(rand() % 2001 - 1000) / 100.0f);
    for (int i = 0; i < T * hidden; i++)
        x[i] = (float)(rand() % 20001 - 10000) / 1000.0f;
    for (int t = 0; t < T; t++)
        for (int r = 0; r < rows; r++) {
            float v = 0.0f;
            for (int h = 0; h < hidden; h++) {
                uint32_t bits = (uint32_t)w[(size_t)r * hidden + h] << 16;
                float wf;
                memcpy(&wf, &bits, sizeof(bits));
                v += wf * x[(size_t)t * hidden + h];
            }
            y_c[(size_t)t * rows + r] = v;
        }
    Dsv4CudaTensor *tw = NULL;
    if (!dsv4_cuda_upload_bf16(&tw, w, rows, hidden, 0)) {
        fprintf(stderr, "vk-comp: bf16 upload refused\n");
        return 1;
    }
    ds4vk_tensor_set_op(tw, "comp");
    int ok = dsv4_cuda_matmul_bf16_batch(tw, x, T, y_g);
    if (!ok) {
        fprintf(stderr, "vk-comp: matmul_bf16_batch refused\n");
        return 1;
    }
    double ma, mr;
    cmp_stats(y_g, y_c, T * rows, &ma, &mr);
    int y_ok = mr > 0 && ma / mr < 1e-3;
    printf("vk-comp bf16 batch (rows=%d hidden=%d T=%d): maxabs/maxref "
           "%.3e/%.3e %s\n", rows, hidden, T, ma, mr, y_ok ? "OK" : "FAIL");
    dsv4_cuda_tensor_free(tw);
    free(y_c); free(y_g); free(x); free(w);
    return y_ok ? 0 : 1;
}

int main(void) {
    /* op mask: mhc + comp groups — proves both M4-1 batch op groups are
     * honoured by the backend gate and that a different group (route) is
     * refused under the same env. */
    setenv("COLI_DSV4_VK_OPS", "mhc,comp", 1);
    if (!coli_vk_init("shaders/qmatmul.spv") || !coli_vk_available()) {
        fprintf(stderr, "vk-mhc: no Vulkan device — skipped\n");
        return 0;
    }
    srand(17);
    int bad = 0;
    /* real DeepSeek-V4-Flash-0731 shape: hc_mult=4, hidden 4096 */
    bad |= run_mhc(4, 4096, 0);
    /* tiny sanity: hc_mult=2 config shape */
    bad |= run_mhc(2, 128, 1);
    /* M4-1: prefill batch mHC — real shape, T=3 chunk */
    bad |= run_mhc_batch(4, 4096, 3, 0);
    /* M4-1: plain batched pre + fused post->pre_norm (tiny shape) */
    bad |= run_mhc_batch_plain_fused(2, 128, 2, 1);
    /* M4-1: compressor projection bf16 batch matmul — real compressor rows
     * (ratio==4 -> 2*head_dim=1024) and the indexer-compressor rows
     * (2*index_head_dim=256). */
    bad |= run_comp_batch(1024, 4096, 3, 0);
    bad |= run_comp_batch(256, 4096, 3, 1);
    /* per-op gating refusal: a route-tagged f32 matvec must NOT run while
     * only mhc/comp are enabled (COLI_DSV4_VK_OPS=mhc,comp). */
    {
        float w[8 * 16], x[16];
        for (int i = 0; i < 8 * 16; i++) w[i] = (float)(rand() % 2001 - 1000) / 100.0f;
        for (int i = 0; i < 16; i++) x[i] = (float)(rand() % 2001 - 1000) / 100.0f;
        Dsv4CudaTensor *t = NULL;
        int up = dsv4_cuda_upload_f32(&t, w, 8, 16, 0);
        if (up) ds4vk_tensor_set_op(t, "route");
        float y[8];
        int ran = up && dsv4_cuda_matvec(t, y, x);
        if (ran) {
            fprintf(stderr, "vk-mhc: route op ran with OPS=mhc — gating broken\n");
            bad = 1;
        } else {
            printf("vk-mhc: route refused under OPS=mhc (gating OK)\n");
        }
        if (t) dsv4_cuda_tensor_free(t);
    }
    if (bad) { fprintf(stderr, "vk-mhc: FAIL\n"); coli_vk_shutdown(); return 1; }
    printf("vk-mhc: OK\n");
    coli_vk_shutdown();
    return 0;
}
