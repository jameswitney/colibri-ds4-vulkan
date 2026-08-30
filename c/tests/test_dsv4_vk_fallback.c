/* M3d-1 L4 fault injection for the dsv4 Vulkan tier (TEST.md §4, M3d gate).
 * Template: tests/test_dsv4_vk_ops.c.
 *
 * The D6 contract: "a wedged GPU can slow a run, never corrupt it." Every
 * dsv4 op returns success/failure and the engine's per-op dispatch falls back
 * to the CPU path on any refusal. The fault-injection hook
 * (COLI_DSV4_VK_FAIL=<group>,... in backend_vulkan_dsv4.c) forces backend
 * entries to fail so the fallback path is exercised deterministically:
 *   - case 2 (per-op fallback): COLI_DSV4_VK_FAIL=wo forces the wo-tagged
 *     matvec/grouped entries to fail while the qkv group keeps working — the
 *     engine would fall back per-op (D6), output unchanged. The hook must be
 *     per-group: a failed group never takes the others down.
 *   - case 4 (graceful degrade): COLI_DSV4_VK_FAIL=upload forces the upload
 *     entries to fail cleanly (*t = NULL, return 0) — no boot hang, the
 *     engine's "layer-N-upload-failed; continuing-CPU" path (L0 semantics).
 *   - parse robustness: an unknown group name (COLI_DSV4_VK_FAIL=bogus) must
 *     parse to an empty mask — no faults, everything works.
 *
 * The post-D7 reload + permanent-CPU mode lives in the ENGINE GPU unit
 * (coli_v4_gpu_layer_post_upload / v4_vk_on_op_failure, deepseek_v4.c), not
 * in the backend — it is exercised end-to-end on the real checkpoint
 * (L4 case 3, see PROGRESS M3d-1): COLI_DSV4_VK_FAIL on a run with the D7
 * drop active must show the one-time reload + "tier=permanent-CPU" warning
 * and byte-identical output.
 *
 * The fault mask is parsed once per process on first use (static cache), so
 * each scenario runs in a forked child with its own env — and each child
 * owns its own Vulkan backend lifecycle: the weight arena allocates a 256 MB
 * BAR-window block on first use, and a block allocated through the shared
 * device fd is NOT released when a child exits while the parent still holds
 * the fd, so a child inheriting the parent's initialized backend exhausts the
 * BAR window for every later child (measured). Child-owned init/shutdown
 * frees each block. Skips (exit 0) without a Vulkan ICD, like the other vk
 * tests.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/wait.h>
#include <unistd.h>
#include "../quant.h"
#include "../tensor.h"
#include "../backend_vulkan.h"
#include "../backend_vulkan_dsv4.h"

/* NaN-free random e4m3 weight byte (M2-4 discipline: 0x7f/0xff are NaN codes
 * — real checkpoints never carry them, and NaN bytes poison every dot
 * product on both sides). */
static uint8_t nn_byte(void) {
    uint8_t c = (uint8_t)rand();
    return (c & 0x7f) == 0x7f ? (uint8_t)(c & 0x7e) : c;
}

/* f32 -> bf16 (round-to-nearest-even), same as the ops test's run_head — raw
 * random 16-bit patterns include bf16 NaN codes, which make every score NaN. */
static uint16_t f32_to_bf16(float value) {
    uint32_t bits; memcpy(&bits, &value, 4);
    uint32_t rounded = (bits + 0x7fffu + ((bits >> 16) & 1)) >> 16;
    return (uint16_t)rounded;
}

static Dsv4CudaTensor *mk_fp8(int O, int I, const char *op) {
    size_t wbytes = (size_t)O * (size_t)I;
    size_t sbytes = (size_t)((O + 127) / 128) * (size_t)((I + 127) / 128);
    uint8_t *w = malloc(wbytes);
    uint8_t *sc = malloc(sbytes);
    if (!w || !sc) { free(w); free(sc); return NULL; }
    for (size_t i = 0; i < wbytes; i++) w[i] = nn_byte();
    for (size_t i = 0; i < sbytes; i++) sc[i] = (uint8_t)(rand() & 0xff);
    Dsv4CudaTensor *t = NULL;
    int ok = dsv4_cuda_upload_fp8(&t, w, sc, O, I, 0);
    free(w); free(sc);
    if (!ok) return NULL;
    if (op && !ds4vk_tensor_set_op(t, op)) { dsv4_cuda_tensor_free(t); return NULL; }
    return t;
}

static float *mk_act(int n) {
    float *x = malloc((size_t)n * sizeof(float));
    for (int i = 0; i < n; i++) x[i] = (float)(rand() % 200001 - 100000) / 100.0f;
    return x;
}

/* ---- scenario: per-op fault isolation (L4 case 2) ---- */
static int scn_op_fault(void) {
    setenv("COLI_DSV4_VK_FAIL", "wo", 1);
    srand(7);
    const int O = 512, I = 512;
    Dsv4CudaTensor *t_wo = mk_fp8(O, I, "wo");
    Dsv4CudaTensor *t_qkv = mk_fp8(O, I, "qkv");
    float *x = mk_act(I);
    float *y = malloc((size_t)O * sizeof(float));
    int bad = 0;
    if (!t_wo || !t_qkv || !x || !y) {
        fprintf(stderr, "  upload refused (unexpected under fail=wo)\n");
        bad = 1;
    } else {
        /* the failed group must refuse... */
        if (dsv4_cuda_matvec(t_wo, y, x)) {
            fprintf(stderr, "  FAIL: wo matvec succeeded under COLI_DSV4_VK_FAIL=wo\n");
            bad = 1;
        }
        if (dsv4_cuda_matvec_grouped(t_wo, y, x, 2)) {
            fprintf(stderr, "  FAIL: wo grouped matvec succeeded under fail=wo\n");
            bad = 1;
        }
        /* ...and only that group: qkv must keep working (D6 per-op fallback
         * never takes the other groups down). */
        if (!dsv4_cuda_matvec(t_qkv, y, x)) {
            fprintf(stderr, "  FAIL: qkv matvec refused while wo failed (isolation broken)\n");
            bad = 1;
        }
        /* the M3d batch ops carry the same per-group hook: the batch matmul
         * on a wo-tagged tensor must fail, on a qkv-tagged one must work */
        Dsv4CudaActivation *bin = dsv4_cuda_activation_create(0, 2 * (long long)I);
        Dsv4CudaActivation *bout = dsv4_cuda_activation_create(0, 2 * (long long)O);
        if (!bin || !bout || !dsv4_cuda_activation_upload(bin, x, 2 * (long long)I)) {
            fprintf(stderr, "  batch activation setup refused (unexpected)\n");
            bad = 1;
        } else {
            if (dsv4_cuda_matmul_batch(t_wo, bin, 2, bout)) {
                fprintf(stderr, "  FAIL: wo batch matmul succeeded under fail=wo\n");
                bad = 1;
            }
            if (!dsv4_cuda_matmul_batch(t_qkv, bin, 2, bout)) {
                fprintf(stderr, "  FAIL: qkv batch matmul refused while wo failed\n");
                bad = 1;
            }
        }
        if (bin) dsv4_cuda_activation_free(bin);
        if (bout) dsv4_cuda_activation_free(bout);
    }
    if (t_wo) dsv4_cuda_tensor_free(t_wo);
    if (t_qkv) dsv4_cuda_tensor_free(t_qkv);
    free(x); free(y);
    printf("  fail=wo: wo refused, qkv ok: %s\n", bad ? "FAIL" : "PASS");
    return bad;
}

/* ---- scenario: graceful upload degrade (L4 case 4) ---- */
static int scn_upload_fault(void) {
    setenv("COLI_DSV4_VK_FAIL", "upload", 1);
    srand(9);
    const int O = 512, I = 512;
    int bad = 0;
    uint8_t *w = malloc((size_t)O * I);
    uint8_t *sc = malloc((size_t)((O + 127) / 128) * (size_t)((I + 127) / 128));
    for (size_t i = 0; i < (size_t)O * I; i++) w[i] = nn_byte();
    for (size_t i = 0; i < (size_t)((O + 127) / 128) * (size_t)((I + 127) / 128); i++)
        sc[i] = (uint8_t)(rand() & 0xff);
    Dsv4CudaTensor *t = (Dsv4CudaTensor *)1;   /* must be reset to NULL */
    if (dsv4_cuda_upload_fp8(&t, w, sc, O, I, 0) || t != NULL) {
        fprintf(stderr, "  FAIL: fp8 upload succeeded under fail=upload\n");
        bad = 1;
    }
    t = (Dsv4CudaTensor *)1;
    if (dsv4_cuda_upload_bf16(&t, (const uint16_t *)w, O, I, 0) || t != NULL) {
        fprintf(stderr, "  FAIL: bf16 upload succeeded under fail=upload\n");
        bad = 1;
    }
    t = (Dsv4CudaTensor *)1;
    if (dsv4_cuda_upload_f32(&t, (const float *)w, O, I, 0) || t != NULL) {
        fprintf(stderr, "  FAIL: f32 upload succeeded under fail=upload\n");
        bad = 1;
    }
    free(w); free(sc);
    printf("  fail=upload: all uploads refused cleanly (*t=NULL): %s\n",
           bad ? "FAIL" : "PASS");
    return bad;
}

/* ---- scenario: unknown group parses to an empty mask ---- */
static int scn_bogus_group(void) {
    setenv("COLI_DSV4_VK_FAIL", "bogus", 1);
    srand(13);
    const int O = 512, I = 512;
    Dsv4CudaTensor *t_wo = mk_fp8(O, I, "wo");
    float *x = mk_act(I);
    float *y = malloc((size_t)O * sizeof(float));
    int bad = 0;
    if (!t_wo || !x || !y) {
        fprintf(stderr, "  upload refused (unexpected)\n");
        bad = 1;
    } else if (!dsv4_cuda_matvec(t_wo, y, x)) {
        fprintf(stderr, "  FAIL: wo matvec refused under COLI_DSV4_VK_FAIL=bogus "
                        "(unknown group must parse to no faults)\n");
        bad = 1;
    }
    if (t_wo) dsv4_cuda_tensor_free(t_wo);
    free(x); free(y);
    printf("  fail=bogus: no faults, wo ok: %s\n", bad ? "FAIL" : "PASS");
    return bad;
}

/* ---- scenario: multi-group + route/head entries ---- */
static int scn_multi_group(void) {
    setenv("COLI_DSV4_VK_FAIL", "wo,route", 1);
    srand(17);
    const int O = 256, I = 512, H = 512;
    int bad = 0;
    Dsv4CudaTensor *t_wo = mk_fp8(O, I, "wo");
    Dsv4CudaTensor *t_head = NULL;
    uint16_t *head = malloc((size_t)O * (size_t)I * 2);
    /* bf16 rows via f32_to_bf16 — raw random 16-bit patterns include NaN
     * codes (exponent all-ones), which make every score NaN and the argmax
     * legitimately fail (the same discipline as the ops test's run_head). */
    for (size_t i = 0; i < (size_t)O * (size_t)I; i++)
        head[i] = f32_to_bf16((float)(rand() % 200001 - 100000) / 1000.0f);
    if (!dsv4_cuda_upload_bf16(&t_head, head, O, I, 0)) {
        fprintf(stderr, "  head bf16 upload refused (unexpected)\n");
        bad = 1;
    }
    Dsv4CudaTensor *t_gate = NULL;
    float *gate = mk_act(O * I);
    if (!dsv4_cuda_upload_f32(&t_gate, gate, O, I, 0) ||
        !ds4vk_tensor_set_op(t_gate, "route")) {
        fprintf(stderr, "  f32 gate upload refused (unexpected)\n");
        bad = 1;
    }
    float *x = mk_act(I);
    float *y = malloc((size_t)O * sizeof(float));
    Dsv4CudaActivation *act = dsv4_cuda_activation_create(0, I);
    int ids[6]; float wts[6];
    if (!t_wo || !t_head || !t_gate || !x || !y || !act) {
        fprintf(stderr, "  setup refused (unexpected)\n");
        bad = 1;
    } else {
        if (dsv4_cuda_matvec(t_wo, y, x)) {
            fprintf(stderr, "  FAIL: wo matvec succeeded under fail=wo,route\n");
            bad = 1;
        }
        if (dsv4_cuda_activation_upload(act, x, I) &&
            dsv4_cuda_route(act, t_gate, NULL, NULL, 1.25f, ids, wts)) {
            fprintf(stderr, "  FAIL: route succeeded under fail=wo,route\n");
            bad = 1;
        }
        /* head is not in the fail list — must keep working */
        int id = -1; float val = 0.0f;
        if (!dsv4_cuda_head_argmax(t_head, x, &id, &val)) {
            fprintf(stderr, "  FAIL: head refused while wo,route failed (isolation broken)\n");
            bad = 1;
        }
    }
    if (act) dsv4_cuda_activation_free(act);
    if (t_wo) dsv4_cuda_tensor_free(t_wo);
    if (t_head) dsv4_cuda_tensor_free(t_head);
    if (t_gate) dsv4_cuda_tensor_free(t_gate);
    free(head); free(gate); free(x); free(y);
    printf("  fail=wo,route: wo+route refused, head ok: %s\n", bad ? "FAIL" : "PASS");
    return bad;
}

/* ---- scenario: no fault env — everything works (sanity baseline) ---- */
static int scn_clean(void) {
    unsetenv("COLI_DSV4_VK_FAIL");
    srand(23);
    const int O = 512, I = 512;
    Dsv4CudaTensor *t_wo = mk_fp8(O, I, "wo");
    float *x = mk_act(I);
    float *y = malloc((size_t)O * sizeof(float));
    int bad = 0;
    if (!t_wo || !x || !y) {
        fprintf(stderr, "  upload refused (unexpected)\n");
        bad = 1;
    } else if (!dsv4_cuda_matvec(t_wo, y, x)) {
        fprintf(stderr, "  FAIL: wo matvec refused with no fault env\n");
        bad = 1;
    }
    if (t_wo) dsv4_cuda_tensor_free(t_wo);
    free(x); free(y);
    printf("  no env: wo ok: %s\n", bad ? "FAIL" : "PASS");
    return bad;
}

int main(void) {
    /* Parent probe: one clean init/shutdown just to test ICD availability
     * (skips exit 0 without a Vulkan device, like the other vk tests).
     * Each scenario then runs in a forked child that inits the backend
     * ITSELF: the fault mask is a process-static cache parsed on first use
     * (so each child needs its own env), and the weight arena allocates a
     * 256 MB BAR-window block on first use that is NOT released on child
     * exit while the parent still holds the shared device fd — a child that
     * inherited the parent's initialized backend would exhaust the BAR
     * window for every later child (measured). A child-owned backend frees
     * its blocks at coli_vk_shutdown. */
    if (!coli_vk_init("shaders/qmatmul.spv") || !coli_vk_available()) {
        fprintf(stderr, "vk-fallback: no Vulkan device — skipped\n");
        return 0;
    }
    coli_vk_shutdown();
    typedef int (*scn_fn)(void);
    static const struct { const char *name; scn_fn fn; } scns[] = {
        { "per-op isolation", scn_op_fault },      /* COLI_DSV4_VK_FAIL=wo */
        { "upload degrade",   scn_upload_fault },  /* COLI_DSV4_VK_FAIL=upload */
        { "bogus group",      scn_bogus_group },   /* COLI_DSV4_VK_FAIL=bogus */
        { "multi-group",      scn_multi_group },   /* COLI_DSV4_VK_FAIL=wo,route */
        { "clean baseline",   scn_clean },         /* unset */
    };
    int bad = 0;
    for (size_t i = 0; i < sizeof(scns) / sizeof(scns[0]); i++) {
        pid_t pid = fork();
        if (pid < 0) { perror("fork"); bad = 1; break; }
        if (pid == 0) {
            int r = 1;
            if (coli_vk_init("shaders/qmatmul.spv") && coli_vk_available()) {
                r = scns[i].fn();
                coli_vk_shutdown();
            }
            fflush(stdout); fflush(stderr);
            _exit(r ? 1 : 0);
        }
        int status = 0;
        if (waitpid(pid, &status, 0) < 0) { perror("waitpid"); bad = 1; continue; }
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
            fprintf(stderr, "vk-fallback: scenario '%s' FAILED (status %d)\n",
                    scns[i].name, status);
            bad = 1;
        }
    }
    if (bad) { fprintf(stderr, "vk-fallback: FAIL\n"); return 1; }
    printf("vk-fallback: OK\n");
    return 0;
}
