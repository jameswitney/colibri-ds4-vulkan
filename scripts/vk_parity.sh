#!/usr/bin/env bash
# vk_parity.sh — L3/L5 parity orchestrator for the dsv4 Vulkan dense tier.
#
# TEST.md §4 promised this script ("build → tiny oracle → real oracle → per-op
# A/B → benches"); this is the L3 part (M3p-1, 2026-08-28). It turns the
# per-slice parity gates (M3a–M3e) into one reproducible command:
#
#   1. build the VK=1 binary (or reuse the one already present)
#   2. run the L1/L2/L4 suites (test-vk-dsv4, -ops, -attn, -mhc, -fallback)
#   3. L3 tiny oracle (test_deepseek_v4_tiny.py + test_deepseek_v4_prefix.py)
#      with COLI_DSV4_VK_DENSE=1 — must be 12/12 + prefix 3/3
#   4. L3 real-checkpoint A/B (tier-off vs tier-on) — stdout byte-identical
#      (the TUNE decode wall line is the only permitted diff)
#   5. per-op A/B on the tiny oracle (--per-op-ab): every COLI_DSV4_VK_OPS
#      combination must stay token-exact vs the all-CPU run
#
# Usage:
#   ./scripts/vk_parity.sh [--rebuild] [--per-op-ab] [--model /path/to/checkpoint]
#
# Env:
#   MODEL              checkpoint path (real-checkpoint A/B; the tiny oracle
#                      needs only the generated fixture at c/deepseek_v4_tiny)
#   VK_ARCH            -march for the build (default: native)
#   VK_PROMPT          validation prompt for the real A/B (default: "The
#                      capital of France is")
#   VK_MAX_TOKENS      greedy tokens for the real A/B (default: 8)
#   VK_MEMORY_GB       --memory-gb for the real A/B (default: 12)
#
# Exit: 0 = all gates green; non-zero = first failing gate (set -e).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
C="$ROOT/c"
MODEL="${MODEL:-${1:-}}"
VK_ARCH="${VK_ARCH:-native}"
VK_PROMPT="${VK_PROMPT:-The capital of France is}"
VK_MAX_TOKENS="${VK_MAX_TOKENS:-8}"
VK_MEMORY_GB="${VK_MEMORY_GB:-12}"
REBUILD=0
PER_OP_AB=0

for arg in "$@"; do
    case "$arg" in
        --rebuild) REBUILD=1 ;;
        --per-op-ab) PER_OP_AB=1 ;;
        --model) : ;;  # positional MODEL handled above
        --*) echo "unknown flag: $arg" >&2; exit 2 ;;
    esac
done
[ -n "$MODEL" ] || MODEL=""
# allow `--model PATH` too
if [ -n "${2:-}" ] && [ "${1:-}" = "--model" ]; then MODEL="$2"; fi

step() { printf '\n=== %s ===\n' "$*"; }

step "1/5 build VK=1 binary (ARCH=$VK_ARCH)"
BIN="$C/deepseek_v4"
if [ "$REBUILD" = 1 ] || [ ! -x "$BIN" ]; then
    make -C "$C" -f Makefile.deepseek-v4 deepseek-v4 VK=1 ARCH="$VK_ARCH"
else
    echo "reusing existing $BIN (--rebuild to force)"
fi
NM_OUT="$(nm "$BIN" 2>/dev/null || true)"
if ! grep -q "coli_vk_init" <<< "$NM_OUT"; then
    echo "ERROR: $BIN has no Vulkan symbols — not a VK=1 build" >&2
    exit 2
fi

step "2/5 L1/L2/L4 suites (VK=1)"
for tgt in test-vk-dsv4 test-vk-dsv4-ops test-vk-dsv4-attn \
           test-vk-dsv4-mhc test-vk-dsv4-fallback; do
    echo "--- $tgt ---"
    make -C "$C" -f Makefile.deepseek-v4 "$tgt" VK=1
done

step "3/5 L3 tiny oracle (COLI_DSV4_VK_DENSE=1)"
FIXTURE="$C/deepseek_v4_tiny"
if [ ! -f "$FIXTURE/model.safetensors" ]; then
    echo "tiny fixture missing at $FIXTURE — run: make -C c deepseek-v4-tiny-generate" >&2
    exit 2
fi
COLI_DSV4_VK_DENSE=1 python3 "$C/tests/test_deepseek_v4_tiny.py" \
    --binary "$BIN" --fixture "$FIXTURE"
COLI_DSV4_VK_DENSE=1 python3 "$C/tests/test_deepseek_v4_prefix.py" \
    --binary "$C/deepseek_v4" --fixture "$C/deepseek_v4_tiny"

if [ "$PER_OP_AB" = 1 ]; then
    step "3b/5 per-op A/B on the tiny oracle"
    for ops in qkv wo route head shared attn mhc \
               qkv,wo route,head qkv,wo,route,head shared,mhc; do
        echo "--- COLI_DSV4_VK_OPS=$ops ---"
        COLI_DSV4_VK_DENSE=1 COLI_DSV4_VK_OPS="$ops" \
            python3 "$C/tests/test_deepseek_v4_tiny.py" \
            --binary "$BIN" --fixture "$FIXTURE" >/dev/null
    done
fi

step "4/5 L3 real-checkpoint A/B (tier-off vs tier-on)"
if [ -z "$MODEL" ]; then
    echo "MODEL unset — skipping the real-checkpoint A/B (tiny oracle is the CI gate)"
else
    [ -d "$MODEL" ] || { echo "ERROR: model path not found: $MODEL" >&2; exit 2; }
    OFF="$ROOT/.vk_parity_off.out"; ON="$ROOT/.vk_parity_on.out"
    COLI_TEMP=0 KVSAVE=0 "$BIN" "$MODEL" "$VK_PROMPT" \
        --max-tokens "$VK_MAX_TOKENS" --memory-gb "$VK_MEMORY_GB" \
        > "$OFF" 2> "$ROOT/.vk_parity_off.err"
    COLI_TEMP=0 KVSAVE=0 COLI_DSV4_VK_DENSE=1 "$BIN" "$MODEL" "$VK_PROMPT" \
        --max-tokens "$VK_MAX_TOKENS" --memory-gb "$VK_MEMORY_GB" \
        > "$ON" 2> "$ROOT/.vk_parity_on.err"
    if diff <(grep -v '^TUNE decode' "$OFF") <(grep -v '^TUNE decode' "$ON") > /dev/null; then
        echo "PASS: stdout byte-identical (mod TUNE wall line)"
    else
        echo "FAIL: stdout differs between tier-off and tier-on" >&2
        diff <(grep -v '^TUNE decode' "$OFF") <(grep -v '^TUNE decode' "$ON") >&2 || true
        exit 1
    fi
    drops="$(grep -c 'D7: layer=' "$ROOT/.vk_parity_on.err" || true)"
    reloads="$(grep -cE 'D7 reload|permanent-CPU' "$ROOT/.vk_parity_on.err" || true)"
    echo "tier-on: $drops layers dropped, $reloads reloads/permanent-CPU"
    grep -E 'v4_gpu vk tier=' "$ROOT/.vk_parity_on.err" | tail -1 || true
    rm -f "$OFF" "$ON" "$ROOT/.vk_parity_off.err" "$ROOT/.vk_parity_on.err"
fi

step "5/5 done — L3 parity gate green"
