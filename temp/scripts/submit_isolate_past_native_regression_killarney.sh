#!/bin/bash
# =============================================================================
# Isolate Jul-08 ordinal_norm vs Jul-12 past_native_g* regressions (traffic + electricity).
#
# USAGE (Killarney login node, repo root = $SCRATCH/ts-sandbox):
#   git pull
#   ./temp/scripts/submit_isolate_past_native_regression_killarney.sh ablations electricity
#   ./temp/scripts/submit_isolate_past_native_regression_killarney.sh eval-replay traffic
#   ./temp/scripts/submit_isolate_past_native_regression_killarney.sh fresh-guidance electricity
#   ./temp/scripts/submit_isolate_past_native_regression_killarney.sh compare electricity
#
# Steps map to the 2x2 + guidance isolation plan in reports/isolate_past_native_*/decision_tree.md
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
STORE="${RESULTS_ROOT:-$REPO/results}"
CKPT_ROOT="$STORE/ckpts"

STEP="${1:-help}"
DATASET="${2:-electricity}"

case "$DATASET" in
    traffic)
        G_TAG="g1p5"
        BASELINE_RUN="07-06-4087565-traffic-binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm"
        REGRESSED_RUN="07-12-4208597-traffic-binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm_past_native_g1p5"
        STRIDE2_G_TUNED="binary_noise_sched_ablation_stride2_resize_g1p5"
        PAST_NATIVE_G_TUNED="binary_noise_sched_ablation_past_native_g1p5"
        FRESH_GUIDANCE="binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm_past_native_g1p5_fresh_guidance"
        ;;
    electricity)
        G_TAG="g4p0"
        BASELINE_RUN="07-08-4122619-electricity-binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm"
        REGRESSED_RUN="07-12-4208598-electricity-binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm_past_native_g4p0"
        STRIDE2_G_TUNED="binary_noise_sched_ablation_stride2_resize_g4p0"
        PAST_NATIVE_G_TUNED="binary_noise_sched_ablation_past_native_g4p0"
        FRESH_GUIDANCE="binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm_past_native_g4p0_fresh_guidance"
        ;;
    *)
        echo "Unknown dataset: $DATASET (use traffic or electricity)" >&2
        exit 1
        ;;
esac

CKPT_CONFIG_BASELINE="binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm"
EVAL_ONLY="binary_anchor_ar_patch_decoder_ctx_lb336_hz720_ordinal_norm_past_native_eval_only"

submit_ablations() {
    cd "$REPO"
    # C/D: past_native geometry; E/F: stride-2 resize=true geometry
    ./submit_binary.sh \
        --configs "binary_noise_sched_ablation_past_native_g1p0,${PAST_NATIVE_G_TUNED},binary_noise_sched_ablation_stride2_resize_g1p0,${STRIDE2_G_TUNED}" \
        --datasets "$DATASET" \
        --time 3:00:00
}

submit_eval_replay() {
    cd "$REPO"
    # H: baseline weights, past_native eval geometry (no retrain)
    ./submit_binary.sh \
        --resume \
        --ckpt-config "$CKPT_CONFIG_BASELINE" \
        --configs "$EVAL_ONLY" \
        --datasets "$DATASET" \
        --time 2:00:00
}

submit_fresh_guidance() {
    cd "$REPO"
    # G: full pipeline, guidance trained in-run on past_native+g
    ./submit_binary.sh \
        --configs "$FRESH_GUIDANCE" \
        --datasets "$DATASET" \
        --time 1-12:00:00 \
        --gpu h100
}

run_compare() {
    cd "$REPO"
    source setup/activate_killarney_venv.sh 2>/dev/null || true
    python utils/compare_past_native_regression_isolate.py \
        --dataset "$DATASET" \
        --baseline-run "$BASELINE_RUN" \
        --regressed-run "$REGRESSED_RUN" \
        --ckpt-root "$CKPT_ROOT" \
        --out-dir "reports/isolate_past_native_${DATASET}"
}

case "$STEP" in
    ablations|step2-ablations)
        submit_ablations
        ;;
    eval-replay|step3-eval-replay)
        submit_eval_replay
        ;;
    fresh-guidance|step4-fresh-guidance)
        submit_fresh_guidance
        ;;
    compare|step5-compare)
        run_compare
        ;;
    all)
        submit_ablations
        submit_eval_replay
        echo "After ablations + eval-replay finish, run: $0 fresh-guidance $DATASET"
        echo "Then: $0 compare $DATASET"
        ;;
    help|*)
        cat <<EOF
Usage: $0 <step> [traffic|electricity]

Steps:
  ablations        Submit short 4-epoch ablations (C/D/E/F rows)
  eval-replay      Re-eval baseline ckpt with past_native eval geometry (H row)
  fresh-guidance   Full run with in-run guidance, no ordinal_norm reuse (G row)
  compare          Aggregate metrics -> reports/isolate_past_native_<ds>/
  all              ablations + eval-replay (not fresh-guidance)

Dataset defaults:
  traffic:     baseline $BASELINE_RUN
  electricity: baseline $BASELINE_RUN
EOF
        ;;
esac
