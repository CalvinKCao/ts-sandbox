#!/bin/bash
# Login-node resub of failed 11 Sep fullT jobs. Do not scancel 2874036 / 2874118 / 2874119.
source /etc/profile >/dev/null 2>&1 || true
set +e
cd /scratch/ccao87/ts-sandbox-fullT || exit 1

echo "HOST=$(hostname) PWD=$(pwd) DATE=$(date)"
echo "===== protected still up? ====="
squeue -j 2874036,2874118,2874119 -o "%.18i %.30j %.2t %.10M %.10l %R"

echo "===== donors ====="
ls -lh /scratch/ccao87/ts-sandbox/reused/pretrain/binary_window_norm_patch_refine_canvas128_p32x6_pretrain/pretrained_coarse/pretrained_diffusion.pt
ls -lh /scratch/ccao87/ts-sandbox/reused/pretrain/binary_window_norm_patch_refine_canvas128_p32x6_lb104_hz60_pretrain/pretrained_coarse/pretrained_diffusion.pt
grep -n "def _staged_anchor_global_norm" models/diffusion_tsf/pipeline/phases/staged_eval.py

P64=configs/binary_window_norm_patch_refine_canvas128_p64x6_allv_fullT_hz720_nostitch_nopretrain_fixedhp_r0_ms35_kv.yaml
P32=configs/binary_window_norm_patch_refine_canvas128_p32x6_allv_fullT_hz720_nostitch_nopretrain_fixedhp_r0_ms35_kv.yaml

echo ""
echo "===== BINARY eval-only resume (train done, ImportError) ====="
./submit_binary.sh --eval-only --resume --configs "$P64" --datasets ETTh1 --time 3:00:00
./submit_binary.sh --eval-only --resume --configs "$P32" --datasets ETTh2 --time 3:00:00
./submit_binary.sh --eval-only --resume --configs "$P32" --datasets ETTm1,ETTm2 --time 4:00:00
./submit_binary.sh --eval-only --resume --configs "$P32" --datasets PeMS --time 6:00:00

echo ""
echo "===== BINARY OOM resume (same yaml; may OOM again) ====="
./submit_binary.sh --resume --configs "$P64" --datasets electricity --time 2-00:00:00
./submit_binary.sh --resume --configs "$P32" --datasets weather --time 1-00:00:00
./submit_binary.sh --resume --configs "$P32" --datasets solar_Alabama --time 2-00:00:00
./submit_binary.sh --resume --configs "$P32" --datasets traffic --time 2-00:00:00

echo ""
echo "===== BINARY with-pretrain exchange eval-only ====="
./submit_binary.sh --eval-only --resume --configs configs/binary_window_norm_patch_refine_canvas128_p32x6_allv_randwin_lr10_cap1x2x_hz192.yaml --datasets exchange_rate --time 8:00:00
./submit_binary.sh --eval-only --resume --configs configs/binary_window_norm_patch_refine_canvas128_p32x6_allv_randwin_lr10_cap1x2x_hz336.yaml --datasets exchange_rate --time 12:00:00
./submit_binary.sh --eval-only --resume --configs configs/binary_window_norm_patch_refine_canvas128_p32x6_allv_randwin_lr10_cap1x2x_hz720.yaml --datasets exchange_rate --time 1-00:00:00

echo ""
echo "===== BINARY with-pretrain illness H24/36/48 ====="
./submit_binary.sh --configs configs/binary_window_norm_patch_refine_canvas128_p32x6_allv_randwin_lr10_illness_lb104_hz24.yaml --datasets illness --time 6:00:00
./submit_binary.sh --configs configs/binary_window_norm_patch_refine_canvas128_p32x6_allv_randwin_lr10_illness_lb104_hz36.yaml --datasets illness --time 6:00:00
./submit_binary.sh --configs configs/binary_window_norm_patch_refine_canvas128_p32x6_allv_randwin_lr10_illness_lb104_hz48.yaml --datasets illness --time 6:00:00

echo ""
echo "===== MMPD force into same output-dir ====="
./submit_mmpd.sh --mmpd-run-config configs/mmpd_decoder_fullT_lb336_hz96.yaml --datasets ETTm1 --output-dir results/datasets/09-11-mmpd-fullT-lb336-hz96 --time 3:00:00 --no-dataset-extra-args --force
./submit_mmpd.sh --mmpd-run-config configs/mmpd_decoder_fullT_lb336_hz192.yaml --datasets solar_Alabama --output-dir results/datasets/09-11-mmpd-fullT-lb336-hz192 --time 12:00:00 --no-dataset-extra-args --force
./submit_mmpd.sh --mmpd-run-config configs/mmpd_decoder_fullT_lb336_hz336.yaml --datasets ETTh1,ETTh2,ETTm1 --output-dir results/datasets/09-11-mmpd-fullT-lb336-hz336 --time 3:00:00 --no-dataset-extra-args --force
./submit_mmpd.sh --mmpd-run-config configs/mmpd_decoder_fullT_lb336_hz720.yaml --datasets exchange_rate --output-dir results/datasets/09-11-mmpd-fullT-lb336-hz720 --time 1:00:00 --no-dataset-extra-args --force
./submit_mmpd.sh --mmpd-run-config configs/mmpd_decoder_fullT_lb336_hz720.yaml --datasets electricity --output-dir results/datasets/09-11-mmpd-fullT-lb336-hz720 --time 1-00:00:00 --no-dataset-extra-args --force

echo ""
echo "===== BASELINES --force ====="
SUB=configs/compare_baselines_allv_fullT.yaml
EVAL=binary_10pct,test_100pct
BASE=./temp/scripts/submit_baselines_canvas128_killarney.sh
HP=(--lr 1e-4 --d-model 512 --d-ff 512 --e-layers 2 --n-heads 8 --batch-size 32 --dropout 0.1 --train-epochs 10)

"$BASE" --model patchtst --datasets ETTh1 --seq-len 336 --pred-len 96 --subset-yaml "$SUB" --eval-subsets "$EVAL" --force --gpu a100
"$BASE" --model patchtst --datasets ETTh2 --seq-len 336 --pred-len 96 --subset-yaml "$SUB" --eval-subsets "$EVAL" --force --gpu a100
"$BASE" --model patchtst --datasets ETTm1 --seq-len 336 --pred-len 336 --subset-yaml "$SUB" --eval-subsets "$EVAL" --force --gpu a100
"$BASE" --model patchtst --datasets ETTh2 --seq-len 336 --pred-len 720 --subset-yaml "$SUB" --eval-subsets "$EVAL" --force --gpu a100
"$BASE" --model patchtst --datasets dynamic --seq-len 336 --pred-len 720 --subset-yaml "$SUB" --eval-subsets "$EVAL" --force --gpu a100

"$BASE" --model itransformer "${HP[@]}" --datasets ETTh2 --seq-len 336 --pred-len 192 --subset-yaml "$SUB" --eval-subsets "$EVAL" --force --gpu a100
"$BASE" --model itransformer "${HP[@]}" --datasets ETTm1 --seq-len 336 --pred-len 192 --subset-yaml "$SUB" --eval-subsets "$EVAL" --force --gpu a100
"$BASE" --model itransformer "${HP[@]}" --datasets ETTm2 --seq-len 336 --pred-len 192 --subset-yaml "$SUB" --eval-subsets "$EVAL" --force --gpu a100
"$BASE" --model itransformer "${HP[@]}" --datasets weather --seq-len 336 --pred-len 720 --subset-yaml "$SUB" --eval-subsets "$EVAL" --force --gpu a100
"$BASE" --model itransformer "${HP[@]}" --datasets electricity --seq-len 336 --pred-len 720 --subset-yaml "$SUB" --eval-subsets "$EVAL" --force --gpu a100 --time 1-00:00:00
"$BASE" --model itransformer "${HP[@]}" --datasets PeMS --seq-len 96 --pred-len 12 --subset-yaml "$SUB" --eval-subsets "$EVAL" --force --gpu a100

echo ""
echo "===== squeue after submit ====="
squeue -u ccao87 -o "%.18i %.9P %.40j %.2t %.10M %.10l %R"
echo "===== protected still untouched ====="
squeue -j 2874036,2874118,2874119 -o "%.18i %.30j %.2t %.10M %.10l %R"
echo DONE
