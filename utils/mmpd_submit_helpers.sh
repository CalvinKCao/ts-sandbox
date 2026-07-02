# Login-node helpers for MMPD submit scripts (no venv / numpy required).

mmpd_dataset_file_path() {
    local ds="$1" repo="$2"
    case "$ds" in
        ETTh1) echo "$repo/datasets/ETT-small/ETTh1.csv" ;;
        ETTh2) echo "$repo/datasets/ETT-small/ETTh2.csv" ;;
        ETTm1) echo "$repo/datasets/ETT-small/ETTm1.csv" ;;
        ETTm2) echo "$repo/datasets/ETT-small/ETTm2.csv" ;;
        illness) echo "$repo/datasets/illness/national_illness.csv" ;;
        exchange_rate) echo "$repo/datasets/exchange_rate/exchange_rate.csv" ;;
        weather) echo "$repo/datasets/weather/weather.csv" ;;
        electricity) echo "$repo/datasets/electricity/electricity.csv" ;;
        traffic) echo "$repo/datasets/traffic/traffic.csv" ;;
        PeMS) echo "$repo/datasets/PeMS/PEMS04.npz" ;;
        solar_Alabama) echo "$repo/datasets/solar_Alabama/solar_Alabama.csv" ;;
        dalia) echo "$repo/datasets/dalia/dalia.csv" ;;
        dynamic) echo "$repo/datasets/dynamic/dynamic_500K.csv" ;;
        *) return 1 ;;
    esac
}

read_mmpd_yaml_datasets() {
    local config="$1"
    awk '
        /^mmpd:/ { in_mmpd=1; next }
        in_mmpd && /^[^[:space:]#]/ { exit }
        in_mmpd && /^  datasets:/ { in_list=1; next }
        in_list && /^  [^ ]/ { exit }
        in_list && /^    - / {
            item=$0
            sub(/^    - /, "", item)
            gsub(/"/, "", item)
            if (n++) printf ","
            printf "%s", item
        }
    ' "$config"
}

# Per-dataset Slurm wall times (tune 7x10ep + train 20ep + 100-sample eval).
mmpd_dataset_wall_time() {
    local ds="$1" default_wall="$2"
    case "$ds" in
        dynamic) echo "12:00:00" ;;
        weather|electricity|traffic|PeMS|dalia) echo "6:00:00" ;;
        *) echo "$default_wall" ;;
    esac
}

# Extra eval_mmpd_gaussian_anchor args for heavy test sets (one arg per line).
mmpd_dataset_worker_extra_args() {
    local ds="$1"
    case "$ds" in
        dynamic) printf '%s\n' --test-max-items 512 ;;
    esac
}
