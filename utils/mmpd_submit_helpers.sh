# Login-node helpers for MMPD submit scripts (no venv / numpy required).

mmpd_dataset_file_path() {
    local ds="$1" repo="$2"
    case "$ds" in
        ETTh1) echo "$repo/datasets/ETT-small/ETTh1.csv" ;;
        ETTh2) echo "$repo/datasets/ETT-small/ETTh2.csv" ;;
        ETTm1) echo "$repo/datasets/ETT-small/ETTm1.csv" ;;
        ETTm2) echo "$repo/datasets/ETT-small/ETTm2.csv" ;;
        illness) echo "$repo/datasets/illness/national_illness.csv" ;;
        exchange|exchange_rate) echo "$repo/datasets/exchange_rate/exchange_rate.csv" ;;
        weather) echo "$repo/datasets/weather/weather.csv" ;;
        electricity) echo "$repo/datasets/electricity/electricity.csv" ;;
        traffic) echo "$repo/datasets/traffic/traffic.csv" ;;
        PeMS) echo "$repo/datasets/PeMS/PEMS04.npz" ;;
        solar_Alabama) echo "$repo/datasets/solar_Alabama/solar_Alabama.csv" ;;
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

# Per-dataset Slurm wall times.
# lb96 default: tune 7x10ep + train 20ep + 100-sample eval.
# lb336/hz720 callers pass --time 24:00:00+; ~1h/epoch train plus heavy eval.
# Caller --time is a ceiling: fat-set bumps never exceed the requested wall.
_mmpd_wall_to_sec() {
    local t="$1" days=0 rest h=0 m=0 s=0
    if [[ "$t" == *-* ]]; then
        days="${t%%-*}"
        rest="${t#*-}"
    else
        rest="$t"
    fi
    IFS=':' read -r a b c <<< "$rest"
    if [[ -n "${c:-}" ]]; then
        h="$a"; m="$b"; s="$c"
    elif [[ -n "${b:-}" ]]; then
        h=0; m="$a"; s="$b"
    else
        h=0; m=0; s="$a"
    fi
    echo $(( days * 86400 + 10#$h * 3600 + 10#$m * 60 + 10#$s ))
}

mmpd_dataset_wall_time() {
    local ds="$1" default_wall="$2"
    local long=0 bumped="$default_wall"
    # Multi-day --time is the requested wall; do not replace with H96 6h/12h floors.
    if [[ "$default_wall" == *-* ]]; then
        echo "$default_wall"
        return
    fi
    if [[ "$default_wall" =~ ^(2[4-9]|48): ]]; then
        long=1
    fi
    if [[ "$long" -eq 1 ]]; then
        case "$ds" in
            dynamic) bumped="48:00:00" ;;
            weather|electricity|traffic|PeMS) bumped="24:00:00" ;;
        esac
        echo "$bumped"
        return
    fi
    case "$ds" in
        dynamic) bumped="12:00:00" ;;
        weather|electricity|traffic|PeMS) bumped="6:00:00" ;;
    esac
    if [[ "$(_mmpd_wall_to_sec "$bumped")" -gt "$(_mmpd_wall_to_sec "$default_wall")" ]]; then
        bumped="$default_wall"
    fi
    echo "$bumped"
}

# Extra eval_mmpd_gaussian_anchor args for heavy test sets (one arg per line).
mmpd_dataset_worker_extra_args() {
    local ds="$1"
    case "$ds" in
        dynamic) printf '%s\n' --test-max-items 512 ;;
    esac
}

mmpd_pick_l40s_partition() {
    local need_s="$1" part max_wall max_s best="" best_s=0
    while read -r part max_wall; do
        [[ "$part" == gpubase_l40s_b* ]] || continue
        part="${part%\*}"
        max_s="$(_mmpd_wall_to_sec "$max_wall")"
        if [[ "$max_s" -ge "$need_s" ]]; then
            if [[ -z "$best" || "$max_s" -lt "$best_s" ]]; then
                best="$part"
                best_s="$max_s"
            fi
        fi
    done < <(sinfo -h -o "%P %l" 2>/dev/null || true)
    if [[ -z "$best" ]]; then
        echo "ERROR: no gpubase_l40s_b* partition allows wall ${need_s}s" >&2
        return 1
    fi
    echo "$best"
}
