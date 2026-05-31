#!/usr/bin/env bash
# Fetch PeMS (NPZ) and Solar-Energy benchmarks into ./datasets/
# Run from repo root on a machine with outbound network (login node is fine).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"  # repo root
DEST="${1:-$ROOT/datasets}"
PEMS_DIR="$DEST/PeMS"
SOLAR_DIR="$DEST/solar_Alabama"

mkdir -p "$PEMS_DIR" "$SOLAR_DIR"

if [[ -f "$PEMS_DIR/PEMS04.npz" ]]; then
    echo "PEMS04.npz already present"
else
    echo "Downloading PEMS04.npz (ASTGNN / iTransformer benchmark)..."
    curl -fsSL -o "$PEMS_DIR/PEMS04.npz" \
        "https://github.com/guoshnBJTU/ASTGNN/raw/main/data/PEMS04/PEMS04.npz"
    echo "Installed $PEMS_DIR/PEMS04.npz"
fi

SOLAR_LSTNET_GZ="https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/solar-energy/solar_AL.txt.gz"

if [[ -f "$SOLAR_DIR/solar_Alabama.csv" ]]; then
    echo "solar_Alabama.csv already present"
elif [[ -f "$SOLAR_DIR/solar_AL.csv" ]]; then
  ln -sf solar_AL.csv "$SOLAR_DIR/solar_Alabama.csv" 2>/dev/null || cp "$SOLAR_DIR/solar_AL.csv" "$SOLAR_DIR/solar_Alabama.csv"
  echo "solar_Alabama.csv already present (from solar_AL.csv)"
else
    echo "Downloading solar_AL from LSTNet multivariate-time-series-data (curl)..."
    tmp_gz="$(mktemp)"
    if curl -fsSL -o "$tmp_gz" "$SOLAR_LSTNET_GZ" && gunzip -c "$tmp_gz" >"$SOLAR_DIR/solar_Alabama.csv"; then
        rm -f "$tmp_gz"
        echo "Installed $SOLAR_DIR/solar_Alabama.csv"
    else
        rm -f "$tmp_gz"
        echo "curl install failed; trying gdown (Autoformer Drive) if available..." >&2
        if command -v gdown >/dev/null 2>&1; then
            tmp="$(mktemp -d)"
            gdown --folder "https://drive.google.com/drive/folders/1Gv1MXjLo5bLGep4bsqDyaNMI2oQC9GH2" -O "$tmp/solar" --remaining-ok || true
            found="$(find "$tmp/solar" \( -iname 'solar_AL*.csv' -o -iname 'solar_AL*.txt' \) 2>/dev/null | head -1)"
            if [[ -n "$found" ]]; then
                cp "$found" "$SOLAR_DIR/solar_Alabama.csv"
                echo "Installed $SOLAR_DIR/solar_Alabama.csv"
            else
                rm -rf "$tmp"
                echo "WARN: gdown did not find solar_AL; copy file manually." >&2
                exit 1
            fi
            rm -rf "$tmp"
        else
            echo "  Expected: $SOLAR_DIR/solar_Alabama.csv" >&2
            echo "  Manual: curl -fsSL -o /tmp/solar_AL.txt.gz '$SOLAR_LSTNET_GZ' && gunzip -c /tmp/solar_AL.txt.gz >'$SOLAR_DIR/solar_Alabama.csv'" >&2
            exit 1
        fi
    fi
fi

echo "OK: $PEMS_DIR/PEMS04.npz and $SOLAR_DIR/solar_Alabama.csv"

if [[ -f "$ROOT/DALIA/Forecast100X.pt" || -f "$ROOT/datasets/DALIA/Forecast100X.pt" ]]; then
    echo ""
    echo "DALIA: converting Forecast100 tensors -> datasets/dalia/dalia.csv"
    bash "$ROOT/setup/convert_dalia_to_csv.sh" --datasets-dir "$DEST"
fi
