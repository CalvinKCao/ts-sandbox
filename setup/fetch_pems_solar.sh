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

if [[ -f "$SOLAR_DIR/solar_Alabama.csv" ]]; then
    echo "solar_Alabama.csv already present"
else
    echo "Solar-Energy: install manually unless gdown is available."
    echo "  Expected path: $SOLAR_DIR/solar_Alabama.csv"
    echo "  Source: LSTNet solar_AL (headerless CSV) or Autoformer Drive folder"
    echo "    https://drive.google.com/drive/folders/1Gv1MXjLo5bLGep4bsqDyaNMI2oQC9GH2"
    if command -v gdown >/dev/null 2>&1; then
        tmp="$(mktemp -d)"
        gdown --folder "https://drive.google.com/drive/folders/1Gv1MXjLo5bLGep4bsqDyaNMI2oQC9GH2" -O "$tmp/solar" --remaining-ok || true
        found="$(find "$tmp/solar" \( -iname 'solar_AL*.csv' -o -iname 'solar_AL*.txt' \) 2>/dev/null | head -1)"
        if [[ -n "$found" ]]; then
            cp "$found" "$SOLAR_DIR/solar_Alabama.csv"
            echo "Installed $SOLAR_DIR/solar_Alabama.csv"
        else
            rm -rf "$tmp"
            echo "WARN: gdown did not find solar_AL in Drive folder; copy file manually." >&2
            exit 1
        fi
        rm -rf "$tmp"
    else
        exit 1
    fi
fi

echo "OK: $PEMS_DIR/PEMS04.npz and $SOLAR_DIR/solar_Alabama.csv"
