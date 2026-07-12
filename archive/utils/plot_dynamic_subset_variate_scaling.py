#!/usr/bin/env python3
"""Backward-compatible wrapper for dynamic-only plots."""

from plot_subset_variate_scaling import main

if __name__ == "__main__":
    import sys

    if "--datasets" not in sys.argv and "-h" not in sys.argv and "--help" not in sys.argv:
        sys.argv[1:1] = ["--datasets", "dynamic", "--out-dir", "reports/dynamic_subset_variate_scaling"]
    raise SystemExit(main())
