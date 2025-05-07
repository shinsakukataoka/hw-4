#!/usr/bin/env bash

set -euo pipefail

TRIALS=50

for exp in {1..10}; do
    n=$((2 ** exp))
    echo "Running n=$n ("$TRIALS" trials)" >&2
    ./matmul "$n" "$TRIALS"
done

echo "All runs complete.  Now generate the plot…" >&2
python plot_results.py

