#!/bin/bash
# Regenerate all paper figures (Figures 2-11).
# Run from the project root, or from anywhere

set -e  # stop on first error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/code/scripts"

echo "Running figure scripts from: $SCRIPT_DIR"

for script in \
    figure_2_rh_conditioning \
    figure_3_amazon_model_construction \
    figure_4_sensitivity_stability \
    figure_5_amazon_rh_model_construction \
    figure_6_sst_importance_by_basin \
    figure_7_average_sensitivity_importance \
    figure_8_sst_importance \
    figure_9_sst_forced_trends \
    figure_10_average_sensitivity_amip \
    figure_11_amip_vs_obs_pattern_correlation; do
    echo "----------------------------------------"
    echo "$(date '+%H:%M:%S')  $script"
    python "$SCRIPT_DIR/${script}.py"
done

echo "----------------------------------------"
echo "$(date '+%H:%M:%S')  All figures complete."
