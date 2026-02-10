#!/bin/bash
# =============================================================================
# Fusion Comparison Experiment Script
# =============================================================================
# Runs evaluation for Late Fusion and Intermediate Fusion (F-Cooper)
# to compare their detection performance on OPV2V dataset.
#
# Prerequisites:
# 1. OpenCOOD installed (conda activate opencood)
# 2. OPV2V validation data downloaded
# 3. Pretrained checkpoints downloaded from benchmark table
# =============================================================================

set -e

# =============================================================================
# USER CONFIGURATION - UPDATE THESE PATHS
# =============================================================================
# Download checkpoints from: https://ucla.app.box.com/v/UCLA-MobilityLab-OPV2V
LATE_FUSION_CHECKPOINT="/path/to/late_fusion_pointpillar_checkpoint"
INTERMEDIATE_FUSION_CHECKPOINT="/path/to/fcooper_pointpillar_checkpoint"

OUTPUT_DIR="./fusion_comparison_results"
# =============================================================================

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}=== OpenCOOD Fusion Comparison Experiment ===${NC}"
echo ""

# Validate paths
if [ ! -d "$LATE_FUSION_CHECKPOINT" ]; then
    echo -e "${RED}ERROR: Late fusion checkpoint not found: $LATE_FUSION_CHECKPOINT${NC}"
    echo "Download from OpenCOOD benchmark and update the path above."
    exit 1
fi

if [ ! -d "$INTERMEDIATE_FUSION_CHECKPOINT" ]; then
    echo -e "${RED}ERROR: Intermediate fusion checkpoint not found: $INTERMEDIATE_FUSION_CHECKPOINT${NC}"
    echo "Download from OpenCOOD benchmark and update the path above."
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Late Fusion
echo -e "${YELLOW}[1/2] Running Late Fusion Evaluation...${NC}"
python opencood/tools/inference.py \
    --model_dir "$LATE_FUSION_CHECKPOINT" \
    --fusion_method late \
    2>&1 | tee "$OUTPUT_DIR/late_fusion_results.log"

echo -e "${GREEN}Late Fusion complete.${NC}"
echo ""

# Intermediate Fusion
echo -e "${YELLOW}[2/2] Running Intermediate Fusion (F-Cooper) Evaluation...${NC}"
python opencood/tools/inference.py \
    --model_dir "$INTERMEDIATE_FUSION_CHECKPOINT" \
    --fusion_method intermediate \
    2>&1 | tee "$OUTPUT_DIR/intermediate_fusion_results.log"

echo ""
echo -e "${GREEN}=== Experiment Complete ===${NC}"
echo ""
echo "Results:"
echo "  Late Fusion:        $OUTPUT_DIR/late_fusion_results.log"
echo "  Intermediate Fusion: $OUTPUT_DIR/intermediate_fusion_results.log"
echo ""
echo -e "${YELLOW}Bandwidth comparison:${NC}"
echo "  Late Fusion:        ~0.024 Mbps"
echo "  Intermediate (F-Cooper): ~72 Mbps (no compression)"
