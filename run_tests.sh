
set -e 

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

DATASET="${1:-all}"
OUTPUT_DIR="${2:-results}"
GRID_SIZE="${3:-64}"
MODES="${4:-12}"
BATCH_SIZE="${5:-20}"
N_TRAIN="${6:-800}"
N_TEST="${7:-200}"
N_EPOCHS="${8:-50}"
LR="${9:-0.001}"
SEED="${10:-49}"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Condition-Aware FNO Test Suite Runner${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Configuration:"
echo "  Dataset: $DATASET"
echo "  Output Directory: $OUTPUT_DIR"
echo "  Grid Size: $GRID_SIZE"
echo "  Modes: $MODES"
echo "  Batch Size: $BATCH_SIZE"
echo "  Training Samples: $N_TRAIN"
echo "  Test Samples: $N_TEST"
echo "  Epochs: $N_EPOCHS"
echo "  Learning Rate: $LR"
echo "  Seed: $SEED"
echo ""

mkdir -p "$OUTPUT_DIR"

if ! command -v python &> /dev/null; then
    echo -e "${RED}Error: python command not found${NC}"
    exit 1
fi

if [ ! -f "test_suite.py" ]; then
    echo -e "${RED}Error: test_suite.py not found in current directory${NC}"
    exit 1
fi

echo -e "${YELLOW}Starting test suite...${NC}"
echo ""

python test_suite.py \
    --dataset "$DATASET" \
    --grid_size "$GRID_SIZE" \
    --modes "$MODES" \
    --batch_size "$BATCH_SIZE" \
    --n_train "$N_TRAIN" \
    --n_test "$N_TEST" \
    --n_epochs "$N_EPOCHS" \
    --lr "$LR" \
    --seed "$SEED" \
    --output_dir "$OUTPUT_DIR"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Test suite completed successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "Results saved to: $OUTPUT_DIR/"
    echo "  - JSON results: ${OUTPUT_DIR}/*_results.json"
    echo "  - Plots: ${OUTPUT_DIR}/*_comparison.png"
else
    echo ""
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}Test suite failed with exit code: $EXIT_CODE${NC}"
    echo -e "${RED}========================================${NC}"
    exit $EXIT_CODE
fi

