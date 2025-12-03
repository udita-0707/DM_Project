#!/bin/bash
# Full Dataset Analysis Runner
# This script orchestrates analysis of the complete 2.2M+ arXiv dataset

set -e  # Exit on error

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Full Dataset Analysis Pipeline${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if full processed dataset exists
PROCESSED_FILE="$PROJECT_ROOT/data/processed/arxiv_processed.csv"
if [ ! -f "$PROCESSED_FILE" ]; then
    echo -e "${YELLOW}⚠ Full processed dataset not found.${NC}"
    echo "Processing full dataset first..."
    echo ""
    
    if [ ! -f "data/raw/arxiv-metadata-oai-snapshot.json" ]; then
        echo -e "${YELLOW}Error: Raw arXiv dataset not found.${NC}"
        echo "Please download from: https://www.kaggle.com/datasets/Cornell-University/arxiv"
        echo "Place it in: data/raw/arxiv-metadata-oai-snapshot.json"
        exit 1
    fi
    
    echo "Step 1: Processing full arXiv dataset..."
    echo "This may take 2-4 hours..."
    source venv/bin/activate
    python3 src/data_acquisition/arxiv_dataset.py --full
    echo -e "${GREEN}✓ Full dataset processed${NC}"
    echo ""
fi

# Activate virtual environment
source venv/bin/activate

# Step 1: Full Dataset Analysis (No Citations Required)
echo -e "${BLUE}Step 1: Analyzing Full Dataset (No Citations Required)${NC}"
echo "This analyzes:"
echo "  - Research area growth"
echo "  - Category distributions"
echo "  - Year distributions"
echo "  - Collaboration patterns"
echo "  - Interdisciplinarity"
echo "  - Keyword extraction"
echo ""

python3 src/analysis/batch_full_dataset_analysis.py \
    --file "$PROCESSED_FILE" \
    --chunk-size 100000 \
    --parallel || python3 src/analysis/batch_full_dataset_analysis.py \
    --chunk-size 100000 \
    --parallel

echo -e "${GREEN}✓ Full dataset analysis complete${NC}"
echo ""

# Step 2: Optional - Enrich Larger Sample with Citations
echo -e "${BLUE}Step 2: Citation Enrichment (Optional)${NC}"
echo ""
echo "Would you like to enrich a larger sample with citations?"
echo "Options:"
echo "  1) Skip (use existing 10K enriched sample)"
echo "  2) Enrich 50K papers (~14 hours)"
echo "  3) Enrich 100K papers (~28 hours)"
echo ""
read -p "Enter choice (1/2/3): " choice

case $choice in
    2)
        echo ""
        echo "Enriching 50K papers..."
        echo "This will take approximately 14 hours"
        echo "You can resume if interrupted using --resume flag"
        echo ""
        
        if [ -z "$SEMANTIC_SCHOLAR_API_KEY" ]; then
            echo -e "${YELLOW}⚠ SEMANTIC_SCHOLAR_API_KEY not set${NC}"
            read -p "Enter your API key: " api_key
            export SEMANTIC_SCHOLAR_API_KEY="$api_key"
        fi
        
        python3 src/data_acquisition/optimized_batch_enrichment.py \
            --sample-size 50000 \
            --strategy stratified \
            --chunk-size 10000
        ;;
    3)
        echo ""
        echo "Enriching 100K papers..."
        echo "This will take approximately 28 hours"
        echo "You can resume if interrupted using --resume flag"
        echo ""
        
        if [ -z "$SEMANTIC_SCHOLAR_API_KEY" ]; then
            echo -e "${YELLOW}⚠ SEMANTIC_SCHOLAR_API_KEY not set${NC}"
            read -p "Enter your API key: " api_key
            export SEMANTIC_SCHOLAR_API_KEY="$api_key"
        fi
        
        python3 src/data_acquisition/optimized_batch_enrichment.py \
            --sample-size 100000 \
            --strategy stratified \
            --chunk-size 10000
        ;;
    *)
        echo "Skipping citation enrichment"
        ;;
esac

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Analysis Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Results saved to:"
echo "  - data/processed/analysis_results/full_dataset_analysis/"
echo ""
echo "To view results, check the generated CSV and JSON files."
