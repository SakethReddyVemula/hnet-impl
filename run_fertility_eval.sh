#!/bin/bash
#SBATCH --job-name=eval_fertility
#SBATCH --ntasks-per-node=1
#SBATCH --time=2-00:00:00
#SBATCH --output=fertility_eval_%j.out
#SBATCH --error=fertility_eval_%j.err

# Setup
source ~/saketh/hnet-venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# Configuration
LANGS=("eng" "san" "kat" "hin" "fin" "snd" "hun" "swe" "gle" "kor" "ita" "afr" "mal" "spa" "tam" "heb" "hrv" "tel" "rus" "kir" "ell" "tur" "lav" "mon" "isl" "ind" "fas")

# Identities
HF_REPO_ID="" # The generated segments dataset
export HF_TOKEN="" 

echo "Starting Fertility evaluation loop..."
echo "Languages: ${LANGS[*]}"
echo "Source Repo: $HF_REPO_ID"

OUTPUT_DIR="results/fertility_eval"
mkdir -p "$OUTPUT_DIR"

for LANG_CODE in "${LANGS[@]}"; do
    echo "------------------------------------------------"
    echo "Processing $LANG_CODE"
    echo "------------------------------------------------"
    
    OUTPUT_FILE="${OUTPUT_DIR}/fertility_${LANG_CODE}.csv"
    
    python3 evaluate_fertility.py \
        --repo_id "$HF_REPO_ID" \
        --lang_code "$LANG_CODE" \
        --output_file "$OUTPUT_FILE" 
        
    echo "Finished $LANG_CODE"
done

echo "Job finished."
