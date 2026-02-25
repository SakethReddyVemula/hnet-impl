#!/bin/bash
#SBATCH --job-name=eval_toklen
#SBATCH --ntasks-per-node=1
#SBATCH --time=2-00:00:00
#SBATCH --output=token_length_eval_%j.out
#SBATCH --error=token_length_eval_%j.err

# Setup
source ~/saketh/hnet-venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# Configuration
LANGS=("eng" "san" "kat" "hin" "fin" "snd" "hun" "swe" "gle" "kor" "ita" "afr" "mal" "spa" "tam" "heb" "hrv" "tel" "rus" "kir" "ell" "tur" "lav" "mon" "isl" "ind" "fas")

# Identities
HF_REPO_ID="" # The generated segments dataset
export HF_TOKEN="" 

echo "Starting Token Length evaluation loop..."
echo "Languages: ${LANGS[*]}"
echo "Source Repo: $HF_REPO_ID"

OUTPUT_DIR="results/token_length_eval"
mkdir -p "$OUTPUT_DIR"

for LANG_CODE in "${LANGS[@]}"; do
    echo "------------------------------------------------"
    echo "Processing $LANG_CODE"
    echo "------------------------------------------------"
    
    OUTPUT_FILE="${OUTPUT_DIR}/avg_token_length_${LANG_CODE}.csv"
    DIST_DIR="${OUTPUT_DIR}/distributions_${LANG_CODE}"
    
    python3 evaluate_token_length.py \
        --repo_id "$HF_REPO_ID" \
        --lang_code "$LANG_CODE" \
        --output_file "$OUTPUT_FILE" \
        --dist_dir "$DIST_DIR"
        
    echo "Finished $LANG_CODE"
done

echo "Job finished."
