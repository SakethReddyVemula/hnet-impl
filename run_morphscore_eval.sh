#!/bin/bash
#SBATCH --job-name=eval_hnet_morph
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --output=morph_eval_%j.out
#SBATCH --error=morph_eval_%j.err
#SBATCH --mail-user=saketh.vemula@research.iiit.ac.in
#SBATCH --mail-type=ALL

# Setup
source ~/saketh/hnet-venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
export CUDA_VISIBLE_DEVICES=0

# Configuration
LANGS=("eng" "san" "kat" "hin" "fin" "snd" "hun" "swe" "gle" "kor" "ita" "afr" "mal" "spa" "tam" "heb" "hrv" "tel" "rus" "kir" "ell" "tur" "lav" "isl" "ind" "fas")

# Model Config
MODEL_DIM="256 256"
MODEL_ARCH="m1 T2"

# Identities
HF_REPO_ID=${HF_REPO_ID:-""} 
HF_UPLOAD_REPO_ID=""
export HF_TOKEN=""

echo "Starting Morphscore evaluation loop..."
echo "Languages: ${LANGS[*]}"
echo "Source Repo: $HF_REPO_ID"
echo "Target Repo: $HF_UPLOAD_REPO_ID"

for LANG_CODE in "${LANGS[@]}"; do
    echo "------------------------------------------------"
    echo "Processing MorphScore for $LANG_CODE"
    echo "------------------------------------------------"
    
    # We remove language suffix from output dir to match evaluate_morphscore.py assumption where
    # args.output_dir = "results/morphscore_eval" and script appends lang_code.
    # Output will go to: results/morphscore_eval/${LANG_CODE}
    OUTPUT_DIR="results/morphscore_eval"
    
    python3 evaluate_morphscore.py \
        --repo_id "$HF_REPO_ID" \
        --upload_repo_id "$HF_UPLOAD_REPO_ID" \
        --lang_code "$LANG_CODE" \
        --output_dir "$OUTPUT_DIR" \
        --model_dim $MODEL_DIM \
        --model_arch $MODEL_ARCH \
        --batch_size 32 
        
    echo "Finished $LANG_CODE"
done

echo "Job finished."
