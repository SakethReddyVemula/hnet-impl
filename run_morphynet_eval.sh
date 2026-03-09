#!/bin/bash
#SBATCH --job-name=eval_hnet_morphynet
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --output=morphynet_eval_%j.out
#SBATCH --error=morphynet_eval_%j.err
#SBATCH --mail-user=saketh.vemula@research.iiit.ac.in
#SBATCH --mail-type=ALL

# Setup
source ~/saketh/hnet-venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
export CUDA_VISIBLE_DEVICES=0

# Configuration
LANGS=("eng" "fin" "hun" "swe" "hrv" "rus" "mon")

# Model Config
MODEL_DIM="256 256"
MODEL_ARCH="m1 T2"

# MorphyNet Data Config
MORPHYNET_DATA_DIR="morphynet/data"

# Identities
HF_REPO_ID=${HF_REPO_ID:-""} 
HF_UPLOAD_REPO_ID=""
export HF_TOKEN=""

echo "Starting MorphyNet evaluation loop..."
echo "Languages: ${LANGS[*]}"
echo "Source Repo: $HF_REPO_ID"
echo "Target Repo: $HF_UPLOAD_REPO_ID"

for LANG_CODE in "${LANGS[@]}"; do
    echo "------------------------------------------------"
    echo "Processing MorphyNet for $LANG_CODE"
    echo "------------------------------------------------"
    
    OUTPUT_DIR="results/morphynet_eval"
    
    python3 evaluate_morphynet.py \
        --repo_id "$HF_REPO_ID" \
        --upload_repo_id "$HF_UPLOAD_REPO_ID" \
        --lang_code "$LANG_CODE" \
        --output_dir "$OUTPUT_DIR" \
        --model_dim $MODEL_DIM \
        --model_arch $MODEL_ARCH \
        --morphynet_data_dir $MORPHYNET_DATA_DIR \
        --batch_size 32 
        
    echo "Finished $LANG_CODE"
done

echo "Job finished."
