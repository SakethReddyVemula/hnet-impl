#!/bin/bash
#SBATCH --job-name=eval_hnet_seg
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --output=seg_eval_%j.out
#SBATCH --error=seg_eval_%j.err
#SBATCH --mail-user=saketh.vemula@research.iiit.ac.in
#SBATCH --mail-type=ALL

# Setup
source ~/saketh/hnet-venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
export CUDA_VISIBLE_DEVICES=0

# Configuration
LANGS=("eng" "san" "kat" "hin" "fin" "snd" "hun" "swe" "gle" "kor" "ita" "afr" "mal" "spa" "tam" "heb" "hrv" "tel" "rus" "kir" "ell" "tur" "lav" "mon" "isl" "ind" "fas")

# Model Config
MODEL_DIM="256 256"
MODEL_ARCH="m1 T2"

# Identities
HF_REPO_ID=${HF_REPO_ID:-""} 
HF_UPLOAD_REPO_ID=""
export HF_TOKEN="" 

echo "Starting evaluation loop..."
echo "Languages: ${LANGS[*]}"
echo "Validation Data Base: ~/saketh/dataset/"
echo "Source Repo: $HF_REPO_ID"
echo "Target Repo: $HF_UPLOAD_REPO_ID"

for LANG_CODE in "${LANGS[@]}"; do
    echo "------------------------------------------------"
    echo "Processing $LANG_CODE"
    echo "------------------------------------------------"
    
    DATA_PATH=~/saketh/dataset/${LANG_CODE}
    OUTPUT_DIR="results/segmentations/${LANG_CODE}"
    mkdir -p $OUTPUT_DIR
    
    # Run evaluation
    # Note: --keep_downloaded is NOT set, so it defaults to cleaning up temp checkpoints if not local
    # We pass upload_repo_id to trigger upload and local result deletion
    
    python3 evaluate_segmentation.py \
        --repo_id "$HF_REPO_ID" \
        --upload_repo_id "$HF_UPLOAD_REPO_ID" \
        --lang_code "$LANG_CODE" \
        --data_path "$DATA_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --model_dim $MODEL_DIM \
        --model_arch $MODEL_ARCH \
        --batch_size 32 
        
    echo "Finished $LANG_CODE"
done

echo "Job finished."
