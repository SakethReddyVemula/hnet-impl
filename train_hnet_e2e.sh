#!/bin/bash
#SBATCH --job-name=all_hnet
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=4-00:00:00
#SBATCH --output=hnet_multilang_%j.out
#SBATCH --error=hnet_multilang_%j.err
#SBATCH --mail-user=saketh.vemula@research.iiit.ac.in
#SBATCH --mail-type=ALL

# 1. Define your languages
LANGS=("san" "kat" "hin" "eng" "fin" "snd" "hun" "swe" "gle" "kor" "ita" "afr" "mal" "spa" "tam" "heb" "hrv" "tel" "rus" "kir" "ell" "tur" "lav" "mon" "isl" "ind" "fas")

# 2. Setup environment once
source ~/saketh/hnet-venv/bin/activate
export CUDA_VISIBLE_DEVICES=0

get_free_port() {
    python -c "import socket; s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.bind(('', 0)); port = s.getsockname()[1]; s.close(); print(port)"
}

# --- HUGGING FACE REPO CONFIGURATION ---
export HF_TOKEN=""
export HF_REPO_ID=""
export HF_DELETE_LOCAL=1

# WandB Configuration
export WANDB_API_KEY=""
export WANDB_PROJECT="hnet-pretraining"

# 3. Start the Loop
for LANG in "${LANGS[@]}"; do
    echo "-------------------------------------------"
    echo "STARTING TRAINING FOR: $LANG"
    echo "-------------------------------------------"

    export LANG_CODE=$LANG
    export DATA_PATH="~/saketh/dataset/${LANG_CODE}"
    export OUTPUT_DIR="~/saketh/checkpoints/hnets/${LANG_CODE}"

    # Prepare directories
    rm -rf "$OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"

    # --- Hugging Face Sub-repo configuration ---
    export HF_SUBFOLDER="${LANG_CODE}"

    # Refresh port for each run to be safe
    export MASTER_PORT=$(get_free_port)
    master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
    export MASTER_ADDR=$master_addr

    # Wandb run name configure for language
    export WANDB_NAME="${LANG_CODE}_hnet"

    echo "Training H-Net for Language: $LANG_CODE"
    echo "Data: $DATA_PATH"
    echo "Output: $OUTPUT_DIR"

    # Hyperparameters
    BATCH_SIZE=128
    EPOCHS=70
    PATIENCE=5
    # Model Configuration (Target: ~3-5M parameters)
    MODEL_DIM="256 256" # "256 512" for ~29M parameters
    MODEL_ARCH="m1 T2" # "m4 T8" for ~29M parameters
    
    # Tuning Hyperparameters
    RATIO_LOSS_SCALE=1.0 # Increase to 2.0 or 5.0 to force compression
    WARMUP_COMPRESSION_EPOCHS=0 # Set to 5 to 10 to delay compression learning
    WEIGHT_DECAY=0.01 # Increase to 0.1 for regularization
    SCHEDULER="cosine" # "cosine" / "trapezoidal" (default)
    CHECKPOINT_INTERVAL=500 # Save checkpoint every N steps (0 = disabled, only epoch-end)
    UPLOAD_BATCH_SIZE=3 # Number of checkpoints to batch into a single HF commit

    # Run Training
    # Using torchrun for distributed training
    torchrun \
        --master_port $MASTER_PORT \
        --nproc_per_node 1 \
        --nnodes 1 \
        train.py \
        --data_path "$DATA_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --batch_size $BATCH_SIZE \
        --epochs $EPOCHS \
        --patience $PATIENCE \
        --wandb_project "$WANDB_PROJECT" \
        --model_dim $MODEL_DIM \
        --model_dim $MODEL_DIM \
        --model_arch $MODEL_ARCH \
        --ratio_loss_scale $RATIO_LOSS_SCALE \
        --warmup_compression_epochs $WARMUP_COMPRESSION_EPOCHS \
        --weight_decay $WEIGHT_DECAY \
        --scheduler $SCHEDULER \
        --checkpoint_interval $CHECKPOINT_INTERVAL \
        --upload_batch_size $UPLOAD_BATCH_SIZE

    rm -r $OUTPUT_DIR
    echo "COMPLETED: $LANG"
done