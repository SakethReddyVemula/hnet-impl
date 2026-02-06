#!/bin/bash
#SBATCH --job-name=santam-tok
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100-SXM4:1
#SBATCH --time=4-00:00:00
#SBATCH --partition=dibdp
#SBATCH --output=hnet_multilang_%j.out
#SBATCH --error=hnet_multilang_%j.err
#SBATCH --mail-user=saketh.vemula@research.iiit.ac.in
#SBATCH --mail-type=ALL
#SBATCH --export=ALL,JOB_DESCRIPTION="Machine translation for Indian languages faces challenges such as rich morphology agglutination free word order and limited annotated resources This project focuses on tokenization strategies for Sanskrit Tamil translation incorporating linguistic knowledge from grammar literature vocabulary and parallel corpora Effective tokenization enables better representation of morphological units compound words and verse structure supporting accurate interpretation of ayurveda itihasa purana poetry prose anvaya philosophy and temple texts",EXPECTED_OUTCOME="The outcome is improved Sanskrit Tamil translation quality through robust tokenization methods that handle morphology compounds and long range dependencies By aligning tokens with linguistic and domain knowledge models better preserve grammatical agreement poetic structure anvaya interpretation and cultural nuance This leads to clearer more consistent translations of ayurvedic concepts historical narratives and literary texts supporting education research digital archives heritage studies and multilingual knowledge dissemination systems"

# 1. Define your languages
LANGS=("tel", "hin", "eng", "fin")

# 2. Setup environment once
source ~/santam-tok/hnet-venv/bin/activate
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
    export DATA_PATH="~/santam-tok/dataset/${LANG_CODE}"
    export OUTPUT_DIR="~/santam-tok/checkpoints/hnets/${LANG_CODE}"

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
    BATCH_SIZE=64
    EPOCHS=50
    PATIENCE=5
    # Model Configuration (Target: ~3-5M parameters)
    MODEL_DIM="256 256" # "256 512" for ~29M parameters
    MODEL_ARCH="m2 T4" # "m4 T8" for ~29M parameters
    
    # Tuning Hyperparameters
    RATIO_LOSS_SCALE=5.0 # Increase to 2.0 or 5.0 to force compression
    WARMUP_COMPRESSION_EPOCHS=2 # Set to 5 to 10 to delay compression learning
    WEIGHT_DECAY=0.01 # Increase to 0.1 for regularization
    SCHEDULER="cosine" # "cosine" / "trapezoidal" (default)

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
        --scheduler $SCHEDULER

    rm -r $OUTPUT_DIR
    echo "COMPLETED: $LANG"
done