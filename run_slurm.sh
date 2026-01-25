#!/bin/bash
#SBATCH --job-name=hnet-train
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:A100-SXM4:2
#SBATCH --time=24:00:00
#SBATCH --output=hnet_train_%j.out
#SBATCH --error=hnet_train_%j.err
#SBATCH --mail-user=saketh.vemula@research.iiit.ac.in
#SBATCH --mail-type=ALL
#SBATCH --export=ALL,JOB_DESCRIPTION="Machine translation for Indian languages faces challenges such as rich morphology agglutination free word order and limited annotated resources This project focuses on tokenization strategies for Sanskrit Tamil translation incorporating linguistic knowledge from grammar literature vocabulary and parallel corpora Effective tokenization enables better representation of morphological units compound words and verse structure supporting accurate interpretation of ayurveda itihasa purana poetry prose anvaya philosophy and temple texts",EXPECTED_OUTCOME="The outcome is improved Sanskrit Tamil translation quality through robust tokenization methods that handle morphology compounds and long range dependencies By aligning tokens with linguistic and domain knowledge models better preserve grammatical agreement poetic structure anvaya interpretation and cultural nuance This leads to clearer more consistent translations of ayurvedic concepts historical narratives and literary texts supporting education research digital archives heritage studies and multilingual knowledge dissemination systems"

# Ensure each process sees unique GPUs
export CUDA_VISIBLE_DEVICES=0,1

# Activate H-Net virtual environment
source ~/santam-tok/hnet-venv/bin/activate

# Default Language (can be overridden via --export=ALL,LANG_CODE=hin)
: "${LANG_CODE:=tel}"

# Define directories
# Assumes dataset structure: dataset/{lang}/*.txt
export DATA_PATH="~/santam-tok/dataset/${LANG_CODE}"
export OUTPUT_DIR="~/santam-tok/checkpoints/hnets/${LANG_CODE}"

# Create output directory
mkdir -p $OUTPUT_DIR

get_free_port() {
    python -c "import socket; s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.bind(('', 0)); port = s.getsockname()[1]; s.close(); print(port)"
}

export MASTER_PORT=$(get_free_port)
echo "MASTER_PORT="$MASTER_PORT

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR 

# WandB Configuration
export WANDB_API_KEY=""
export WANDB_PROJECT="hnet-pretraining"

# Hyperparameters
BATCH_SIZE=32
EPOCHS=40
PATIENCE=1

echo "Training H-Net for Language: $LANG_CODE"
echo "Data: $DATA_PATH"
echo "Output: $OUTPUT_DIR"

# Run Training
# Using torchrun for distributed training
torchrun \
    --master_port $MASTER_PORT \
    --nproc_per_node 2 \
    --nnodes 1 \
    train.py \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --patience $PATIENCE \
    --wandb_project "$WANDB_PROJECT"
