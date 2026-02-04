### --- Generated with Antigravity ---

# H-Net Hyperparameter Tuning Guide for Telugu (Agglutinative Languages)

This guide lists potential modifications to improve H-Net convergence and segmentation quality, addressing issues like early stopping and fine-grained (character-level) segmentation.

## 1. Optimization & Convergence
These changes address the "converged within 9 epochs" issue.

- **Patience (`PATIENCE`)**
  - **Current**: 2
  - **Recommendation**: **5 - 10**.
  - **Reason**: Validation loss for H-Nets can be noisy as the model explores discrete segmentation choices. 2 epochs is too strict.
  
- **Batch Size (`BATCH_SIZE`)**
  - **Current**: 32
  - **Recommendation**: **64 or 128** (or use gradient accumulation).
  - **Reason**: Larger batches provide more stable gradient estimates, preventing the loss from fluctuating wildly and triggering early stopping.

- **Learning Rate (`lr`)**
  - **Current**: 3e-4
  - **Recommendation**: **5e-4 or 1e-4**.
  - **Reason**: If the model converges to a local minimum (all characters) too locally, a higher LR might help escape. Conversely, if training is unstable, lower it.

- **Weight Decay**
  - **Current**: 0.01
  - **Recommendation**: **0.1**.
  - **Reason**: Higher weight decay can sometimes prevent the model from over-fitting to simple "copy-paste" behaviors (like character-level encoding).

- **Scheduler**
  - **Current**: Custom Trapezoidal (10% Linear Warmup, 80% Constant, 10% Linear Decay).
  - **Recommendation**: **Cosine Decay with Warmup**.
  - **Reason**: The current scheduler keeps the learning rate high for 80% of training. This might prevent the model from settling into finer minima in the later stages. Cosine decay gradually lowers the LR, which is crucial for convergence in complex sequence tasks.
  - **Alternative**: **ReduceLROnPlateau** (if using high patience). This dynamically lowers LR when validation loss stalls, which pairs well with the "early stopping" issues you are facing.

## 2. Segmentation Quality (Addressing Fine-Grained Issues)
These changes address the "one token per byte" issue.

### A. Loss Function Balancing (Requires Code Change in `train.py`)
- **Ratio Loss Coefficient (`alpha`)**
  - **Idea**: The total loss is currently `L = L_lm + L_ratio`.
  - **Recommendation**: Change to `L = L_lm + alpha * L_ratio`. Start with **alpha = 2.0 or 5.0**.
  - **Reason**: the LM loss (predicting the next byte) is "safe" and easy to minimize by selecting every byte. The `ratio_loss` pushes for compression. If the model ignores compression, you need to penalize it more heavily.

### B. Compression Target (`N_compress`) (Requires Config Change)
- **Target Compression Rate**
  - **Current**: Default is usually `[1, 5]` (5x compression).
  - **Recommendation**: For Telugu, which has complex syllables, maybe start deeper? Or stay at 5.
  - **Note**: Ensure `N_compress` is actually acting as intended. If the model is not compressing, it's violating this constraint.

### C. Model Capacity & Dimensions
- **Model Dimensions (`MODEL_DIM`)**
  - **Current**: `"256 256"` (Both levels same size).
  - **Recommendation**: **`"256 512"`** or **`"256 768"`**.
  - **Reason**: The segmentation layer (Layer 1) has to represent *sequences* of bytes (Layer 0) in a single vector. It usually needs *more* dimensions than the byte layer. If D1 = D0, the bottleneck is too tight, forcing the model to select smaller segments (bytes) to preserve information.

### D. Architecture (`MODEL_ARCH`)
- **Current**: `"m2 T4"` (Mamba backbone, Transformer layer?). 
- **Recommendation**: Try **`"m4 T4"`** or **`"m4 T8"`**.
- **Reason**: A stronger backbone (more layers) might be able to learn better contextual representations, allowing it to be more confident in grouping characters.

## 3. Training Strategy
- **Warmup for Compression**
  - **Idea**: For the first few epochs, disable `ratio_loss` or `N_compress` constraints, let the model learn good byte representations, then gradually introduce the compression penalty.
  - **Modification**: Requires editing `train.py` to scale `l_ratio` by `min(1.0, epoch / warmup_epochs)`.

## Summary of High-Impact Experiments to Run First

1.  **Change Config**: Set `PATIENCE=10` and `Batch Size=64`.
2.  **Change Model**: Set `MODEL_DIM="256 512"`.
3.  **Code Edit**: Multiply `l_ratio` by 2.0 or 5.0 in `train.py`.
