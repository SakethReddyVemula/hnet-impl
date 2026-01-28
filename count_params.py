import sys
import os
import torch

# Add src to path
sys.path.append(os.path.abspath("src"))

from hnet_impl import HNetLM, HNetConfig

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def main():
    # Config matching run_slurm.sh
    cfg_args = {"D": [256, 256], "arch": ["m2", "T4"]}
    
    print(f"\nTesting config: {cfg_args}")
    try:
        c = HNetConfig.create_reasonable_config(**cfg_args)
        m = HNetLM(c)
        total_params = count_parameters(m)
        print(f"Total parameters: {total_params}")
        print(f"Total parameters (M): {total_params / 1e6:.2f}M")
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    main()
