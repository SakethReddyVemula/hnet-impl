import os
import argparse
import random
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import nested, Tensor as TT
from torch.distributed import device_mesh as tdm, fsdp
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist

from hnet_impl import HNetLM, HNetConfig, ByteTokenizer, completion_sync
from tqdm import tqdm
import wandb
try:
    from huggingface_hub import HfApi
except ImportError:
    HfApi = None

# --- Distributed Init ---
def setup_distributed():
    if "WORLD_SIZE" in os.environ:
        r = int(os.environ["WORLD_SIZE"])
        ws = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        mesh = tdm.init_device_mesh("cuda", (ws,), mesh_dim_names=("dp",))
        return r, ws, local_rank, mesh
    else:
        # Single GPU fallback
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        return 1, 1, 0, None

# --- Data Handling ---
class TextDataset(Dataset):
    def __init__(self, data, max_length=4096):
        self.data = data
        self.tokenizer = ByteTokenizer()
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        tokens = self.tokenizer.encode([text])[0]
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        return tokens

def collate_fn(batch):
    batch = [b for b in batch if len(b) > 1]
    if not batch:
        return None
    
    iids_list = [b[:-1] for b in batch]
    lbls_list = [b[1:] for b in batch]
    
    def NJT(ls: list[TT]):
        return nested.nested_tensor(ls, layout=torch.jagged)

    return NJT(iids_list), NJT(lbls_list).long()

def load_data_from_file(file_path):
    # print(f"Loading {file_path}...") # Optional: too verbose?
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines

def load_and_split_data(data_path, split_ratios=(0.8, 0.1, 0.1), seed=42, local_rank=0):
    path = Path(data_path).expanduser()
    train_data, val_data, test_data = [], [], []

    # Get the language extension from environment (defaults to 'eng' if not set)
    lang = os.getenv("LANG_CODE", "eng")
    lang_ext = f".{lang}"
    
    if path.is_dir():
        # Look for pre-split files
        train_files = list(path.glob(f'train*{lang_ext}'))
        val_files = list(path.glob(f'valid*{lang_ext}')) + list(path.glob(f'dev*{lang_ext}'))
        test_files = list(path.glob(f'test*{lang_ext}'))
        
        if train_files and (val_files or test_files):
            if local_rank == 0:
                print(f"Found pre-split files in {data_path}:")
                print(f"  Train: {[f.name for f in train_files]}")
                print(f"  Valid: {[f.name for f in val_files]}")
                print(f"  Test:  {[f.name for f in test_files]}")
            
            if local_rank == 0: print("Loading train files...")
            for f in train_files: train_data.extend(load_data_from_file(f))
            if local_rank == 0: print(f"Loaded {len(train_data)} training samples.")
            
            if local_rank == 0: print("Loading validation files...")
            for f in val_files: val_data.extend(load_data_from_file(f))
            if local_rank == 0: print(f"Loaded {len(val_data)} validation samples.")
            
            if local_rank == 0: print("Loading test files...")
            for f in test_files: test_data.extend(load_data_from_file(f))
            if local_rank == 0: print(f"Loaded {len(test_data)} test samples.")
            
            # Shuffle training data
            if local_rank == 0: print("Shuffling training data...")
            random.seed(seed)
            random.shuffle(train_data)
            return train_data, val_data, test_data
            
    # Fallback to original logic if no pre-split files found
    data = []
    if path.is_file():
        data = load_data_from_file(path)
    elif path.is_dir():
        for file_path in path.glob('*.txt'):
            data.extend(load_data_from_file(file_path))
    else:
        raise ValueError(f"Invalid data path: {data_path}")

    # Shuffle
    random.seed(seed)
    random.shuffle(data)
    
    n = len(data)
    train_end = int(n * split_ratios[0])
    val_end = int(n * (split_ratios[0] + split_ratios[1]))
    
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    return train_data, val_data, test_data

# --- Training Logic ---
def train_epoch(model, dataloader, optimizer, scheduler, ws, device, log_interval=100, epoch=0, ratio_loss_scale=1.0, warmup_compression_epochs=0):
    model.train()
    total_loss = 0
    steps = 0
    max_steps = (len(dataloader) * (model.c.d_model[0] * 0 + 100)) # dummy
    
    iterator = dataloader
    if dist.get_rank() == 0:
        iterator = tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=False)
        
    for batch in iterator:
        if batch is None: continue
        iids, lbls = batch
        iids, lbls = iids.to(device), lbls.to(device)
        
        with torch.autocast("cuda", torch.bfloat16) if ws > 1 else nullcontext():
            (l_avg, l_sum), extra = model(iids, lbls)
            zero = torch.tensor(0.0, device=device)
            l_ratio = sum([e.loss_ratio for e in extra], zero)
            
            # Weighted ratio loss with simple linear warmup
            alpha = ratio_loss_scale
            if warmup_compression_epochs > 0:
                alpha *= min(1.0, (epoch + 1) / (warmup_compression_epochs + 1))
            
            loss = l_avg + alpha * l_ratio
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if scheduler:
            scheduler.step()
            
        total_loss += loss.item()
        steps += 1
        
        if steps % log_interval == 0 and dist.get_rank() == 0:
            wandb.log({"step_loss": loss.item(), "lr": scheduler.get_last_lr()[0], "step": steps + epoch * len(dataloader)})
            # print(f"Epoch {epoch+1} | Step {steps} | Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
        
    return total_loss / steps if steps > 0 else 0
        
    return total_loss / steps if steps > 0 else 0

def validate(model, dataloader, ws, device):
    model.eval()
    total_loss = 0
    steps = 0
    
    with torch.no_grad():
        for batch in dataloader:
            if batch is None: continue
            iids, lbls = batch
            iids, lbls = iids.to(device), lbls.to(device)
            
            with torch.autocast("cuda", torch.bfloat16) if ws > 1 else nullcontext():
                (l_avg, l_sum), extra = model(iids, lbls)
                zero = torch.tensor(0.0, device=device)
                l_ratio = sum([e.loss_ratio for e in extra], zero)
                loss = l_avg + l_ratio
            
            total_loss += loss.item()
            steps += 1
            
    return total_loss / steps if steps > 0 else float('inf')

def upload_checkpoint(file_path, repo_id, token, subfolder=None, delete_local=False):
    if HfApi is None:
        print("huggingface_hub not installed, skipping upload.")
        return

    if not repo_id or not token:
        print("HF_REPO_ID or HF_TOKEN not set, skipping upload.")
        return

    api = HfApi()
    path_in_repo = os.path.basename(file_path)
    if subfolder:
        path_in_repo = f"{subfolder}/{path_in_repo}"

    try:
        print(f"Uploading {file_path} to {repo_id}...")
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            token=token
        )
        print(f"Successfully uploaded {file_path}")
        
        if delete_local:
            os.remove(file_path)
            print(f"Deleted local file: {file_path}")
            
    except Exception as e:
        print(f"Failed to upload to Hugging Face: {e}")


def main():
    parser = argparse.ArgumentParser(description="Train H-Net on custom dataset")
    parser.add_argument("--data_path", type=str, required=True, help="Path to data file or directory")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=40, help="Maximum number of epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--patience", type=int, default=1, help="Early stopping patience (epochs)")
    parser.add_argument("--log_interval", type=int, default=1, help="Log interval (steps)")
    parser.add_argument("--wandb_project", type=str, default="hnet-training", help="WandB project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="WandB entity")
    parser.add_argument("--max_length", type=int, default=4096, help="Maximum sequence length")
    parser.add_argument("--model_dim", type=int, nargs="+", default=[512, 1024], help="Model dimensions (D)")
    parser.add_argument("--model_arch", type=str, nargs="+", default=["m4", "T9"], help="Model architecture (arch)")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--scheduler", type=str, default="trapezoidal", choices=["trapezoidal", "cosine"], help="LR Scheduler")
    parser.add_argument("--ratio_loss_scale", type=float, default=1.0, help="Scale factor for ratio loss")
    parser.add_argument("--warmup_compression_epochs", type=int, default=0, help="Epochs to warmup compression loss")
    args = parser.parse_args()

    r, ws, local_rank, mesh = setup_distributed()
    device = torch.device("cuda")

    if local_rank == 0:
        args.output_dir = os.path.expanduser(args.output_dir)
        print(f"Training with {ws} GPUs. Output dir: {args.output_dir}")
        os.makedirs(args.output_dir, exist_ok=True)

    if local_rank == 0:
        print("Loading data...")
    train_data, val_data, test_data = load_and_split_data(args.data_path, seed=args.seed, local_rank=local_rank)
    
    if local_rank == 0:
        print(f"Data split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")

    if local_rank == 0: print("Creating datasets...")
    train_dataset = TextDataset(train_data, max_length=args.max_length)
    val_dataset = TextDataset(val_data, max_length=args.max_length)
    test_dataset = TextDataset(test_data, max_length=args.max_length)

    if local_rank == 0: print("Creating dataloaders...")
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=ws, rank=local_rank, shuffle=True) if ws > 1 else None
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=ws, rank=local_rank, shuffle=False) if ws > 1 else None
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, num_replicas=ws, rank=local_rank, shuffle=False) if ws > 1 else None

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn, sampler=train_sampler, shuffle=(train_sampler is None))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn, sampler=val_sampler, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn, sampler=test_sampler, shuffle=False)

    c = HNetConfig.create_reasonable_config(D=args.model_dim, arch=args.model_arch)
    if local_rank == 0: print("Initializing model...")
    with device:
        m = HNetLM(c)

    if local_rank == 0: print("Compiling model backbone...")
    # m.backbone.block_compile(ac=False)
    if ws > 1:
        if local_rank == 0: print("Applying FSDP...")
        m.apply_fsdp(
            m,
            mp_policy=fsdp.MixedPrecisionPolicy(param_dtype=torch.bfloat16),
            reshard_after_forward=False,
            mesh=mesh["dp"],
        )

    steps_per_epoch = len(train_loader)
    max_steps = args.epochs * steps_per_epoch
    
    opt = torch.optim.AdamW(
        [
            dict(params=ls, lr=args.lr * lr_mod)
            for ls, lr_mod in zip(m.split_params_by_hierachy(), c.lambda_s())
        ],
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    
    if args.scheduler == "cosine":
        # Cosine decay with linear warmup (first 10% steps)
        def lr_lambda(step):
            pct = step / max_steps
            if pct < 0.1:
                return pct * 10
            else:
                progress = (pct - 0.1) / 0.9
                return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
        
        lrs = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    else:
        # Trapezoidal
        lrs = torch.optim.lr_scheduler.LambdaLR(
            opt,
            lambda step: (pct := step / max_steps)
            and (pct * 10 if pct < 0.1 else (1 if pct < 0.9 else (1 - pct) * 10)),
        )

    best_val_loss = float('inf')
    patience_counter = 0
    
    if local_rank == 0:
        print("Starting training loop...")
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=args)
        
    for epoch in range(args.epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)
            
        start_time = time.time()
        start_time = time.time()
        train_loss = train_epoch(m, train_loader, opt, lrs, ws, device, log_interval=args.log_interval, epoch=epoch, 
                                 ratio_loss_scale=args.ratio_loss_scale, warmup_compression_epochs=args.warmup_compression_epochs)
        val_loss = validate(m, val_loader, ws, device)
        val_loss = validate(m, val_loader, ws, device)
        
        if ws > 1:
            val_loss_tensor = torch.tensor(val_loss, device=device)
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
            val_loss = val_loss_tensor.item()
        
        if local_rank == 0:
            print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {time.time()-start_time:.2f}s")
            wandb.log({"train_loss": train_loss, "val_loss": val_loss, "epoch": epoch+1})
            
        # Visualization (Run on ALL ranks to keep FSDP sync, but only print on rank 0)
        if local_rank == 0: print("Visualizing segments...")
        vis_samples = val_data[:5] 
        with m.sampling_mode():
            for i, sample in enumerate(vis_samples):
                sample = sample[:100] 
                if local_rank == 0: print(f"Sample {i+1}:")
                from hnet_impl.sampling import colorize_byte_prefill
                try:
                    # This calls m(input), which requires FSDP sync across all ranks
                    vis_output = colorize_byte_prefill(sample, train_dataset.tokenizer, m)
                    if local_rank == 0: print(vis_output)
                except Exception as e:
                    if local_rank == 0: print(f"Visualization failed: {e}")
        
        # Save checkpoint for this epoch
        state_dict = None
        if ws > 1:
             save_policy = fsdp.FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
             with fsdp.StateDictType(m, fsdp.StateDictType.FULL_STATE_DICT, save_policy):
                 state_dict = m.state_dict()
        else:
             state_dict = m.state_dict()

        if local_rank == 0:
            try:
                epoch_ckpt_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pt")
                torch.save(state_dict, epoch_ckpt_path)
                print(f"Saved checkpoint to {epoch_ckpt_path}")
                
                # Upload to HF
                upload_checkpoint(
                    epoch_ckpt_path,
                    repo_id=os.getenv("HF_REPO_ID"),
                    token=os.getenv("HF_TOKEN"),
                    subfolder=os.getenv("HF_SUBFOLDER"),
                    delete_local=os.getenv("HF_DELETE_LOCAL", "0") == "1"
                )
            except Exception as e:
                print(f"Failed to save epoch checkpoint: {e}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                if local_rank == 0:
                    try:
                        torch.save(state_dict, os.path.join(args.output_dir, "best_model.pt"))
                        print(f"New best model saved with val loss {best_val_loss:.4f}")
                        
                        # Upload best model to HF
                        best_model_path = os.path.join(args.output_dir, "best_model.pt")
                        upload_checkpoint(
                            best_model_path,
                            repo_id=os.getenv("HF_REPO_ID"),
                            token=os.getenv("HF_TOKEN"),
                            subfolder=os.getenv("HF_SUBFOLDER"),
                            delete_local=False 
                        )
                    except Exception as e:
                        print(f"Failed to save best model: {e}")
            else:
                patience_counter += 1
                print(f"Val loss increased. Patience: {patience_counter}/{args.patience}")
        
        # Broadcast early stopping decision
        stop_signal = torch.tensor(0, device=device)
        if local_rank == 0 and patience_counter >= args.patience:
            stop_signal = torch.tensor(1, device=device)
        
        if ws > 1:
            dist.broadcast(stop_signal, src=0)
            
        if stop_signal.item() == 1:
            if local_rank == 0:
                print("Early stopping triggered.")
            break

    if local_rank == 0:
        print("Training complete. Running test set...")
        
    test_loss = validate(m, test_loader, ws, device)
    if ws > 1:
        test_loss_tensor = torch.tensor(test_loss, device=device)
        dist.all_reduce(test_loss_tensor, op=dist.ReduceOp.AVG)
        test_loss = test_loss_tensor.item()
        
    if local_rank == 0:
        print(f"Test Loss: {test_loss:.4f}")

    if ws > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
