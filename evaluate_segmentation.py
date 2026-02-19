import os
import argparse
import json
import json
import glob
import re
import torch
import tqdm
from pathlib import Path
from huggingface_hub import hf_hub_download, HfApi
from hnet_impl import HNetLM, HNetConfig, ByteTokenizer
from torch import nested, Tensor as TT
from torch.utils.data import DataLoader, Dataset
import random

# --- Data Loading (Adapted from train.py) ---
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
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines

def load_validation_data(data_path, lang="eng", split_ratios=(0.8, 0.1, 0.1), seed=42):
    path = Path(data_path).expanduser()
    val_data = []
    lang_ext = f".{lang}"
    
    if path.is_dir():
        # Look for pre-split files
        val_files = list(path.glob(f'valid*{lang_ext}')) + list(path.glob(f'dev*{lang_ext}'))
        
        if val_files:
            print(f"Found pre-split validation files: {[f.name for f in val_files]}")
            for f in val_files: val_data.extend(load_data_from_file(f))
            return val_data
            
    # Fallback: split from single file/dir
    data = []
    if path.is_file():
        data = load_data_from_file(path)
    elif path.is_dir():
        for file_path in path.glob('*.txt'):
            data.extend(load_data_from_file(file_path))
    else:
        raise ValueError(f"Invalid data path: {data_path}")
    
    # Shuffle deterministically
    random.seed(seed)
    random.shuffle(data)
    
    n = len(data)
    train_end = int(n * split_ratios[0])
    val_end = int(n * (split_ratios[0] + split_ratios[1]))
    
    return data[train_end:val_end]

# --- Model Loading ---
def load_model(checkpoint_path, model_dim, model_arch, device):
    config = HNetConfig.create_reasonable_config(D=model_dim, arch=model_arch)
    with device:
        m = HNetLM(config)
    
    try:
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
        new_state_dict = {}
        for k, v in state_dict.items():
            if hasattr(v, 'to_local'):
                v = v.to_local()
            new_state_dict[k] = v
        
        m.load_state_dict(new_state_dict)
        m.eval()
        return m, config
    except Exception as e:
        print(f"Error loading checkpoint {checkpoint_path}: {e}")
        return None, None

def reconstruct_tokens(byte_tokens, layers_indices):
    """
    byte_tokens: list of ints (bytes + potentially BOS)
    layers_indices: list of list of ints. layers_indices[L] contains indices where b=1.
    """
    current_level_tokens = byte_tokens 
    reconstructions = [] 
    
    for l_idx, boundaries in enumerate(layers_indices):
        boundaries.sort()
        next_level_tokens = []
        token_strings = [] 
        
        start = 0
        for end_idx in boundaries:
            if end_idx >= len(current_level_tokens):
                continue
            chunk = current_level_tokens[start : end_idx+1]
            if l_idx == 0:
                try:
                    s = bytearray(chunk).decode('utf-8')
                except:
                    s = "".join([chr(c) if c<128 else f"\\x{c:02x}" for c in chunk])
                token_strings.append(s)
                next_level_tokens.append(s)
            else:
                s = "".join(chunk)
                token_strings.append(s)
                next_level_tokens.append(s)
            start = end_idx + 1
            
        if start < len(current_level_tokens):
            chunk = current_level_tokens[start:]
            if l_idx == 0:
                try:
                     s = bytearray(chunk).decode('utf-8')
                except:
                     s = "".join([chr(c) if c<128 else f"\\x{c:02x}" for c in chunk])
                token_strings.append(s)
                next_level_tokens.append(s)
            else:
                s = "".join(chunk)
                token_strings.append(s)
                next_level_tokens.append(s)
        
        reconstructions.append(token_strings)
        current_level_tokens = next_level_tokens
        
    return reconstructions

def extract_segmentations(model, dataloader, device):
    all_segmentations = []
    
    model.eval()
    
    with torch.inference_mode(), model.sampling_mode():
        for batch in tqdm.tqdm(dataloader, desc="Extracting"):
            if batch is None: continue
            iids, _ = batch
            iids_tensor = iids
            iids = iids.to(device)
            
            # Forward pass
            _, extras = model(iids)
            
            batch_size = iids.size(0)
            
            # Helper to get raw tokens list
            # iids is NestedTensor. 
            # If we can use unbind, that's great. 
            # If not, use values/offsets. 
            # But we need cpu lists for reconstruction.
            # iids.tolist() usually works for NJT to return list of lists?
            # Actually NJT.tolist() might not be fully supported in all versions.
            # Safe bet: unbind
            
            if hasattr(iids, 'unbind'):
                raw_tokens_list = [t.tolist() for t in iids.unbind()]
            else:
                # Fallback if unbind not available on NJT (unlikely given it's used elsewhere)
                # But careful about device. iids is on cuda.
                # move to cpu first?
                # iids.cpu() might not work for NJT directly?
                # Let's hope unbind works.
                 raw_tokens_list = [t.tolist() for t in iids.unbind()] # Assuming works

            b_per_layer = []
            for e in extras:
                if hasattr(e.b, 'unbind'):
                    b_per_layer.append(e.b.unbind())
                else:
                    b_per_layer.append(e.b.unbind())

            # Process batch
            for i in range(batch_size):
                # get boundaries
                sample_bounds = []
                for l in range(len(b_per_layer)):
                    b_tensor = b_per_layer[l][i] 
                    indices = torch.nonzero(b_tensor).squeeze(-1).tolist()
                    sample_bounds.append(indices)
                
                # get tokens
                tokens = raw_tokens_list[i]
                
                # reconstruct
                recons = reconstruct_tokens(tokens, sample_bounds)
                all_segmentations.append(recons)
                
    return all_segmentations

def get_checkpoints(args):
    checkpoints = [] 
    
    if args.local_dir:
        base_dir = os.path.expanduser(args.local_dir)
        if os.path.exists(base_dir):
            files = glob.glob(os.path.join(base_dir, "**", "*.pt"), recursive=True)
            if args.pattern:
                 files = [f for f in files if re.search(args.pattern, os.path.basename(f))]
            
            def sort_key(f):
                base = os.path.basename(f)
                m = re.match(r'checkpoint_(\d+)_(\d+)\.pt$', base)
                if m: return (int(m.group(1)), int(m.group(2)))
                m = re.match(r'checkpoint(\d+)\.pt$', base)
                if m: return (int(m.group(1)), 999999)
                return (0, 0)
            
            files.sort(key=sort_key)
            return [(os.path.basename(f), f) for f in files]
            
    elif args.repo_id:
        api = HfApi(token=args.hf_token)
        print(f"Listing checkpoints from {args.repo_id}...")
        files = api.list_repo_files(repo_id=args.repo_id)
        
        subfolder = args.lang_code
        candidates = [f for f in files if f.endswith(".pt")]
        if subfolder:
            strict_candidates = [f for f in candidates if f.startswith(subfolder + "/")]
            if strict_candidates:
                candidates = strict_candidates
        
        if args.pattern:
            candidates = [f for f in candidates if re.search(args.pattern, os.path.basename(f))]
            
        def sort_key(f):
            base = os.path.basename(f)
            m = re.match(r'checkpoint_(\d+)_(\d+)\.pt$', base)
            if m: return (int(m.group(1)), int(m.group(2)))
            m = re.match(r'checkpoint(\d+)\.pt$', base)
            if m: return (int(m.group(1)), 999999)
            return (0, 0)
            
        candidates.sort(key=sort_key)
        return [(os.path.basename(f), f) for f in candidates] 

    return []

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, default=None, help="HF Repo ID")
    parser.add_argument("--local_dir", type=str, default=None, help="Local directory containing checkpoints")
    parser.add_argument("--data_path", type=str, required=True, help="Path to data")
    parser.add_argument("--output_dir", type=str, default="segmentations_eval")
    parser.add_argument("--model_dim", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--model_arch", type=str, nargs="+", default=["m1", "T2"])
    parser.add_argument("--lang_code", type=str, default="eng")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pattern", type=str, default=None, help="Regex pattern for checkpoints")
    parser.add_argument("--max_samples", type=int, default=None, help="Max validation samples to evaluate")
    parser.add_argument("--keep_downloaded", action="store_true", help="Keep downloaded checkpoints (default: delete if not local)")
    parser.add_argument("--upload_repo_id", type=str, default=None, help="HF Repo ID to upload results to")
    
    args = parser.parse_args()
    
    if not args.repo_id and not args.local_dir:
        parser.error("Must specify either --repo_id or --local_dir")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Setup output
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load Validation Data
    print(f"Loading validation data for {args.lang_code} from {args.data_path}...")
    val_data = load_validation_data(args.data_path, lang=args.lang_code, seed=args.seed)
    if args.max_samples:
        val_data = val_data[:args.max_samples]
    print(f"Loaded {len(val_data)} validation samples.")
    
    # Create DataLoader
    val_dataset = TextDataset(val_data)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)
    
    # List Checkpoints
    checkpoints = get_checkpoints(args) 
    print(f"Found {len(checkpoints)} checkpoints to evaluate.")
    
    if not checkpoints:
        print("No checkpoints found. Check args.")
        return

    # Metadata manifest
    manifest_path = os.path.join(args.output_dir, "manifest.json")
    manifest = []
    if os.path.exists(manifest_path):
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

    for ckpt_name, ckpt_ref in checkpoints:
        output_filename = f"seg_{args.lang_code}_{ckpt_name}.json"
        output_file = os.path.join(args.output_dir, output_filename)
        
        if os.path.exists(output_file):
            print(f"Skipping {ckpt_name}, already exists at {output_file}")
            continue

        print(f"Processing {ckpt_name}...")
        
        # Resolve Path
        ckpt_path = None
        is_temp_download = False
        
        if args.local_dir:
            ckpt_path = ckpt_ref
        else:
            try:
                print(f"Downloading {ckpt_ref}...")
                # Download to specific local dir if we want to delete it easily
                if not args.keep_downloaded:
                    local_dir = os.path.join(args.output_dir, "temp_checkpoints")
                    os.makedirs(local_dir, exist_ok=True)
                    ckpt_path = hf_hub_download(
                        repo_id=args.repo_id, 
                        filename=ckpt_ref, 
                        token=args.hf_token,
                        local_dir=local_dir,
                        local_dir_use_symlinks=False
                    )
                    is_temp_download = True
                else:
                    ckpt_path = hf_hub_download(
                        repo_id=args.repo_id, 
                        filename=ckpt_ref, 
                        token=args.hf_token
                    )
            except Exception as e:
                print(f"Failed to download {ckpt_ref}: {e}")
                continue
            
        # Load Model
        model, config = load_model(ckpt_path, args.model_dim, args.model_arch, device)
        if model is None:
            if is_temp_download and os.path.exists(ckpt_path):
                os.remove(ckpt_path)
            continue
            
        # Extract
        try:
            segmentations = extract_segmentations(model, val_loader, device)
            
            # Save list of lists directly
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(segmentations, f, indent=2)  # Added indent for readability since it's uncompressed
                
            print(f"Saved to {output_file}")
            manifest.append({"checkpoint": ckpt_name, "file": output_filename})
            
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
                
            # Upload to HF if requested
            if args.upload_repo_id:
                queue_for_upload(output_file)
                # Try flush
                flush_uploads(
                    repo_id=args.upload_repo_id,
                    token=args.hf_token,
                    subfolder=args.lang_code,
                    delete_local=True,
                    batch_size=3
                )

        except Exception as e:
             print(f"Error during extraction for {ckpt_name}: {e}")
             import traceback
             traceback.print_exc()
        except Exception as e:
             print(f"Error during extraction for {ckpt_name}: {e}")
             import traceback
             traceback.print_exc()
        finally:
            # Cleanup temp checkpoint
            if is_temp_download and ckpt_path and os.path.exists(ckpt_path):
                # print(f"Deleting temp checkpoint: {ckpt_path}")
                os.remove(ckpt_path)
                try:
                    os.rmdir(os.path.dirname(ckpt_path))
                except:
                    pass
        
    # Final flush
    if args.upload_repo_id:
        flush_uploads(
            repo_id=args.upload_repo_id,
            token=args.hf_token,
            subfolder=args.lang_code,
            delete_local=True,
            force=True
        )
        
    print("Done.")

# --- Batch Upload Logic ---
pending_uploads = []

def queue_for_upload(file_path):
    pending_uploads.append(file_path)

def flush_uploads(repo_id, token, subfolder=None, delete_local=False, force=False, batch_size=3):
    if not pending_uploads:
        return
    if not force and len(pending_uploads) < batch_size:
        return

    try:
        from huggingface_hub import HfApi, CommitOperationAdd
    except ImportError:
        print("huggingface_hub not installed.")
        return

    api = HfApi(token=token)
    operations = []
    files_to_upload = list(pending_uploads)

    for fp in files_to_upload:
        path_in_repo = os.path.basename(fp)
        if subfolder:
            path_in_repo = f"{subfolder}/{path_in_repo}"
        operations.append(CommitOperationAdd(path_or_fileobj=fp, path_in_repo=path_in_repo))

    try:
        filenames = [os.path.basename(fp) for fp in files_to_upload]
        commit_msg = f"Upload {len(files_to_upload)} segmentations: {', '.join(filenames)}"
        print(f"Batch uploading {len(files_to_upload)} files to {repo_id}...")
        api.create_commit(
            repo_id=repo_id,
            operations=operations,
            commit_message=commit_msg,
            repo_type="dataset"
        )
        print(f"Successfully uploaded: {', '.join(filenames)}")

        for fp in files_to_upload:
            pending_uploads.remove(fp)

        if delete_local:
            for fp in files_to_upload:
                if os.path.exists(fp):
                    os.remove(fp)
                    print(f"Deleted local file: {fp}")
    except Exception as e:
        print(f"Failed batch upload: {e}")

if __name__ == "__main__":
    main()
