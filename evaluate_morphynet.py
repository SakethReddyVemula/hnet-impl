import os
import argparse
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
import pandas as pd
import numpy as np

# Import MorphyNetScore
from morphynet_score import MorphyNetScore

# Language Mapping
def get_full_lang(code):
    mapping = {
        "eng": "english", "san": "sanskrit", "kat": "georgian", "hin": "hindi", "fin": "finnish", 
        "snd": "sindhi", "hun": "hungarian", "swe": "swedish", "gle": "irish", "kor": "korean",
        "ita": "italian", "afr": "afrikaans", "mal": "malayalam", "spa": "spanish", "tam": "tamil",
        "heb": "hebrew", "hrv": "croatian", "tel": "telugu", "rus": "russian", "kir": "kirghiz",
        "ell": "greek", "tur": "turkish", "lav": "latvian", "mon": "mongolian", "isl": "icelandic",
        "ind": "indonesian", "fas": "persian"
    }
    return mapping.get(code)

# --- Data Loading ---
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
    batch = [b for b in batch if len(b) > 0]
    if not batch: return None
    iids_list = [torch.cat([b, torch.tensor([32], dtype=b.dtype, device=b.device)]) for b in batch]
    def NJT(ls: list[TT]): return nested.nested_tensor(ls, layout=torch.jagged)
    return NJT(iids_list), None

# --- Model Loading ---
def load_model(checkpoint_path, model_dim, model_arch, device):
    config = HNetConfig.create_reasonable_config(D=model_dim, arch=model_arch)
    with device:
        m = HNetLM(config)
    try:
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
        new_state_dict = {}
        for k, v in state_dict.items():
            if hasattr(v, 'to_local'): v = v.to_local()
            new_state_dict[k] = v
        m.load_state_dict(new_state_dict)
        m.eval()
        return m, config
    except Exception as e:
        print(f"Error loading checkpoint {checkpoint_path}: {e}")
        return None, None

def reconstruct_tokens(byte_tokens, layers_indices):
    current_level_tokens = byte_tokens 
    reconstructions = [] 
    for l_idx, boundaries in enumerate(layers_indices):
        boundaries.sort()
        next_level_tokens = []
        token_strings = [] 
        start = 0
        for end_idx in boundaries:
            if end_idx >= len(current_level_tokens): continue
            chunk = current_level_tokens[start : end_idx+1]
            if l_idx == 0:
                try: s = bytearray(chunk).decode('utf-8')
                except: s = "".join([chr(c) if c<128 else f"\\x{c:02x}" for c in chunk])
            else:
                s = "".join(chunk)
            token_strings.append(s)
            next_level_tokens.append(s)
            start = end_idx + 1
        if start < len(current_level_tokens):
            chunk = current_level_tokens[start:]
            if l_idx == 0:
                try: s = bytearray(chunk).decode('utf-8')
                except: s = "".join([chr(c) if c<128 else f"\\x{c:02x}" for c in chunk])
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
            iids = iids.to(device)
            _, extras = model(iids)
            batch_size = iids.size(0)
            
            if hasattr(iids, 'unbind'): raw_tokens_list = [t.tolist() for t in iids.unbind()]
            else: raw_tokens_list = [t.tolist() for t in iids.unbind()]
                
            b_per_layer = []
            for e in extras:
                if hasattr(e.b, 'unbind'): b_per_layer.append(e.b.unbind())
                else: b_per_layer.append(e.b.unbind())
                    
            for i in range(batch_size):
                sample_bounds = []
                for l in range(len(b_per_layer)):
                    b_tensor = b_per_layer[l][i] 
                    indices = torch.nonzero(b_tensor).squeeze(-1).tolist()
                    if isinstance(indices, int): indices = [indices]
                    sample_bounds.append(indices)
                tokens = raw_tokens_list[i]
                recons = reconstruct_tokens(tokens, sample_bounds)
                
                cleaned_recons = []
                for layer in recons:
                    cl = [t.replace("\\xfe", "") for t in layer]
                    if len(cl) > 0:
                        cl[-1] = cl[-1].rstrip(" ")
                    cl = [t for t in cl if t]
                    cleaned_recons.append(cl)
                    
                all_segmentations.append(cleaned_recons)
    return all_segmentations

def get_checkpoints(args):
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
            if strict_candidates: candidates = strict_candidates
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

# --- Batch Upload Logic ---
pending_uploads = []
def queue_for_upload(file_path):
    pending_uploads.append(file_path)

def flush_uploads(repo_id, token, subfolder=None, delete_local=False, force=False, batch_size=3):
    if not pending_uploads: return
    if not force and len(pending_uploads) < batch_size: return
    try: from huggingface_hub import HfApi, CommitOperationAdd
    except ImportError: return
    api = HfApi(token=token)
    operations = []
    files_to_upload = list(pending_uploads)
    for fp in files_to_upload:
        path_in_repo = os.path.basename(fp)
        if subfolder: path_in_repo = f"{subfolder}/{path_in_repo}"
        operations.append(CommitOperationAdd(path_or_fileobj=fp, path_in_repo=path_in_repo))
    try:
        filenames = [os.path.basename(fp) for fp in files_to_upload]
        commit_msg = f"Upload {len(files_to_upload)} morphynet segmentations"
        print(f"Batch uploading {len(files_to_upload)} files to {repo_id}...")
        api.create_commit(repo_id=repo_id, operations=operations, commit_message=commit_msg, repo_type="dataset")
        print(f"Successfully uploaded: {', '.join(filenames)}")
        for fp in files_to_upload: pending_uploads.remove(fp)
        if delete_local:
            for fp in files_to_upload:
                if os.path.exists(fp):
                    os.remove(fp)
    except Exception as e:
        print(f"Failed batch upload: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, default=None, help="HF Repo ID for models")
    parser.add_argument("--local_dir", type=str, default=None, help="Local directory containing checkpoints")
    parser.add_argument("--morphynet_data_dir", type=str, default="morphynet/data", help="Path to morphynet data")
    parser.add_argument("--output_dir", type=str, default="results/morphynet_eval")
    parser.add_argument("--model_dim", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--model_arch", type=str, nargs="+", default=["m1", "T2"])
    parser.add_argument("--lang_code", type=str, default="eng")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--pattern", type=str, default=None, help="Regex pattern for checkpoints")
    parser.add_argument("--keep_downloaded", action="store_true", help="Keep downloaded checkpoints")
    parser.add_argument("--upload_repo_id", type=str, default=None, help="HF Repo ID to upload segmentations to")
    args = parser.parse_args()
    
    if not args.repo_id and not args.local_dir: parser.error("Must specify either --repo_id or --local_dir")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    lang_output_dir = os.path.join(args.output_dir, args.lang_code)
    os.makedirs(lang_output_dir, exist_ok=True)
    
    print(f"Loading MorphyNet data for {args.lang_code} from {args.morphynet_data_dir}...")
    try:
        morph_evaluator = MorphyNetScore()
        infl_path = os.path.join(args.morphynet_data_dir, args.lang_code, f"{args.lang_code}.inflectional.v1.tsv")
        deriv_path = os.path.join(args.morphynet_data_dir, args.lang_code, f"{args.lang_code}.derivational.v1.tsv")
        
        infl_dataset = morph_evaluator.load_inflectional(infl_path) if os.path.exists(infl_path) else None
        deriv_dataset = morph_evaluator.load_derivational(deriv_path) if os.path.exists(deriv_path) else None
        
        infl_words = [e['wordform'] for e in infl_dataset] if infl_dataset else []
        deriv_words = [e['wordform'] for e in deriv_dataset] if deriv_dataset else []
        
        words = set(infl_words + deriv_words)
        words = [w.strip() for w in words if w.strip()]
    except Exception as e:
        print(f"MorphyNet data error: {e}")
        return
        
    print(f"Found {len(words)} unique words to segment.")
    if not words: return

    val_dataset = TextDataset(words)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)
    
    checkpoints = get_checkpoints(args) 
    print(f"Found {len(checkpoints)} checkpoints to evaluate.")
    if not checkpoints: return

    scores_file = os.path.join(lang_output_dir, f"{args.lang_code}_scores.csv")
    all_scores = []
    if os.path.exists(scores_file):
        try: all_scores = pd.read_csv(scores_file).to_dict('records')
        except: pass

    for ckpt_name, ckpt_ref in checkpoints:
        infl_output_file = os.path.join(lang_output_dir, f"seg_infl_{args.lang_code}_{ckpt_name}.json")
        deriv_output_file = os.path.join(lang_output_dir, f"seg_deriv_{args.lang_code}_{ckpt_name}.json")
        
        already_done = False
        for score in all_scores:
            if score.get('checkpoint') == ckpt_name and 'layer' in score and not pd.isna(score['layer']):
                already_done = True
                break
                
        infl_done = os.path.exists(infl_output_file) if infl_words else True
        deriv_done = os.path.exists(deriv_output_file) if deriv_words else True
                
        if infl_done and deriv_done and already_done:
            print(f"Skipping {ckpt_name}, already exists in {scores_file} and locally.")
            continue

        print(f"Processing {ckpt_name}...")
        ckpt_path = None
        is_temp_download = False
        
        if args.local_dir: ckpt_path = ckpt_ref
        else:
            try:
                if not args.keep_downloaded:
                    local_dir = os.path.join(lang_output_dir, "temp_checkpoints")
                    os.makedirs(local_dir, exist_ok=True)
                    ckpt_path = hf_hub_download(repo_id=args.repo_id, filename=ckpt_ref, token=args.hf_token, local_dir=local_dir, local_dir_use_symlinks=False)
                    is_temp_download = True
                else: ckpt_path = hf_hub_download(repo_id=args.repo_id, filename=ckpt_ref, token=args.hf_token)
            except Exception as e:
                print(f"Failed to download {ckpt_ref}: {e}")
                continue
            
        model, config = load_model(ckpt_path, args.model_dim, args.model_arch, device)
        if model is None:
            if is_temp_download and os.path.exists(ckpt_path): os.remove(ckpt_path)
            continue
            
        try:
            segmentations = extract_segmentations(model, val_loader, device)
            word_to_seg = {w: seg for w, seg in zip(words, segmentations)}
            
            if infl_words:
                infl_save_data = {
                    "words": infl_words,
                    "segmentations": [word_to_seg.get(w, [[w]]) for w in infl_words]
                }
                with open(infl_output_file, 'w', encoding='utf-8') as f:
                    json.dump(infl_save_data, f, indent=2)
                if args.upload_repo_id: queue_for_upload(infl_output_file)
                
            if deriv_words:
                deriv_save_data = {
                    "words": deriv_words,
                    "segmentations": [word_to_seg.get(w, [[w]]) for w in deriv_words]
                }
                with open(deriv_output_file, 'w', encoding='utf-8') as f:
                    json.dump(deriv_save_data, f, indent=2)
                if args.upload_repo_id: queue_for_upload(deriv_output_file)

            num_layers = len(segmentations[0]) if segmentations else 0
            for layer_idx in range(num_layers):
                pred_dict = {}
                for w, recons in zip(words, segmentations):
                    idx = min(layer_idx, len(recons) - 1)
                    pred_dict[w] = recons[idx]
                    
                def dummy_segmenter(batch_words):
                    return [pred_dict.get(w, [w]) for w in batch_words]
                
                eval_results = morph_evaluator.evaluate_breakdown(
                    infl_entries=infl_dataset,
                    deriv_entries=deriv_dataset,
                    segment_fn=dummy_segmenter,
                    morph_type='all',
                    affix_type='all',
                    batch_size=args.batch_size
                )
                
                flat_metrics = {'checkpoint': ckpt_name, 'layer': layer_idx}
                for category, res in eval_results.items():
                    flat_metrics[f'{category}_precision'] = res['precision']
                    flat_metrics[f'{category}_recall'] = res['recall']
                    flat_metrics[f'{category}_f1'] = res['f1']
                    
                all_scores.append(flat_metrics)
                
                print(f"Evaluated {ckpt_name} Layer {layer_idx}")
                for category, res in eval_results.items():
                    print(f"  {category} - P: {res['precision']:.4f}, R: {res['recall']:.4f}, F1: {res['f1']:.4f}")
            
            pd.DataFrame(all_scores).to_csv(scores_file, index=False)

            if args.upload_repo_id:
                flush_uploads(repo_id=args.upload_repo_id, token=args.hf_token, subfolder=args.lang_code, delete_local=True, batch_size=3)

        except Exception as e:
             print(f"Error for {ckpt_name}: {e}")
             import traceback
             traceback.print_exc()
        finally:
            if is_temp_download and ckpt_path and os.path.exists(ckpt_path):
                try: os.remove(ckpt_path)
                except: pass
        
    if args.upload_repo_id:
        flush_uploads(repo_id=args.upload_repo_id, token=args.hf_token, subfolder=args.lang_code, delete_local=True, force=True)
        
    print("Done.")

if __name__ == "__main__":
    main()
