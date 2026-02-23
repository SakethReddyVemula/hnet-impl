import os
import argparse
import json
import re
import csv
from huggingface_hub import HfApi, hf_hub_download
import tqdm

def count_tokens_in_segmentation(segmentations):
    """
    segmentations: list of sample segmentations.
    A sample segmentation is a list of layers, where each layer is a list of tokens string.
    The top-most layer is the target tokenizer output. 
    Here, layers[-1] should be the top-most level tokens (e.g. byte -> subword -> word). 
    Wait, in `evaluate_segmentation.py`, the reconstructions list matches `layers_indices`.
    If layers = [layer0, layer1, ...], layer[-1] is typically the highest level.
    Actually, let's just use the last layer.
    """
    total_tokens = 0
    for sample in segmentations:
        if sample and len(sample) > 0:
            top_layer = sample[-1]
            total_tokens += len(top_layer)
    return total_tokens

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, required=True, help="HF Repo ID for segmentations")
    parser.add_argument("--lang_code", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True, help="CSV file to save results")
    parser.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN"))
    
    args = parser.parse_args()
    
    api = HfApi(token=args.hf_token)
    
    print(f"Listing segmentations for {args.lang_code} from {args.repo_id}...")
    try:
        files = api.list_repo_files(repo_id=args.repo_id, repo_type="dataset")
    except Exception as e:
        print(f"Error accessing repo: {e}")
        return

    subfolder = args.lang_code
    candidates = [f for f in files if f.startswith(f"{subfolder}/seg_{args.lang_code}_")]
    
    # Sort by checkpoint
    def sort_key(f):
        base = os.path.basename(f)
        # Assuming format seg_{lang}_checkpoint_{epoch}_{step}.pt.json or similar
        # Actual format from evaluate_segmentation.py: seg_{lang}_{ckpt_name}.json
        # ckpt_name could be checkpoint_{epoch}_{step}.pt
        m = re.search(r'checkpoint_(\d+)_(\d+)', base)
        if m: return (int(m.group(1)), int(m.group(2)))
        m = re.search(r'checkpoint(\d+)', base)
        if m: return (int(m.group(1)), 999999)
        return (0, 0)
        
    candidates.sort(key=sort_key)
    print(f"Found {len(candidates)} segmentations.")
    
    if not candidates:
        return

    # Check existing results
    existing_results = {}
    if os.path.exists(args.output_file):
        with open(args.output_file, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_results[row['checkpoint_name']] = int(row['CTC'])
                
    results = []
    
    temp_dir = "temp_ctc_downloads"
    os.makedirs(temp_dir, exist_ok=True)
    
    for file_path in tqdm.tqdm(candidates, desc=f"Evaluating {args.lang_code}"):
        file_name = os.path.basename(file_path)
        
        # Extract original checkpoint name
        # format: seg_{lang}_{ckpt_name}.json
        prefix = f"seg_{args.lang_code}_"
        ckpt_name = file_name[len(prefix):]
        if ckpt_name.endswith('.json'):
            ckpt_name = ckpt_name[:-5]
            
        if ckpt_name in existing_results:
            results.append({"checkpoint_name": ckpt_name, "CTC": existing_results[ckpt_name]})
            continue
            
        # Download
        try:
            local_path = hf_hub_download(
                repo_id=args.repo_id,
                filename=file_path,
                repo_type="dataset",
                token=args.hf_token,
                local_dir=temp_dir,
                local_dir_use_symlinks=False
            )
            
            with open(local_path, 'r', encoding='utf-8') as f:
                segmentations = json.load(f)
                
            ctc = count_tokens_in_segmentation(segmentations)
            results.append({"checkpoint_name": ckpt_name, "CTC": ctc})
            
            # Immediately delete to save memory
            os.remove(local_path)
            
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            
    # Write to CSV
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["checkpoint_name", "CTC"])
        writer.writeheader()
        for r in results:
            writer.writerow(r)
            
    # Clean up temp dir if empty
    try:
        os.rmdir(os.path.join(temp_dir, subfolder))
        os.rmdir(temp_dir)
    except:
        pass
        
    print(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    main()
