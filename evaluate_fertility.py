import os
import argparse
import json
import re
import csv
from huggingface_hub import HfApi, hf_hub_download
import tqdm

def calculate_fertility_for_segmentations(segmentations):
    """
    segmentations: list of sample segmentations.
    A sample segmentation is a list of layers, where each layer is a list of tokens string.
    The top-most layer is the target tokenizer output.
    Fertility is defined as the average number of tokens per word.
    A "word" is defined by space-delimited text.
    However, H-Net tokens can span multiple words (contain spaces within the token).
    So we should count the number of words in the original text,
    and then compute the ratio: (total tokens) / (total words).
    
    The original text can be reconstructed by joining the tokens.
    """
    total_tokens = 0
    total_words = 0
    
    for sample in segmentations:
        if sample and len(sample) > 0:
            top_layer = sample[-1]
            total_tokens += len(top_layer)
            
            # Reconstruct original text to count words
            # The tokens might contain \xfe for BOS, we can just strip it or let split() handle it
            # actually \xfe is a single character or string "\\xfe".
            text = "".join(top_layer)
            # Remove \xfe if it exists
            text = text.replace("\\xfe", "")
            
            # Count words by space splitting
            words = text.split()
            total_words += len(words)
            
    if total_words == 0:
        return 0.0
        
    return total_tokens / total_words

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
                existing_results[row['checkpoint_name']] = float(row['Fertility'])
                
    results = []
    
    temp_dir = "temp_fertility_downloads"
    os.makedirs(temp_dir, exist_ok=True)
    
    for file_path in tqdm.tqdm(candidates, desc=f"Evaluating {args.lang_code}"):
        file_name = os.path.basename(file_path)
        
        prefix = f"seg_{args.lang_code}_"
        ckpt_name = file_name[len(prefix):]
        if ckpt_name.endswith('.json'):
            ckpt_name = ckpt_name[:-5]
            
        if ckpt_name in existing_results:
            results.append({"checkpoint_name": ckpt_name, "Fertility": existing_results[ckpt_name]})
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
                
            fertility = calculate_fertility_for_segmentations(segmentations)
            results.append({"checkpoint_name": ckpt_name, "Fertility": fertility})
            
            # Immediately delete to save memory
            os.remove(local_path)
            
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            
    # Write to CSV
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["checkpoint_name", "Fertility"])
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
