import os
import argparse
import json
import re
import csv
from collections import Counter
import matplotlib.pyplot as plt
from huggingface_hub import HfApi, hf_hub_download
import tqdm

def calculate_token_lengths(segmentations):
    """
    segmentations: list of sample segmentations.
    A sample segmentation is a list of layers, where each layer is a list of tokens string.
    The top-most layer is the target tokenizer output.
    """
    total_tokens = 0
    total_bytes = 0
    length_counts = Counter()
    
    for sample in segmentations:
        if sample and len(sample) > 0:
            top_layer = sample[-1]
            
            for token in top_layer:
                # Remove \xfe if it exists
                clean_token = token.replace("\\xfe", "")
                
                # Get length in bytes
                byte_len = len(clean_token.encode('utf-8'))
                
                total_tokens += 1
                total_bytes += byte_len
                length_counts[byte_len] += 1
                
    if total_tokens == 0:
        return 0.0, {}
        
    avg_length = total_bytes / total_tokens
    return avg_length, dict(length_counts)

def plot_distribution(length_counts, ckpt_name, lang_code, output_path):
    lengths = sorted(length_counts.keys())
    counts = [length_counts[l] for l in lengths]
    
    plt.figure(figsize=(10, 6))
    plt.bar(lengths, counts, color='skyblue', edgecolor='black')
    plt.xlabel('Token Length (in bytes)')
    plt.ylabel('Count')
    plt.title(f'Token Length Distribution - {lang_code} (Checkpoint: {ckpt_name})')
    plt.xticks(lengths)
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, required=True, help="HF Repo ID for segmentations")
    parser.add_argument("--lang_code", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True, help="CSV file to save average results")
    parser.add_argument("--dist_dir", type=str, required=True, help="Directory to save distribution CSVs and plots")
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

    # Check existing average results
    existing_results = {}
    if os.path.exists(args.output_file):
        with open(args.output_file, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_results[row['checkpoint_name']] = float(row['Average_Token_Length'])
                
    results = []
    
    temp_dir = "temp_token_length_downloads"
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(args.dist_dir, exist_ok=True)
    
    for file_path in tqdm.tqdm(candidates, desc=f"Evaluating {args.lang_code}"):
        file_name = os.path.basename(file_path)
        
        prefix = f"seg_{args.lang_code}_"
        ckpt_name = file_name[len(prefix):]
        if ckpt_name.endswith('.json'):
            ckpt_name = ckpt_name[:-5]
            
        # Dist files paths
        dist_csv_path = os.path.join(args.dist_dir, f"{ckpt_name}_dist.csv")
        dist_plot_path = os.path.join(args.dist_dir, f"{ckpt_name}_plot.png")
        
        # Determine if we already ran for this checkpoint fully
        if ckpt_name in existing_results and os.path.exists(dist_csv_path) and os.path.exists(dist_plot_path):
            results.append({"checkpoint_name": ckpt_name, "Average_Token_Length": existing_results[ckpt_name]})
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
                
            avg_length, length_counts = calculate_token_lengths(segmentations)
            results.append({"checkpoint_name": ckpt_name, "Average_Token_Length": avg_length})
            
            # Save distribution CSV
            with open(dist_csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["Token_Length_Bytes", "Count"])
                for length, count in sorted(length_counts.items()):
                    writer.writerow([length, count])
            
            # Save distribution plot
            plot_distribution(length_counts, ckpt_name, args.lang_code, dist_plot_path)
            
            # Immediately delete to save memory
            os.remove(local_path)
            
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            
    # Write to average CSV
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["checkpoint_name", "Average_Token_Length"])
        writer.writeheader()
        for r in results:
            writer.writerow(r)
            
    # Clean up temp dir if empty
    try:
        os.rmdir(os.path.join(temp_dir, subfolder))
        os.rmdir(temp_dir)
    except:
        pass
        
    print(f"Results saved to {args.output_file} and {args.dist_dir}")

if __name__ == "__main__":
    main()
