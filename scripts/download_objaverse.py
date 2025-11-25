# scripts/download_filtered.py
import json
import os
import objaverse
from tqdm import tqdm
import argparse

def create_download_list(filtered_uids_file, output_file, batch_size=1000):
    """
    Create download lists from filtered UIDs for distributed rendering.
    
    Args:
        filtered_uids_file: JSON file with filtered UIDs from filter_objaverse.py
        output_file: Base name for output batch files
        batch_size: Number of objects per batch
    """
    
    # Load filtered UIDs
    with open(filtered_uids_file, 'r') as f:
        filtered_uids = json.load(f)
    
    print(f"Loaded {len(filtered_uids)} filtered UIDs")
    
    # Load object paths from Objaverse
    object_paths = objaverse._load_object_paths()
    
    # Convert to download URLs
    download_urls = []
    missing_count = 0
    
    for uid in tqdm(filtered_uids, desc="Creating download URLs"):
        if uid in object_paths:
            url = f"https://huggingface.co/datasets/allenai/objaverse/resolve/main/{object_paths[uid]}"
            download_urls.append(url)
        else:
            missing_count += 1
    
    print(f"Successfully mapped {len(download_urls)} UIDs to URLs")
    print(f"Missing objects: {missing_count}")
    
    # Create batches for distributed rendering
    os.makedirs("download_batches", exist_ok=True)
    
    for i in range(0, len(download_urls), batch_size):
        batch = download_urls[i:i + batch_size]
        batch_file = f"download_batches/{output_file}_batch_{i//batch_size}.json"
        
        with open(batch_file, 'w') as f:
            json.dump(batch, f, indent=2)
        
        print(f"Created batch {i//batch_size} with {len(batch)} objects")
    
    # Also create a single file with all URLs
    with open(f"download_batches/{output_file}_all.json", 'w') as f:
        json.dump(download_urls, f, indent=2)
    
    print(f"\nCreated {len(download_urls)//batch_size + 1} batch files in 'download_batches/'")
    return download_urls

def check_existing_renders(output_dir="./views"):
    """
    Check which objects have already been rendered to avoid re-rendering.
    """
    completed_uids = set()
    
    if os.path.exists(output_dir):
        for item in os.listdir(output_dir):
            item_path = os.path.join(output_dir, item)
            if os.path.isdir(item_path):
                # Check if this object has all 16 views + cameras.json
                png_files = [f for f in os.listdir(item_path) if f.endswith('.png')]
                if len(png_files) >= 16 and os.path.exists(os.path.join(item_path, 'cameras.json')):
                    completed_uids.add(item)
    
    print(f"Found {len(completed_uids)} already rendered objects")
    return completed_uids

def main():
    parser = argparse.ArgumentParser(description='Create download lists for filtered Objaverse objects')
    parser.add_argument('--filtered_file', type=str, required=True,
                       help='JSON file with filtered UIDs from filter_objaverse.py')
    parser.add_argument('--output_name', type=str, default='mvd_fusion',
                       help='Base name for output batch files')
    parser.add_argument('--batch_size', type=int, default=1000,
                       help='Number of objects per batch')
    parser.add_argument('--skip_completed', action='store_true',
                       help='Skip objects that are already rendered')
    
    args = parser.parse_args()
    
    # Skip already rendered objects if requested
    if args.skip_completed:
        completed_uids = check_existing_renders()
        # Load filtered UIDs and remove completed ones
        with open(args.filtered_file, 'r') as f:
            filtered_uids = json.load(f)
        
        filtered_uids = [uid for uid in filtered_uids if uid not in completed_uids]
        
        # Save updated list
        updated_file = args.filtered_file.replace('.json', '_remaining.json')
        with open(updated_file, 'w') as f:
            json.dump(filtered_uids, f, indent=2)
        
        print(f"Filtered to {len(filtered_uids)} remaining objects (saved to {updated_file})")
        args.filtered_file = updated_file
    
    # Create download lists
    create_download_list(args.filtered_file, args.output_name, args.batch_size)

if __name__ == "__main__":
    main()