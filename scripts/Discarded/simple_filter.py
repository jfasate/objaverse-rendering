# simple_filter.py
import json
import objaverse
import random
from tqdm import tqdm

def simple_quality_filter(sample_size=1000):
    """
    Simple filter based on file paths and basic heuristics
    This avoids the thumbnail dependency
    """
    print("Loading Objaverse UIDs...")
    uids = objaverse.load_uids()
    object_paths = objaverse._load_object_paths()
    
    print(f"Total UIDs available: {len(uids)}")
    
    # Take a random sample
    random.seed(42)
    sampled_uids = random.sample(uids, min(sample_size, len(uids)))
    
    filtered_uids = []
    exclusion_terms = [
        'test', 'temp', 'placeholder', 'broken', 'wip', 
        'untextured', 'lowpoly', 'simple', 'cube', 'sphere',
        'plane', 'default', 'example'
    ]
    
    print("Filtering based on file paths...")
    for uid in tqdm(sampled_uids):
        path = object_paths.get(uid, "")
        
        # Skip if no path
        if not path:
            continue
            
        # Skip non-GLB files
        if not path.endswith('.glb'):
            continue
            
        # Skip obviously bad objects based on path names
        path_lower = path.lower()
        if any(bad_term in path_lower for bad_term in exclusion_terms):
            continue
            
        # Skip very short paths (often indicate test objects)
        if len(path) < 10:
            continue
            
        filtered_uids.append(uid)
    
    print(f"Filtered: {len(filtered_uids)}/{len(sampled_uids)} objects passed")
    return filtered_uids

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_size', type=int, default=1000)
    parser.add_argument('--output_file', type=str, default='simple_filtered.json')
    args = parser.parse_args()
    
    filtered_uids = simple_quality_filter(args.sample_size)
    
    with open(args.output_file, 'w') as f:
        json.dump(filtered_uids, f, indent=2)
    
    print(f"Saved {len(filtered_uids)} filtered UIDs to {args.output_file}")

if __name__ == "__main__":
    main()