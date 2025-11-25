# create_subset_lists.py
import json
import os
import random

# Path to your rendered data
RENDERS_DIR = "/home/jayesh/objaverse-rendering/scripts/objaverse/renders/15k"
OUTPUT_DIR = "/home/jayesh/objaverse-rendering/scripts/objaverse/renders/subset_list"

# Get all UIDs
all_uids = [d for d in os.listdir(RENDERS_DIR) 
            if os.path.isdir(os.path.join(RENDERS_DIR, d))]

print(f"Found {len(all_uids)} rendered objects")

# Train/test split (90/10)
random.seed(42)
random.shuffle(all_uids)
split_idx = int(len(all_uids) * 0.9)

train_uids = all_uids[:split_idx]
test_uids = all_uids[split_idx:]

print(f"Train: {len(train_uids)}, Test: {len(test_uids)}")

# Save subset lists
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(f"{OUTPUT_DIR}/15k_train.json", 'w') as f:
    json.dump(train_uids, f, indent=2)

with open(f"{OUTPUT_DIR}/15k_test.json", 'w') as f:
    json.dump(test_uids, f, indent=2)

print("✅ Created subset list files")