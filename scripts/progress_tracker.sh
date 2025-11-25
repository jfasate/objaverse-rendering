#!/bin/bash
# progress_tracker_15k.sh - Updated for 15K objects
TOTAL_BATCHES=30  # 15000 objects / 500 per batch = 30 batches
BATCH_SIZE=500
START_TIME=$(date +%s)

echo "🚀 Starting MVD-Fusion 15K Dataset Generation"
echo "=============================================="

for i in {0..29}; do  # 0 to 29 = 30 batches
    BATCH_FILE="download_batches/mvd_50k_batch_${i}.json"
    
    # Check if batch file exists
    if [ ! -f "$BATCH_FILE" ]; then
        echo "❌ ERROR: Batch file not found: $BATCH_FILE"
        echo "Please create batch files first with:"
        echo "  python3 download_objaverse.py --filtered_file ../filtered_15k.json --output_name mvd_15k --batch_size 500"
        exit 1
    fi
    
    echo ""
    echo "=== Batch $((i+1))/$TOTAL_BATCHES ==="
    echo "Start time: $(date)"
    echo "Batch file: $(basename $BATCH_FILE)"
    
    # Show GPU status
    echo "GPU Status:"
    nvidia-smi --query-gpu=utilization.gpu,temperature.gpu,memory.used --format=csv
    
    # Render this batch
    echo "Starting rendering..."
    python3 distributed.py \
        --workers_per_gpu 3 \
        --input_models_path $BATCH_FILE \
        --num_gpus 1
    
    # Calculate progress - UPDATED for 15K objects
    CURRENT_OBJECTS=$(( (i+1) * BATCH_SIZE ))
    ELAPSED=$(( $(date +%s) - START_TIME ))
    ETA=$(( (ELAPSED * TOTAL_BATCHES / (i+1)) - ELAPSED ))
    
    echo "Progress: $CURRENT_OBJECTS/15000 objects"
    echo "Elapsed: $((ELAPSED/3600))h $(( (ELAPSED%3600)/60 ))m"
    echo "ETA: $((ETA/3600))h $(( (ETA%3600)/60 ))m"
    echo "----------------------------------------"
done

echo "🎉 All 15K objects completed!"
