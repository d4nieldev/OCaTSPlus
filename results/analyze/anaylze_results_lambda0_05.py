import os
import json
import pandas as pd

# Base directory for results
base_dir = "results/Evaluate/Lambda0_05"

# Cache types to include
cache_types = ["LFU", "LRU", "Original"]

# Sizes to iterate through
sizes = list(range(100, 1001, 100))

# List to store extracted results
records = []

for cache_type in cache_types:

        
    for size in sizes:
        if cache_type == "Original":
            filename = f"summary_knn_simple_lambda0.05_size{size}.json"
        else:
            filename = f"summary_knn_{cache_type.lower()}_lambda0.05_size{size}.json"
        file_path = os.path.join(base_dir, cache_type, filename)

        if not os.path.exists(file_path):
            print(f"⚠️ File not found: {file_path}")
            continue

        with open(file_path, 'r') as f:
            data = json.load(f)

        record = {
            "cache_type": cache_type,
            "size": size,
            "final_accuracy": data.get("final_accuracy"),
            "final_discounted_accuracy": data.get("final_discounted_accuracy"),
            "total_calls": data.get("total_calls"),
            "cache_actual_size": data.get("cache_actual_size"),
            "normalized_accuracy": data.get("normalized_accuracy"),
            "hit_ratio": data.get("hit_ratio")
        }
        records.append(record)



# Export to CSV
df = pd.DataFrame(records)
os.makedirs("results/analyze", exist_ok=True)
output_path = "results/analyze/cache_eval_summary.csv"
df.to_csv(output_path, index=False)
print(f"✅ Exported summary to {output_path}")