import os
import json
import pandas as pd
import re

def extract_results_to_csv(base_folder, output_csv):
    results = []
    target_dirs = ["LFU", "LRU", "Original"]

    for cache_type in target_dirs:
        dir_path = os.path.join(base_folder, "Second", cache_type)
        if not os.path.exists(dir_path):
            continue
        for file in os.listdir(dir_path):
            if file.endswith(".json") and file.startswith("summary_"):
                full_path = os.path.join(dir_path, file)
                with open(full_path, "r") as f:
                    data = json.load(f)
                m = re.search(r'lambda([0-9]+(?:\.[0-9]+)?)', file)
                lambda_val = float(m.group(1)) if m else None
                results.append({
                    "file": file,
                    "cache_type": cache_type,
                    "model": "knn",  # hardcoded based on your file naming
                    "lambda": lambda_val,
                    "final_accuracy": data.get("final_accuracy"),
                    "final_discounted_accuracy": data.get("final_discounted_accuracy"),
                    "total_calls": data.get("total_calls"),
                    "cache_actual_size": data.get("cache_actual_size"),
                    "normalized_accuracy": data.get("normalized_accuracy"),
                    "hit_ratio": data.get("hit_ratio")
                })

    df = pd.DataFrame(results)
    df.sort_values(by=["cache_type", "lambda"], inplace=True)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved extracted results to {output_csv}")

# Usage
base_folder = "/home/danieloh/projects/OCaTSPlus/results/Evaluate"
output_csv = "results/analyze/normilaized_results.csv"
extract_results_to_csv(base_folder, output_csv)
