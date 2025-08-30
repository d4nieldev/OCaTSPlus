import os
import json
import csv

# Configuration
MODELS = ["knn", "mpnet"]
CACHE_TYPES = ["lfu", "lru", "simple"]
LAMBDAS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
SUMMARY_FOLDER = "results/Evaluate/First"
OUTPUT_CSV = "results/analyze/all_model_summaries2.csv"

# Fields to export
fieldnames = [
    "model",
    "cache_type",
    "lambda",
    "final_accuracy",
    "final_discounted_accuracy",
    "normalized_accuracy",
    "cache_actual_size",
    "total_calls"
]

rows = []

for model in MODELS:
    for cache in CACHE_TYPES:
        for lam in LAMBDAS:
            json_path = os.path.join(SUMMARY_FOLDER, f"summary_{model}_{cache}_lambda{lam}.json")
            if os.path.exists(json_path):
                with open(json_path, "r") as f:
                    summary = json.load(f)
                    rows.append({
                        "model": model,
                        "cache_type": cache,
                        "lambda": lam,
                        "final_accuracy": summary.get("final_accuracy"),
                        "final_discounted_accuracy": summary.get("final_discounted_accuracy"),
                        "normalized_accuracy": summary.get("normalized_accuracy"),
                        "cache_actual_size": summary.get("cache_actual_size"),
                        "total_calls": summary.get("total_calls"),
                    })
            else:
                print(f"⚠️ Missing file: {json_path}")

# Write CSV
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"✅ Exported {len(rows)} entries to: {OUTPUT_CSV}")
