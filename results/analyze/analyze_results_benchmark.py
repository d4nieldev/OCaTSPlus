import json
import csv

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def compare_caches_export_csv(file1, file2, file3, output_csv):
    data_list = [load_json(file) for file in (file1, file2, file3)]

    # Weights for the combined efficiency score (can be adjusted)
    weight_hit = 0.4
    weight_thr = 0.3
    weight_lat = 0.2
    weight_llm = 0.1

    rows = [["cache_type", "cache_hit_rate", "mean_latency_sec", "throughput_qps", "total_llm_calls", "efficiency_score"]]

    for data in data_list:
        ch = data.get("cache_hit_rate", 0)
        lat = data.get("mean_latency_sec", 1)
        thr = data.get("throughput_qps", 0)
        llm = data.get("total_llm_calls", 0)

        score = (ch * weight_hit) + (thr * weight_thr / 1000) + ((1 / lat) * weight_lat * 0.001) + ((1 / (llm + 1)) * weight_llm)

        rows.append([data["cache_type"], ch, lat, thr, llm, score])

    # Write results to CSV file
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(rows)

    print(f"Results exported to {output_csv}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 5:
        print("Usage: python compare_caches_export.py <file1.json> <file2.json> <file3.json> <output.csv>")
        sys.exit(1)

    compare_caches_export_csv(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
