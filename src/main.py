# External imports
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from evaluate import load
import argparse
import json
import os

# Local imports
from src.utils.seeding import set_seed
from src.models.knn import KNNClassifier
from src.models.mpnet import MPNetClassifier
from src.caches.base import CACHE_REGISTRY

# TPU
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    TPU_FLAG = True
except ImportError:
    TPU_FLAG = False

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default="src/utils/config.json")
parser.add_argument('-m', '--model', type=str, default="knn")
parser.add_argument('-s', '--seed', type=int, default=54321)
parser.add_argument('-t', '--train_path', type=str, default="data/processed/banking77/best3_train.csv")
parser.add_argument('-d', '--test_path', type=str, default="data/raw/banking77/test.csv")
args = parser.parse_args()

# Setup
CONFIG_PATH = args.config
MODEL_NAME = args.model
SEED = args.seed
TRAIN_PATH = args.train_path
TEST_PATH = args.test_path
set_seed(SEED)

device = xm.xla_device() if TPU_FLAG else torch.device("cuda" if torch.cuda.is_available() else "mps")
encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").to(device)
encoder.eval()

# Data
train_data = pd.read_csv(TRAIN_PATH)
test_data = pd.read_csv(TEST_PATH)
with torch.no_grad():
    train_embeddings = encoder.encode(train_data["text"].tolist(), show_progress_bar=True, convert_to_tensor=True, device=device)
    test_embeddings = encoder.encode(test_data["text"].tolist(), show_progress_bar=True, convert_to_tensor=True, device=device)
train_labels = train_data["label"].tolist()
test_labels = test_data["label"].tolist()
test_gpt_labels = test_data["gpt-label"].tolist()
test_set = TensorDataset(test_embeddings, torch.tensor(test_labels), torch.tensor(test_gpt_labels))
eval_fn = load("accuracy")

# Core eval function
def run_eval(lamda, config, model_name, cache_type, cache_capacity, result_dir):
    RETRAIN_NUM = 100
    num_runs = 5
    e_thresh = config[model_name]["e_thresh"][str(lamda)]
    d_thresh = config[model_name]["d_thresh"][str(lamda)]

    avg_accs, avg_disc_accs, avg_calls, avg_hits, avg_total = [], [], [], [], []

    for _ in range(num_runs):
        accs, disc_accs, calls_, hits, total = [], [], [], [], []
        cache = CACHE_REGISTRY[cache_type](train_embeddings, train_labels, d_thresh, capacity=cache_capacity)

        if model_name == "knn":
            model = KNNClassifier(cache=cache)
        elif model_name == "mpnet":
            mpnet_cfg = config[model_name]
            model = MPNetClassifier(mpnet_cfg["hidden_size"], mpnet_cfg["dropout"], mpnet_cfg["activation"]).to(device)
            model.load_state_dict(torch.load(mpnet_cfg["checkpoint"]))
            model.eval()

        model.train() if model_name == "mpnet" else None
        online_stream = DataLoader(test_set, batch_size=1, shuffle=True)
        predictions, labels = [], []
        calls = 0
        cache_hits = 0

        for v, gt, l in tqdm(online_stream, desc=f"Lambda {lamda}"):
            v, l, gt = v.to(device), l.to(device), gt.to(device)
            cls_probs = model(v)
            if model_name == "mpnet":
                cls_probs = torch.softmax(cls_probs, dim=1)
            pred = torch.argmax(cls_probs)

            entropy = -torch.sum(cls_probs * torch.log(cls_probs))
            query = v.squeeze(0) if v.dim() == 2 and v.size(0) == 1 else v

            is_hit = torch.le(entropy, e_thresh) and cache.is_near(query)
            if not is_hit:
                cache.add(query, l)
                pred = l
                calls += 1
            else:
                cache_hits += 1

            predictions.append(pred)
            labels.append(gt)

            if model_name == "mpnet" and calls % RETRAIN_NUM == 0 and calls != 0:
                loader = DataLoader(cache.get_last_p_added(p=100), batch_size=32, shuffle=True)
                train(model, loader, config)

            accuracy = eval_fn.compute(predictions=predictions, references=labels)["accuracy"]
            disc_acc = accuracy - lamda * calls / len(predictions)
            accs.append(accuracy)
            disc_accs.append(disc_acc)
            calls_.append(calls)
            hits.append(cache_hits)
            total.append(len(predictions))


        avg_accs.append(accs)
        avg_disc_accs.append(disc_accs)
        avg_calls.append(calls_)
        avg_hits.append(hits)
        avg_total.append(total)


    avg_accs = np.mean(avg_accs, axis=0)
    avg_disc_accs = np.mean(avg_disc_accs, axis=0)
    avg_calls = np.mean(avg_calls, axis=0)
    final_hits = np.mean(avg_hits, axis=0)[-1]
    final_total = np.mean(avg_total, axis=0)[-1]
    hit_ratio = np.mean(avg_hits[-1]) / np.mean(avg_total[-1])

    cache_actual_size = len(cache)
    normalized_accuracy = float(avg_accs[-1]) / cache_actual_size if cache_actual_size > 0 else 0.0

    os.makedirs(result_dir, exist_ok=True)
    np.savez(os.path.join(result_dir, f"results_{model_name}_{cache_type}_lambda{lamda}_size{cache_capacity}.npz"),
             accs=avg_accs, disc_accs=avg_disc_accs, calls=avg_calls)

    json_summary = {
        "final_accuracy": float(avg_accs[-1]),
        "final_discounted_accuracy": float(avg_disc_accs[-1]),
        "total_calls": int(avg_calls[-1]),
        "cache_actual_size": cache_actual_size,
        "normalized_accuracy": normalized_accuracy,
        "hit_ratio": hit_ratio
    }
    with open(os.path.join(result_dir, f"summary_{model_name}_{cache_type}_lambda{lamda}_size{cache_capacity}.json"), "w") as f:
        json.dump(json_summary, f, indent=2)

# Main bulk eval loop
if __name__ == "__main__":
    config = json.load(open(CONFIG_PATH, "r"))
    LAMBDA = 0.05
    CACHE_TYPES = ["lru"]
    CACHE_LABELS = {"lfu": "LFU", "lru": "LRU", "simple": "Original"}

    for cache_type in CACHE_TYPES:
        if cache_type == "lru":
            for size in range(100, 1001, 100):
                result_dir = os.path.join("results", "Evaluate", "Lambda0_05", CACHE_LABELS[cache_type])
                run_eval(LAMBDA, config, MODEL_NAME, cache_type, size, result_dir)
            
            result_dir = os.path.join("results", "Evaluate", "Lambda0_05", CACHE_LABELS[cache_type])   
            run_eval(LAMBDA, config, MODEL_NAME, cache_type, 1700, result_dir)
        
        elif cache_type == "lfu":
            result_dir = os.path.join("results", "Evaluate", "Lambda0_05", CACHE_LABELS[cache_type])
            run_eval(LAMBDA, config, MODEL_NAME, cache_type, 1700, result_dir)

        elif cache_type == "simple":
            for size in range(800, 1001, 100):
                result_dir = os.path.join("results", "Evaluate", "Lambda0_05", CACHE_LABELS[cache_type])
                run_eval(LAMBDA, config, MODEL_NAME, cache_type, size, result_dir)
