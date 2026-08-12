# Class to handle the full lifecycle of an experiment with logging
import os
import json
import numpy as np
from datetime import datetime
import cupy as cp

class ExperimentManager:
    def __init__(self, name, base_dir="runs", overwrite=False):
        self.base_name = name
        self.overwrite = overwrite

        if overwrite:
            self.dir = os.path.join(base_dir, name)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.dir = os.path.join(base_dir, f"{name}_{timestamp}")

        os.makedirs(self.dir, exist_ok=True)

        self.log_path = os.path.join(self.dir, "experimental_log.txt")
        self.metrics_path = os.path.join(self.dir, "metrics.json")
        self.summary_path = os.path.join(base_dir, "summary.log")
        self.index_file = os.path.join(base_dir, "best_models.json")
        self.metrics = []

        if overwrite:
            # Clear the experiment log when overwriting
            if os.path.exists(self.log_path):
                with open(self.log_path, "w", encoding="utf-8") as f:
                    f.truncate(0)


    def log(self, msg):
        # print(msg)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    def prepend(self, header_msg):
        if os.path.exists(self.log_path):
            with open(self.log_path, "r", encoding="utf-8") as f:
                existing = f.read()
        else:
            existing = ""
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write(header_msg + "\n" + existing)

    def write_summary(self, msg):
        with open(self.summary_path, "a", encoding="utf-8") as f:
            f.write(msg)

    def save_metrics(self, epoch, loss, accuracy):
        entry = {"epoch": epoch, "loss": loss, "accuracy": accuracy}
        self.metrics.append(entry)
        with open(self.metrics_path, "w") as f:
            json.dump(self.metrics, f, indent=2)

    def save_model(self, network, filename="network_params.npz"):
        path = os.path.join(self.dir, filename)
        params = {f"param_{i}": cp.asnumpy(p) for i, p in enumerate(network.params())}
        np.savez(path, **params)
        self.log(f"Saved model to {path}")

    def update_global_best_model(self, timestamp, dataset_name, config, val_acc, model, epoch, filename="best_model.npz"):
        entry = {
            "timestamp": timestamp,
            "experiment": os.path.basename(self.dir),
            "config_file": config,
            "model": model,
            "val_acc": val_acc,
            "epoch": epoch,
            "model_path": os.path.join(self.dir, filename),
        }

        if os.path.exists(self.index_file):
                with open(self.index_file, "r") as f:
                    best_models = json.load(f)
        else:
            best_models = {}

        best = best_models.get(dataset_name, {}).get(config, {}).get("val_acc", -1)

        if val_acc > best:
            best_models.setdefault(dataset_name, {})[config] = entry
            with open(self.index_file, "w") as f:
                json.dump(best_models, f, indent=2)
            self.log(f"🎉 New best model for {dataset_name}/{config}: {val_acc:.2f}%")
        else:
            self.log(f"(Not best for {dataset_name}/{config}: {val_acc:.2f}%)")

