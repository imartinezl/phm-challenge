# %%

import os
import json
import pandas as pd
import torch

folder = "results"
results = []

for run in sorted(os.listdir(folder), reverse=True):

    path = os.path.join(folder, run)
    print(run)

    if "epoch_best.pth" not in os.listdir(path):
        print(f"Bad run {run}")
        continue

    with open(os.path.join(path, "config.json"), "r") as fp:
        config = json.load(fp)
    config["run"] = run
    

    checkpoint = torch.load(os.path.join(path, "epoch_best.pth"))

    epoch = checkpoint["epoch"]
    # model_state_dict = checkpoint["model_state_dict"]
    # optimizer_state_dict = checkpoint["optimizer_state_dict"]

    results.append([])

columns = [
    "run", 
]
pd.DataFrame(results, columns=columns).to_csv("results.csv", index=False)
