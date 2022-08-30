# %%

import os
import json
import pandas as pd
import torch

folder = "results_validation"
results = []
submissions = []

for run in sorted(os.listdir(folder), reverse=False):

    path = os.path.join(folder, run)
    print(run)

    if ("submission.txt" not in os.listdir(path)) or ("epoch_best.pth" not in os.listdir(path)):
        print(f"Bad run {run}")
        continue
    submission = pd.read_csv(os.path.join(path, 'submission.txt'), header=None)
    submission.columns = [run]
    submissions.append(submission)
    
    with open(os.path.join(path, "config.json"), "r") as fp:
        config = json.load(fp)
    config["run"] = run

    max_n_features = config["max_n_features"]
    n_classes = config["n_classes"]
    batch_size = config["batch_size"]
    test_size = config["test_size"]
    epochs = config["epochs"]
    lr = config["lr"]
    model = config["model"]
    cost_classification = config["cost_classification"]
    cost_align_ref = config["cost_align_ref"]
    cost_align_dat = config["cost_align_dat"]

    checkpoint = torch.load(os.path.join(path, "epoch_best.pth"))

    epoch = checkpoint["epoch"]
    model_state_dict = checkpoint["model_state_dict"]
    optimizer_state_dict = checkpoint["optimizer_state_dict"]
    train_loss = checkpoint["train_loss"]
    train_auc = checkpoint["train_auc"]
    train_acc = checkpoint["train_acc"]
    test_loss = checkpoint["test_loss"]
    test_auc = checkpoint["test_auc"]
    test_acc = checkpoint["test_acc"]

    results.append([
        run, max_n_features, n_classes, batch_size, test_size, epochs, lr, model, 
        cost_classification, cost_align_ref, cost_align_dat,
        epoch, train_loss, train_auc, train_acc, test_loss, test_auc, test_acc])


pd.concat(submissions, axis=1).to_csv("submissions.csv", index=False)

columns = ["run", "max_n_features", "n_classes", "batch_size", "test_size", "epochs", "lr", "model", 
        "cost_classification", "cost_align_ref", "cost_align_dat",
        "epoch", "train_loss", "train_auc", "train_acc", "test_loss", "test_auc", "test_acc"]
pd.DataFrame(results, columns=columns).to_csv("results.csv", index=False)

