# %%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import argparse
import torch


# %% CONFIGURATION

from src.utils import Parameters, setup

if __name__ == "__main__":
    pass
    # parser = argparse.ArgumentParser(description='GKN Grading Predictor')

    # parser.add_argument('--folder', type=str, help='results folder', default="results")
    # parser.add_argument('--batch-size', type=int, help='batch size', default=32)

    # args = parser.parse_args()

args = Parameters(
    {
        "folder": "results",
        "batch_size": 10,
        "epochs": 1,
        "lr": 0.003,
    }
)

path = setup(args)

# %% DATASET

from src.dataset import get_loader

max_n_features = 748
resample = True
batch_size = 10

train_loader_params = {
    "folder": "training_data_reduced",
    "max_n_features": max_n_features,
    "resample": resample,
    "test_size": 0.3,
    "shuffle_split": False,
    "shuffle_sample": False,
    "batch_size": batch_size,
}
validation_loader_params = {
    "folder": "testing_data",
    "max_n_features": max_n_features,
    "resample": resample,
    "test_size": None,
    "shuffle_split": False,
    "shuffle_sample": False,
    "batch_size": batch_size,
}

train_loader, test_loader = get_loader(**train_loader_params)
validation_loader = get_loader(**validation_loader_params)
# train_loader.dataset.plot()


# %% MODEL PARAMETERS

cnn_params = {
    "n_channels": 3,
    "n_features": max_n_features,
    "dropout": None,
}
fcn_params = {
    "hidden_size": 10,
    "hidden_layers": 2,
    "dropout": None,
    "output_size": 11,  # n_classes
}

n_classes = 11
epochs = 2
lr = 0.001

from src.models import CustomModel, BaselineModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = CustomModel(cnn_params, fcn_params).to(device)
# model = BaselineModel().to(device)
loss_func = torch.nn.CrossEntropyLoss(reduction="mean")
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# %% TRAINING
from src.base import train_and_evaluate, plot_results

model, train_results, test_results = train_and_evaluate(
    model, epochs, optimizer, loss_func, train_loader, test_loader, device, path
)

plot_results(train_results, test_results, path)


# %% LOAD BEST
from src.base import load_best_model

model = load_best_model(model, path)

# %% TEST RESULTS
# from src.base import evaluate, predict, plot_confusion_matrix

test_loss, test_auc, test_acc = evaluate(model, test_loader, loss_func, device)
print(f"Loss: {test_loss:.4f}; AUC: {test_acc:.4f}; Accuracy: {test_acc:.4f}")

y_true, y_pred = predict(model, test_loader, device)
plot_confusion_matrix(y_true, y_pred, path)

# %% SUBMISSION FILE

from src.base import save_submission

save_submission(model, validation_loader, device)
