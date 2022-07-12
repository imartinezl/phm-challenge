# %%

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch

# %% PARAMETERS

fn_torchsave = "epoch_best.pth"
fn_submission = "submission.txt"
fn_training = "training.pdf"
fn_confusion = "confusion_matrix.pdf"

# %%
def train(model, train_loader, optimizer, loss_func, device):
    model.train()

    for x_dat, x_ref, y_dat, y_ref, individual in train_loader:
        optimizer.zero_grad(set_to_none=True)

        x_dat = x_dat.to(device)
        x_ref = x_ref.to(device)
        y_dat = y_dat.to(device)
        y_ref = y_ref.to(device)

        z_dat, x_dat_aligned, x_ref_aligned = model(x_dat, x_ref)
        z_dat = torch.softmax(z_dat, -1)

        loss = loss_func(z_dat, y_dat, x_dat_aligned, x_ref_aligned, individual)
        loss.backward()
        optimizer.step()

    return model


from sklearn.metrics import roc_auc_score, accuracy_score


@torch.no_grad()
def evaluate(model, data_loader, loss_func, device, log=False):
    model.eval()
    loss = 0

    y_true, y_pred = [], []
    for x_dat, x_ref, y_dat, y_ref, individual in data_loader:

        x_dat = x_dat.to(device)
        x_ref = x_ref.to(device)
        y_dat = y_dat.to(device)
        y_ref = y_ref.to(device)

        z_dat, x_dat_aligned, x_ref_aligned = model(x_dat, x_ref)
        z_dat = torch.softmax(z_dat, -1)

        loss += loss_func(z_dat, y_dat, x_dat_aligned, x_ref_aligned, individual).item()

        y_true.append(y_dat.cpu().numpy())
        y_pred.append(z_dat.cpu().numpy())

    y_true, y_pred = np.concatenate(y_true), np.concatenate(y_pred)

    loss /= len(data_loader)
    auc = roc_auc_score(
        y_true.argmax(axis=-1), y_pred, multi_class="ovo", labels=range(11)
    )
    acc = accuracy_score(y_true.argmax(-1), y_pred.argmax(-1))

    if log:
        print(f"Loss: {loss:.4f}; AUC: {auc:.4f}; Accuracy: {acc:.4f}")
    return loss, auc, acc


@torch.no_grad()
def predict(model, data_loader, device):
    model.eval()

    y_true, y_pred = [], []
    for x_dat, x_ref, y_dat, y_ref, individual in data_loader:

        x_dat = x_dat.to(device)
        x_ref = x_ref.to(device)
        y_dat = y_dat.to(device)
        y_ref = y_ref.to(device)

        z_dat, x_dat_aligned, x_ref_aligned = model(x_dat, x_ref)

        y_true.append(y_dat.argmax(dim=-1).cpu().numpy() + 1)
        y_pred.append(z_dat.argmax(dim=-1).cpu().numpy() + 1)

    y_true, y_pred = np.concatenate(y_true), np.concatenate(y_pred)

    return y_true, y_pred


def train_and_evaluate(
    model, epochs, optimizer, loss_func, train_loader, test_loader, device, path="."
):
    train_results, test_results = [], []
    min_test_loss = np.inf
    with tqdm(total=epochs) as pbar:
        for epoch in range(epochs):
            model = train(model, train_loader, optimizer, loss_func, device)
            train_loss, train_auc, train_acc = evaluate(
                model, train_loader, loss_func, device
            )
            test_loss, test_auc, test_acc = evaluate(
                model, test_loader, loss_func, device
            )

            train_results.append([train_loss, train_auc, train_acc])
            test_results.append([test_loss, test_auc, test_acc])

            if test_loss <= min_test_loss:
                min_test_loss = test_loss
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "train_loss": train_loss,
                        "train_auc": train_auc,
                        "train_acc": train_acc,
                        "test_loss": test_loss,
                        "test_auc": test_auc,
                        "test_acc": test_acc,
                    },
                    os.path.join(path, fn_torchsave),
                )
            description = f"Loss: {train_loss:.4f}/{test_loss:.4f}; AUC: {train_auc:.4f}/{test_auc:.4f}; Accuracy: {train_acc:.4f}/{test_acc:.4f}"
            pbar.set_description(description)
            pbar.update()
        pbar.close()
    columns = ["loss", "auc", "acc"]
    train_results = pd.DataFrame(train_results, columns=columns)
    test_results = pd.DataFrame(test_results, columns=columns)

    return model, train_results, test_results


# %%
def load_best_model(model, path):
    checkpoint = torch.load(os.path.join(path, fn_torchsave))
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


# %%
def save_submission(model, validation_loader, device, path):
    y_true, y_pred = predict(model, validation_loader, device)
    np.savetxt(os.path.join(path, fn_submission), y_pred, fmt="%d")


# %%
def plot_results(train_results, test_results, path):
    fig, axs = plt.subplots(3, 1, figsize=(6, 8), sharex=True)
    plt.subplots_adjust(hspace=0.1)
    axs[0].plot(train_results.loss, label="train")
    axs[0].plot(test_results.loss, label="test")
    axs[0].set_ylabel("Loss")
    # axs[0].set_ylim(0,None)

    axs[1].plot(train_results.auc, label="train")
    axs[1].plot(test_results.auc, label="test")
    axs[1].set_ylabel("ROC AUC")
    axs[1].set_ylim(0, 1)

    axs[2].plot(train_results.acc, label="train")
    axs[2].plot(test_results.acc, label="test")
    axs[2].set_xlabel("Epoch")
    axs[2].set_ylabel("Accuracy")
    axs[2].set_ylim(0, 1)
    axs[2].legend()

    fig.align_ylabels(axs)

    fig.savefig(os.path.join(path, fn_training), bbox_inches="tight")


# %%
from matplotlib.colors import LogNorm


def plot_confusion_matrix(y_true, y_pred, path, n_classes=11):
    bins = range(1, n_classes + 2)
    H, xedges, yedges = np.histogram2d(y_pred, y_true, bins=[bins, bins])

    plt.figure(figsize=(6, 6))
    plt.axline((0, 0), slope=1, ls="--", c="k", lw=0.5)
    sns.heatmap(
        data=H.astype(int),
        fmt="d",
        norm=LogNorm(),
        vmin=0,
        square=True,
        annot=True,
        cmap="YlGnBu",
        cbar=False,
        xticklabels=xedges,
        yticklabels=yedges,
    )
    plt.xlim(0, n_classes)
    plt.ylim(0, n_classes)
    plt.xlabel("True Class")
    plt.ylabel("Predicted Class")

    plt.savefig(os.path.join(path, fn_confusion), bbox_inches="tight")


# %%

def alignment_loss(x_aligned, labels):
    loss = 0
    classes = labels.unique()
    for c in classes:
        loss += x_aligned[labels == c].var(dim=0, unbiased=False).mean(dim=1).mean()
    loss /= len(classes)
    return loss

def custom_loss(z_dat, y_dat, x_dat_aligned=None, x_ref_aligned=None, individual=None):
    w1, w2, w3 = 1, 1, 1
        
    # L1) alignment loss per individual: x_dat_aligned
    L1 = 0 if x_ref_aligned is None else alignment_loss(x_ref_aligned, individual)

    # L2) alignment loss per individual: x_ref_aligned
    L2 = 0 if x_dat_aligned is None else alignment_loss(x_dat_aligned, individual)

    # L3) classification loss:
    L3 = torch.nn.CrossEntropyLoss(reduction="mean")(z_dat, y_dat)

    loss = w1 * L1 + w2 * L2 + w3 * L3
    return loss


# %%

from src.dataset import load_data


def run_pipeline(
    model, path, max_n_features, batch_size, test_size, epochs, lr, n_classes, train
):
    # RANDOM STATE
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)

    # DATASET
    train_loader, test_loader, validation_loader = load_data(
        max_n_features, batch_size, test_size
    )

    # SETUP
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # loss_func = torch.nn.CrossEntropyLoss(reduction="mean")
    loss_func = custom_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if train:
        # TRAINING
        model, train_results, test_results = train_and_evaluate(
            model, epochs, optimizer, loss_func, train_loader, test_loader, device, path
        )

        # PLOT TRAINING
        plot_results(train_results, test_results, path)

    # LOAD BEST MODEL
    model = load_best_model(model, path)

    # TEST LOSS
    test_loss, test_auc, test_acc = evaluate(
        model, test_loader, loss_func, device, log=True
    )

    # TEST CONFUSION MATRIX
    y_true, y_pred = predict(model, test_loader, device)
    plot_confusion_matrix(y_true, y_pred, path, n_classes)

    # SUBMISSION FILE
    save_submission(model, validation_loader, device, path)

    return True
