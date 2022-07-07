# %%

import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset, DataLoader, Subset

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from src.generator import read_folder, preprocess_data

# %%


class CustomDataset(Dataset):
    def __init__(self, folder, max_n_features=748, resample=True, normalize=True):
        metadata, X, Y = read_folder(folder, max_n_features, resample)

        X_dat, Y_dat, X_ref, Y_ref, individual_dat, individual_ref = preprocess_data(
            metadata, X, Y, normalize
        )

        # one-hot encoding
        # n_classes = 11
        # categories = list(range(n_classes))
        encoder = OneHotEncoder(categories="auto", sparse=False).fit(
            np.expand_dims(Y, -1)
        )
        Y_dat_enc = encoder.transform(np.expand_dims(Y_dat, -1))
        Y_ref_enc = encoder.transform(np.expand_dims(Y_ref, -1))

        self.X_dat = X_dat
        self.Y_dat = Y_dat
        self.Y_dat_enc = Y_dat_enc

        self.X_ref = X_ref
        self.Y_ref = Y_ref
        self.Y_ref_enc = Y_ref_enc

        self.individual_dat = individual_dat
        self.individual_ref = individual_ref

        self.n_channels = 3
        self.classes = np.unique(Y)
        self.classes_dat = np.unique(Y_dat)
        self.classes_ref = np.unique(Y_ref)
        self.individuals = np.unique(metadata.individual)

    def __len__(self):
        return len(self.X_dat)

    def __getitem__(self, idx):

        idx_dat = idx
        idx_ref = self.individual_ref == self.individual_dat[idx]

        x_dat = self.X_dat[idx_dat]
        x_ref = self.X_ref[idx_ref]

        y_dat = self.Y_dat_enc[idx_dat]
        y_ref = self.Y_ref_enc[idx_ref]

        individual = self.individual_dat[idx]

        return x_dat, x_ref, y_dat, y_ref, individual

    def get_individual(self, individual):

        idx_dat = self.individual_dat == individual
        idx_ref = self.individual_ref == individual

        x_dat = self.X_dat[idx_dat]
        x_ref = self.X_ref[idx_ref]

        y_dat = self.Y_dat[idx_dat]
        y_ref = self.Y_ref[idx_ref]

        return x_dat, x_ref, y_dat, y_ref, individual

    def plot_timeline(self, data_type="dat", average=True, save=True):
        cond = data_type == "dat"
        classes = self.classes_dat if cond else self.classes_ref

        for y in classes:
            h, w = self.n_channels, len(self.individuals)
            fig, axs = plt.subplots(
                h, w, figsize=(w * 2, h * 2), dpi=150, sharex=True, sharey="row"
            )
            plt.subplots_adjust(hspace=0, wspace=0)
            axs = np.expand_dims(axs, -1) if axs.ndim == 1 else axs
            for k, individual in enumerate(self.individuals):
                x_dat, x_ref, y_dat, y_ref, individual = self.get_individual(individual)

                for channel in range(self.n_channels):
                    x = (
                        x_dat[y_dat == y, :, channel]
                        if cond
                        else x_ref[y_ref == y, :, channel]
                    )
                    alpha = max(0.01, 1 / len(x))
                    axs[channel, k].plot(x.T, c="black", alpha=alpha)

                    if average:
                        t = range(x.shape[1])
                        x_mean = x.mean(axis=0)
                        x_std = x.std(axis=0)
                        x_upper = x_mean + x_std
                        x_lower = x_mean - x_std

                        axs[channel, k].plot(x_mean, c="red", alpha=0.8, label=r"$\mu$")
                        axs[channel, k].fill_between(
                            t,
                            x_upper,
                            x_lower,
                            color="red",
                            alpha=0.5,
                            label=r"$\pm\sigma$",
                        )

                axs[0, k].set_title(f"Individual {individual}")

            plt.suptitle(f"Class {y}" if cond else f"Reference")
            if save:
                filename = os.path.join(
                    "figures",
                    "plot_timeline_" + ("class" if cond else "ref") + f"_{y}.jpg",
                )
                fig.savefig(filename, bbox_inches="tight")
                plt.close(fig)

    def plot_heatmap(self, data_type="dat", average=True, save=True):
        cond = data_type == "dat"
        classes = self.classes_dat if cond else self.classes_ref

        for y in classes:
            h, w = self.n_channels * 2, len(self.individuals)
            fig, axs = plt.subplots(
                h, w, figsize=(w * 4, h * 2), dpi=150, sharex=True, sharey="row"
            )
            plt.subplots_adjust(hspace=0.05, wspace=0.05)
            axs = np.expand_dims(axs, -1) if axs.ndim == 1 else axs
            for k, individual in enumerate(self.individuals):
                x_dat, x_ref, y_dat, y_ref, individual = self.get_individual(individual)

                for channel in range(self.n_channels):
                    x = (
                        x_dat[y_dat == y, :, channel]
                        if cond
                        else x_ref[y_ref == y, :, channel]
                    )
                    alpha = max(0.01, 1 / len(x))
                    axs[channel * 2, k].plot(x.T, c="black", lw=0.5, alpha=alpha)

                    if average:
                        t = range(x.shape[1])
                        x_mean = x.mean(axis=0)
                        x_std = x.std(axis=0)
                        x_upper = x_mean + 3 * x_std
                        x_lower = x_mean - 3 * x_std

                        axs[channel * 2, k].plot(
                            x_mean, c="white", lw=1, alpha=1.0, label=r"$\mu$"
                        )
                        axs[channel * 2, k].fill_between(
                            t,
                            x_upper,
                            x_lower,
                            color="red",
                            alpha=0.3,
                            label=r"$\pm\sigma$",
                        )

                    axs[channel * 2 + 1, k].imshow(x)

                    if k > 0:
                        axs[channel * 2, k].set_xticks([])
                        axs[channel * 2, k].set_yticks([])
                        axs[channel * 2 + 1, k].set_xticks([])
                        axs[channel * 2 + 1, k].set_yticks([])

                    # axs[channel*2, k].set_xlim(-5,None)
                    axs[channel * 2, k].set_ylim(-0.1, 1.1)
                    # axs[channel*2, k].set_aspect(1./axs[channel*2, k].get_data_ratio())
                    # axs[channel*2+1, k].set_aspect(1./axs[channel*2+1, k].get_data_ratio())
                axs[0, k].set_title(f"Individual {individual}")

            for ax in axs.flatten():
                ax.set_anchor("N")

            plt.suptitle(f"Class {y}" if cond else f"Reference")
            if save:
                filename = os.path.join(
                    "figures",
                    "plot_heatmap_" + ("class" if cond else "ref") + f"_{y}.jpg",
                )
                fig.savefig(filename, bbox_inches="tight")
                plt.close(fig)

    def plot_error(self, data_type="dat", save=True):
        cond = data_type == "dat"
        classes = self.classes_dat if cond else self.classes_ref

        for y in classes:
            h, w = self.n_channels * 2, len(self.individuals)
            fig, axs = plt.subplots(
                h, w, figsize=(w * 4, h * 2), dpi=150, sharex=True, sharey="row"
            )
            plt.subplots_adjust(hspace=0.05, wspace=0.05)
            axs = np.expand_dims(axs, -1) if axs.ndim == 1 else axs
            for k, individual in enumerate(self.individuals):
                x_dat, x_ref, y_dat, y_ref, individual = self.get_individual(individual)

                for channel in range(self.n_channels):
                    x = (
                        x_dat[y_dat == y, :, channel]
                        if cond
                        else x_ref[y_ref == y, :, channel]
                    )
                    alpha = max(0.01, 1 / len(x))

                    t = range(x.shape[1])
                    x_mean = x.mean(axis=0)
                    x_std = x.std(axis=0)
                    x_upper = x_mean + 3 * x_std
                    x_lower = x_mean - 3 * x_std
                    x_error = x - x_mean

                    axs[channel * 2, k].plot(x.T, c="black", lw=0.5, alpha=alpha)
                    axs[channel * 2, k].plot(
                        x_mean, c="white", lw=1, alpha=1.0, label=r"$\mu$"
                    )
                    axs[channel * 2, k].fill_between(
                        t,
                        x_upper,
                        x_lower,
                        color="red",
                        alpha=0.3,
                        label=r"$\pm\sigma$",
                    )
                    axs[channel * 2, k].set_ylim(-0.1, 1.1)

                    axs[channel * 2 + 1, k].plot(
                        x_error.T, c="black", lw=0.5, alpha=alpha, label=r"error"
                    )
                    axs[channel * 2 + 1, k].set_ylim(-1, 1)

                axs[0, k].set_title(f"Individual {individual}")

            for ax in axs.flatten():
                ax.set_anchor("N")

            plt.suptitle(f"Class {y}" if cond else f"Reference")
            if save:
                filename = os.path.join(
                    "figures",
                    "plot_error_" + ("class" if cond else "ref") + f"_{y}.jpg",
                )
                fig.savefig(filename, bbox_inches="tight")
                plt.close(fig)

    def plot(self):
        for data_type in ["ref", "dat"]:
            self.plot_timeline(data_type)
            self.plot_heatmap(data_type)
            self.plot_error(data_type)


# %%


def get_loader(
    folder,
    max_n_features,
    resample,
    normalize,
    test_size,
    shuffle_split,
    shuffle_sample,
    batch_size,
):
    dataset = CustomDataset(folder, max_n_features, resample, normalize)
    idx = range(len(dataset))
    if test_size is None:
        return DataLoader(dataset, batch_size, shuffle_sample)
    else:
        train_idx, test_idx = train_test_split(
            idx, test_size=test_size, shuffle=shuffle_split, random_state=0
        )
        train_dataset = Subset(dataset, train_idx)
        test_dataset = Subset(dataset, test_idx)
        train_loader = DataLoader(train_dataset, batch_size, shuffle_sample)
        test_loader = DataLoader(test_dataset, batch_size, shuffle_sample)
        return train_loader, test_loader


# %%


def load_data(max_n_features=748, batch_size=32, test_size=0.3):

    folder_train = "data/training_data_reduced"
    folder_test = "data/testing_data"

    resample = True
    normalize = True

    train_loader_params = {
        "folder": folder_train,
        "max_n_features": max_n_features,
        "resample": resample,
        "normalize": normalize,
        "test_size": test_size,
        "shuffle_split": True,
        "shuffle_sample": True,
        "batch_size": batch_size,
    }

    validation_loader_params = {
        "folder": folder_test,
        "max_n_features": max_n_features,
        "resample": resample,
        "normalize": normalize,
        "test_size": None,
        "shuffle_split": None,
        "shuffle_sample": False,  # ALWAYS FALSE
        "batch_size": batch_size,
    }

    train_loader, test_loader = get_loader(**train_loader_params)
    validation_loader = get_loader(**validation_loader_params)
    # train_loader.dataset.plot()

    return train_loader, test_loader, validation_loader
