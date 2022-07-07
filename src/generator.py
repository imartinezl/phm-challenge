# %%

import os
import numpy as np
import pandas as pd

import scipy
import scipy.signal

# %%


def read_metadata(filename):
    fn = filename.split(".")[0]
    data_type, signal = fn.split("_")
    individual = signal[-1]
    signal = signal[:-1]

    return data_type, signal, individual


def read_file(folder, filename, max_n_features=None, resample=True):
    path = os.path.join(folder, filename)

    data_type, signal, individual = read_metadata(filename)

    data = pd.read_table(path, header=None).values.flatten().astype(str)
    data = np.char.split(data, ",").tolist()
    n_signals = len(data)

    max_n_features = (
        max(map(len, data)) - 1 if max_n_features is None else max_n_features
    )

    metadata = np.array([folder, filename, data_type, signal, individual])
    metadata = np.tile(metadata, (n_signals, 1))
    metadata = np.insert(metadata, metadata.shape[1], range(n_signals), axis=1)

    Y = np.zeros((n_signals), dtype=int)

    if resample:
        X = np.empty((n_signals, max_n_features))
        for i, d in enumerate(data):
            Y[i] = int(d[0])
            x = np.float32(d[1:])
            X[i] = scipy.signal.resample(x, max_n_features)
    else:
        X = np.full((n_signals, max_n_features), np.nan)
        for i, d in enumerate(data):
            Y[i] = int(d[0])
            n = len(d[1:])
            X[i, :n] = np.float32(d[1:])

    return metadata, X, Y


def read_folder(folder, max_n_features=None, resample=True):
    metadata, X, Y = [], [], []
    for filename in os.listdir(folder):
        # print("LOADED", filename)
        metadata_, X_, Y_ = read_file(folder, filename, max_n_features, resample)
        metadata.append(metadata_)
        X.append(X_)
        Y.append(Y_)

    metadata_cols = ["folder", "filename", "data_type", "signal", "individual", "ts"]
    metadata = np.vstack(metadata)
    metadata = pd.DataFrame(metadata, columns=metadata_cols)
    metadata.individual = metadata.individual.astype(int)
    metadata.ts = metadata.ts.astype(int)

    X = np.vstack(X).astype(np.float32)
    Y = np.concatenate(Y).astype(np.int32)

    return metadata, X, Y


# %%


def normalize_01(X, min_value, max_value):
    ptp_value = max_value - min_value
    return np.divide(np.subtract(X, min_value), ptp_value)


def interpolate_signal(X):
    n_signals, n_features, n_channels = X.shape
    xp = np.arange(n_features)
    return np.stack(
        [
            np.column_stack(
                [
                    np.interp(
                        xp,
                        np.arange(sum(np.isfinite(x[:, c]))),
                        x[:, c][np.isfinite(x[:, c])],
                        n_features,
                    )
                    for c in range(n_channels)
                ]
            )
            for x in X
        ]
    )


def resample_signal(X):
    n_signals, n_features, n_channels = X.shape
    return np.stack(
        [
            np.column_stack(
                [
                    scipy.signal.resample(x[:, c][np.isfinite(x[:, c])], n_features)
                    for c in range(n_channels)
                ]
            )
            for x in X
        ]
    )
    # return np.stack([scipy.signal.resample(x[np.isfinite(x).reshape(-1, n_channels)], n_features) for x in X])


# %%


def preprocess_data(metadata, X, Y, normalize=True):
    n_features = X.shape[1]
    n_channels = 3

    # relevant metadata order
    metadata = metadata.sort_values(["data_type", "individual", "ts", "signal"])

    meta_dat = metadata[metadata.data_type == "data"]
    meta_ref = metadata[metadata.data_type == "ref"]

    X_dat = X[meta_dat.index].reshape(-1, n_channels, n_features).transpose(0, 2, 1)
    X_ref = X[meta_ref.index].reshape(-1, n_channels, n_features).transpose(0, 2, 1)

    Y_dat = Y[meta_dat.index].reshape(-1, n_channels)[:, 0]
    Y_ref = Y[meta_ref.index].reshape(-1, n_channels)[:, 0]

    if normalize:
        X_tmp = X[metadata.index].reshape(-1, n_channels, n_features).transpose(0, 2, 1)
        min_value = np.nanmin(X_tmp, axis=(0, 1))
        max_value = np.nanmax(X_tmp, axis=(0, 1))
        X_dat = normalize_01(X_dat, min_value, max_value)
        X_ref = normalize_01(X_ref, min_value, max_value)

    individual_dat = (
        meta_dat.drop(["filename", "signal"], axis=1)
        .drop_duplicates(ignore_index=True)
        .individual
    )
    individual_ref = (
        meta_ref.drop(["filename", "signal"], axis=1)
        .drop_duplicates(ignore_index=True)
        .individual
    )

    return X_dat, Y_dat, X_ref, Y_ref, individual_dat, individual_ref


# %%

if __name__ == "__main__":
    folder = "training_data_reduced"
    max_n_features = 748
    resample = True
    normalize = True
    metadata, X, Y = read_folder(folder, max_n_features, resample, normalize)
