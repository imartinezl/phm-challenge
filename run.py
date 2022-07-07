# %% SETUP

import argparse
from src.utils import Parameters, setup

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GKN Grading Predictor')

    parser.add_argument('--max-n-features', type=int, help='max-n-features', default=748)
    parser.add_argument('--n-classes', type=int, help='n-classes', default=11)
    parser.add_argument('--batch-size', type=int, help='batch size', default=32)
    parser.add_argument('--test-size', type=float, help='test size', default=0.3)
    parser.add_argument('--epochs', type=int, help='epochs', default=100)
    parser.add_argument('--lr', type=float, help='batch size', default=0.001)

    args = Parameters(vars(parser.parse_args()))

args = Parameters(
    {
        "max_n_features": 748,
        "n_classes": 11,
        "batch_size": 10,
        "test_size": 0.3,
        "epochs": 1,
        "lr": 0.003,
    }
)
path = setup(args)

# %% MODEL

cnn_params = {
    "n_channels": 3,
    "n_features": args.max_n_features,
    "dropout": None,
}
fcn_params = {
    "hidden_size": 10,
    "hidden_layers": 2,
    "dropout": None,
    "output_size": args.n_classes,
}

from src.models import CustomModel

model = CustomModel(cnn_params, fcn_params)

# %% RUN

from src.base import run_pipeline

run_pipeline(model, path, **args.dictionary)

# %%
