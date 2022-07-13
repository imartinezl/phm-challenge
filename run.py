# %% SETUP
import argparse

parser = argparse.ArgumentParser(description='GKN Grading Predictor')

parser.add_argument('--folder', type=str, help='folder', default="results")
parser.add_argument('--max-n-features', type=int, help='max-n-features', default=748)
parser.add_argument('--n-classes', type=int, help='n-classes', default=11)
parser.add_argument('--batch-size', type=int, help='batch size', default=32)
parser.add_argument('--test-size', type=float, help='test size', default=0.4)
parser.add_argument('--epochs', type=int, help='epochs', default=200)
parser.add_argument('--lr', type=float, help='batch size', default=0.0001)
# parser.add_argument('--path', type=str, help='path', default="test")
parser.add_argument('--path', type=str, help='path', default=None)
parser.add_argument('--train', type=bool, help='train/evaluate model', default=True)

args = parser.parse_args()

# %% SETUP
from src.utils import setup
path = setup(args)

# %% MODEL 1

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

# model = CustomModel(cnn_params, fcn_params)

# %% MODEL 2
params = {
    "in_channels": 3,
    "out_channels": args.n_classes,
    "channels": [128,128,128],
    "kernels": [8,8,8],
    "strides": [1,1,1]
}

from src.models2 import CustomModel

# model = CustomModel(params)

# %% MODEL 3
params_alignment_ref = {
    "tess_size": 100,
    "zero_boundary": True,
    "n_recurrence": 4,
    "outsize": args.max_n_features,
    "N": 4,
    "in_channels": 3,
    "out_channels": 16,
    "channels": [30, 20, 10],
    "kernels": [24, 16, 8],
    "strides": [1, 1, 1],
}
params_alignment_dat = {
    "tess_size": 100,
    "zero_boundary": True,
    "n_recurrence": 4,
    "outsize": args.max_n_features,
    "N": 4,
    "in_channels": 6,
    "out_channels": 16,
    "channels": [30, 20, 10],
    "kernels": [24, 16, 8],
    "strides": [1, 1, 1],
}

params_classification = {
    "in_channels": 6,
    "out_channels": args.n_classes,
    "channels": [128,128,128],
    "kernels": [8,8,8],
    "strides": [1,1,1]
}

from src.models2 import CustomModelAlign

# model = CustomModelAlign(params_alignment_ref, params_alignment_dat, params_classification)

# %% MODEL BASELINE

from src.models2 import BaselineModel

params = {
    "input_size": args.max_n_features,
    "hidden_size": 10,
    "hidden_layers": 0,
    "dropout": None,
    "output_size": args.n_classes,
}
model = BaselineModel(params)


# %% RUN

from src.base import run_pipeline

run_pipeline(model, path, args.max_n_features, args.batch_size, args.test_size, args.epochs, args.lr, args.n_classes, args.train)
