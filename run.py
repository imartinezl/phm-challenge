# %% SETUP
import argparse

parser = argparse.ArgumentParser(description='GKN Grading Predictor')

parser.add_argument('--folder', type=str, help='folder', default="results")
parser.add_argument('--max-n-features', type=int, help='max-n-features', default=748)
parser.add_argument('--n-classes', type=int, help='n-classes', default=11)
parser.add_argument('--batch-size', type=int, help='batch size', default=32)
parser.add_argument('--test-size', type=float, help='test size', default=0.3)
parser.add_argument('--epochs', type=int, help='epochs', default=100)
parser.add_argument('--lr', type=float, help='batch size', default=0.001)
parser.add_argument('--path', type=str, help='path', default=None)
parser.add_argument('--train', type=bool, help='train/evaluate model', default=True)

args = parser.parse_args()

# %% SETUP
from src.utils import setup
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

# %%
params = {
    "in_channels": 3,
    "out_classes": args.n_classes,
    "channels": [128,128,128],
    "kernels": [8,8,8],
    "strides": [1,1,1]
}

from src.models2 import CustomModel

model = CustomModel(params)

# %% RUN

from src.base import run_pipeline

run_pipeline(model, path, args.max_n_features, args.batch_size, args.test_size, args.epochs, args.lr, args.n_classes, args.train)
