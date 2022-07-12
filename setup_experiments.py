# %% CONFIGURATION
import os
from pathlib import Path

folder = ["results"]
max_n_features = [748]
n_classes = [11]
batch_size = [64]
test_size = [0.3]
epochs = [300]
lr = [5e-3]
path = [None]
train = [True]

from itertools import product

hyperparameters = product(
    folder, max_n_features, n_classes, batch_size, test_size, epochs, lr, path, train
)

directory = "runs"
Path(directory).mkdir(parents=True, exist_ok=True)

for k, params in enumerate(hyperparameters):
    f = open(os.path.join(directory, f"run{k+1:03}.sh"), "w")
    command = """
    cd .. \n
python main.py  \
--folder {} \
--max-n-features {} \
--batch-size {} \
--test-size {} \
--epochs {} \
--lr {} \
--path {} \
--train {} \
""".format(
        *params
    )
    f.write(command)
    f.write("\n")
    f.close()

print("Prepared {} experiments".format(k + 1))
