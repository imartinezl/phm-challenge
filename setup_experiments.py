# %% CONFIGURATION
import os
from pathlib import Path
import shutil

folder = ["results_validation"]
max_n_features = [748]
n_classes = [11]
batch_size = [64]
test_size = [0.4]
epochs = [200]
lr = [5e-5]
train = [True]
model = ["BaselineModel", "CustomModel", "CustomModelRef", "CustomModelAlign"]
cost_classification = [1]
cost_align_ref = [1, 10, 100]
cost_align_dat = [1, 10, 100]

from itertools import product

hyperparameters = product(
    folder, max_n_features, n_classes, batch_size, test_size,
    epochs, lr, train,
    model, cost_classification, cost_align_ref, cost_align_dat
)

directory = "runs"
# if os.path.isdir(directory):
#     shutil.rmtree(directory)
Path(directory).mkdir(parents=True, exist_ok=True)

for k, params in enumerate(hyperparameters):
    f = open(os.path.join(directory, f"run{k+1:03}.sh"), "w")
    command = """
cd .. \n
python run.py  \
--folder {} \
--max-n-features {} \
--n-classes {} \
--batch-size {} \
--test-size {} \
--epochs {} \
--lr {} \
--train {} \
--model {} \
--cost-classification {} \
--cost-align-ref {} \
--cost-align-dat {} \
""".format(
        *params
    )
    f.write(command)
    f.write("\n")
    f.close()

print("Prepared {} experiments".format(k + 1))
