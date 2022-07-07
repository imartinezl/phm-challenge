# %%

import os
import sys
import time
import json
from pathlib import Path

# %%


def print_args(args):
    print("python " + " ".join(sys.argv))


def now():
    return round(time.time() * 1000)


def get_path(folder="results"):
    return os.path.join(folder, str(now()))


def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_args(args, path, name="config"):
    with open(os.path.join(path, name + ".json"), "w") as fp:
        json.dump(args.dictionary, fp)


def setup(args):
    print_args(args)
    path = get_path()
    mkdir(path)
    save_args(args, path)
    return path


class Parameters:
    def __init__(self, dictionary):
        self.dictionary = dictionary
        for k, v in dictionary.items():
            setattr(self, k, v)

    def __str__(self):
        return self.dictionary.__str__()
