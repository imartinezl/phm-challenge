# %%

import os
import sys
import time
import json
from pathlib import Path

# %%


def print_command():
    print("python " + " ".join(sys.argv))


def now():
    return round(time.time() * 1000)


def get_path(args):
    if args.path is None:
        return os.path.join(args.folder, str(now()))
    return os.path.join(args.folder, args.path)


def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_args(args, path, name="config"):
    filename = os.path.join(path, name + ".json")
    if os.path.isfile(filename):
        return
    with open(filename, "w") as fp:
        json.dump(vars(args), fp)


def setup(args):
    print_command()
    path = get_path(args)
    mkdir(path)
    save_args(args, path)
    return path

# %%

class Parameters:
    def __init__(self, dictionary):
        self.dictionary = dictionary
        for k, v in dictionary.items():
            setattr(self, k, v)

    def __str__(self):
        return self.dictionary.__str__()
