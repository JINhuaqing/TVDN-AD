import sys
import numpy as np
from numpy.linalg import inv
from pathlib import Path
import pickle


def in_notebook():
    """
    Return True if the module is runing in Ipython kernel
    """
    return "ipykernel" in sys.modules


def load_pkl(fil):
    with open(fil, "rb") as f:
        result = pickle.load(f)
    return result


# save file to pkl
def save_pkl(fil, result, is_force=False):
    if is_force or (not fil.exists()):
        print(f"Save to {fil}")
        with open(fil, "wb") as f:
            pickle.dump(result, f)
    else:
        print(f"{fil} exists! Use is_force=True to save it anyway")

