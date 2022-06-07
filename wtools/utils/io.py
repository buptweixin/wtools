#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import pickle as pkl

import numpy as np
import yaml


def load_pickle(path: str, verbose=False):
    if verbose:
        print(f"Loading pickle file from {path}", end="")
    with open(path, "rb") as f:
        data = pkl.load(f)
    if verbose:
        print(" => Done.")
    return data


def dump_pickle(obj, path, verbose=False):
    if verbose:
        print(f"Dumping pickle file to {path}", end="")
    with open(path, "wb") as f:
        pkl.dump(obj, f)
    if verbose:
        print(" => Done.")


def load_json(path: str, verbose=False):
    if verbose:
        print(f"Loading json file from {path}", end="")
    with open(path, "r") as f:
        data = json.load(f)
    if verbose:
        print(" => Done.")
    return data


def dump_json(obj, path, verbose=False):
    if verbose:
        print(f"Dumping json file to {path}", end="")
    with open(path, "w") as f:
        json.dump(obj, f)
    if verbose:
        print(" => Done.")


def load_yaml(path, verbose=False):
    """_summary_

    Arguments:
        path {_type_} -- _description_
    """
    if verbose:
        print(f"Loading yaml file from {path}", end="")
    with open(path, "r") as f:
        data = yaml.load(f)
    if verbose:
        print(" => Done.")
    return data


def load_pts(path, verbose=False):
    if verbose:
        print(f"Loading pts file from {path}", end="")
    with open(path, "r") as f:
        data = [l.strip().split() for l in f.readlines()]
    data = np.array(data, dtype=np.float32)
    if verbose:
        print(" => Done.")
    return data


def dump_pts(obj, path, verbose=False):
    if verbose:
        print(f"Dumping pts object to {path}", end="")
    with open(path, "w") as f:
        f.write("\n".join([" ".join(map(str, p)) for p in obj]))
    if verbose:
        print(" => Done.")


def dump_jsonlines(obj, path, verbose=False):
    if verbose:
        print(f"Dumping jsonlines file to {path}", end="")
    with open(path, "w") as f:
        f.write("\n".join(json.dumps(line) for line in obj))
    if verbose:
        print(" => Done.")


def load_jsonlines(path, verbose=False):
    if verbose:
        print(f"Loading jsonlines file from {path}", end="")
    with open(path, "r") as f:
        data = [json.loads(l.strip()) for l in f.readlines()]
    if verbose:
        print(" => Done.")
    return data
