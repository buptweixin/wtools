#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle as pkl
import json
import yaml


def load_pickle(path: str, verbose=True):
    if verbose:
        print(f"Loading pickle file from {path}", end="")
    with open(path, "rb") as f:
        data = pkl.load(f)
    if verbose:
        print(" => Done.")
    return data


def dump_pickle(obj, path, verbose=True):
    if verbose:
        print(f"Dumping pickle file to {path}", end="")
    with open(path, "wb") as f:
        pkl.dump(obj, f)
    if verbose:
        print(" => Done.")


def load_json(path: str, verbose=True):
    if verbose:
        print(f"Loading json file from {path}", end="")
    with open(path, "r") as f:
        data = json.load(f)
    if verbose:
        print(" => Done.")
    return data


def dump_json(obj, path, verbose=True):
    if verbose:
        print(f"Dumping json file to {path}", end="")
    with open(path, "w") as f:
        json.dump(obj, f)
    if verbose:
        print(" => Done.")


def load_yaml(path, ):
    """_summary_

    Arguments:
        path {_type_} -- _description_
    """
    pass