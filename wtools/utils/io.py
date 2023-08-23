#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import pickle as pkl
try:
    from collections import MutableMapping
except ImportError as e:
    from collections.abc import MutableMapping
from pathlib import Path

import lmdb
import numpy as np
import yaml
from typing import Generic, Iterator, TypeVar, Tuple

T = TypeVar("T")
KT = TypeVar("KT")
VT = TypeVar("VT")


class MissingOk:

    # for python < 3.8 compatibility

    def __init__(self, ok: bool) -> None:
        self.ok = ok

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if isinstance(exc_value, FileNotFoundError) and self.ok:
            return True


def remove_lmdbm(file: str, missing_ok: bool = True) -> None:

    base = Path(file)
    with MissingOk(missing_ok):
        (base / "data.mdb").unlink()
    with MissingOk(missing_ok):
        (base / "lock.mdb").unlink()
    with MissingOk(missing_ok):
        base.rmdir()


class LMDB(MutableMapping, Generic[KT, VT]):
    # reference: https://github.com/Dobatymo/lmdb-python-dbm/blob/master/lmdbm/lmdbm.py#L185
    def __init__(self, path, flag="r", mode=0o755, map_size=1e12) -> None:
        if flag == "r":  # Open existing database for reading only (default)
            env = lmdb.open(
                path,
                map_size=map_size,
                max_dbs=1,
                readonly=True,
                create=False,
                mode=mode,
            )
        elif flag == "w":  # Open existing database for reading and writing
            env = lmdb.open(
                path,
                map_size=map_size,
                max_dbs=1,
                readonly=False,
                create=False,
                mode=mode,
            )
        elif (
            flag == "c"
        ):  # Open database for reading and writing, creating it if it doesn't exist
            env = lmdb.open(
                path,
                map_size=map_size,
                max_dbs=1,
                readonly=False,
                create=True,
                mode=mode,
            )
        elif (
            flag == "n"
        ):  # Always create a new, empty database, open for reading and writing
            remove_lmdbm(path)
            env = lmdb.open(
                path,
                map_size=map_size,
                max_dbs=1,
                readonly=False,
                create=True,
                mode=mode,
            )
        else:
            raise ValueError("Invalid flag")
        self.env = env

    @property
    def map_size(self) -> int:

        return self.env.info()["map_size"]

    def _pre_key(self, key: KT) -> bytes:
        if isinstance(key, bytes):
            return key
        elif isinstance(key, str):
            return key.encode("Latin-1")

        raise TypeError(key)

    def _post_key(self, key: bytes) -> KT:
        return key

    def _pre_value(self, value: VT) -> bytes:
        if isinstance(value, bytes):
            return value
        elif isinstance(value, str):
            return value.encode("Latin-1")
        raise TypeError(value)

    def _post_value(self, value: bytes) -> VT:
        return value

    @map_size.setter
    def map_size(self, value: int) -> None:
        self.env.set_mapsize(value)

    def __getitem__(self, key: KT) -> VT:
        if isinstance(key, str):
            key = key.encode()
        with self.env.begin() as txn:
            value = txn.get(self._pre_key(key))
        if value is None:
            raise KeyError(key)
        return self._post_value(value)

    def __setitem__(self, key: KT, value: VT):
        with self.env.begin(write=True) as txn:
            txn.put(self._pre_key(key), self._pre_value(value))

    def __delitem__(self, key: KT) -> None:
        with self.env.begin(write=True) as txn:
            txn.delete(self._pre_key(key))

    def update(self, key: KT, value: VT):
        with self.env.begin(write=True) as txn:
            txn.replace(self._pre_key(key), self._pre_value(value))

    def keys(self):
        with self.env.begin() as txn:
            for key in txn.cursor().iternext(keys=True, values=False):
                yield self._post_key(key)

    def values(self):
        with self.env.begin() as txn:
            for value in txn.cursor().iternext(keys=False, values=True):
                yield self._post_value(value)

    def items(self) -> Iterator[Tuple[KT, VT]]:
        with self.env.begin() as txn:
            for key, value in txn.cursor().iternext(keys=True, values=True):
                yield (self._post_key(key), self._post_value(value))

    def __len__(self) -> int:
        with self.env.begin() as txn:
            return txn.stat()["entries"]

    def __iter__(self):
        return self.keys()

    def sync(self):
        self.env.sync()

    def close(self):
        self.env.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


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
