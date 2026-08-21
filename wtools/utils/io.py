#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import pickle as pkl
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any, Generic, Iterable, Iterator, List, Optional, Tuple, TypeVar, Union

import lmdb
import numpy as np
import yaml

T = TypeVar("T")
KT = TypeVar("KT")
VT = TypeVar("VT")


class MissingOk:
    """Context manager that optionally suppresses ``FileNotFoundError``.

    This is a small utility used internally by :func:`remove_lmdbm` to make
    file/directory removal idempotent. When ``ok`` is ``True`` any
    ``FileNotFoundError`` raised inside the ``with`` block is swallowed;
    otherwise the exception propagates as usual.

    Args:
        ok: When ``True``, ``FileNotFoundError`` exceptions raised inside the
            context are suppressed. When ``False``, they propagate normally.

    Examples:
        >>> with MissingOk(True):
        ...     Path("does_not_exist").unlink()
        >>> # No exception is raised even though the file does not exist.
        >>> with MissingOk(False):
        ...     Path("does_not_exist").unlink()
        Traceback (most recent call last):
            ...
        FileNotFoundError: ...
    """

    def __init__(self, ok: bool) -> None:
        self.ok = ok

    def __enter__(self) -> "MissingOk":
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_value: Optional[BaseException],
        traceback: Optional[Any],
    ) -> Optional[bool]:
        if isinstance(exc_value, FileNotFoundError) and self.ok:
            return True
        return None


def remove_lmdbm(file: str, missing_ok: bool = True) -> None:
    """Remove an LMDB database directory and its data/lock files.

    An LMDB database is stored as a directory containing at least
    ``data.mdb`` and ``lock.mdb``. This helper removes both files and then
    the directory itself.

    Args:
        file: Path to the LMDB database directory.
        missing_ok: If ``True`` (default), no error is raised when the
            directory or its files do not exist. If ``False``, a
            ``FileNotFoundError`` is raised for missing paths.

    Examples:
        >>> remove_lmdbm("/path/to/lmdb_db")
        >>> # Removes data.mdb, lock.mdb, and the directory itself.
    """
    base = Path(file)
    with MissingOk(missing_ok):
        (base / "data.mdb").unlink()
    with MissingOk(missing_ok):
        (base / "lock.mdb").unlink()
    with MissingOk(missing_ok):
        base.rmdir()


class LMDB(MutableMapping, Generic[KT, VT]):
    """A dict-like wrapper around an LMDB key-value store.

    This class implements the ``collections.abc.MutableMapping`` protocol so
    that an on-disk LMDB database can be used with the familiar Python dict
    API (``db[key] = value``, ``db.get(key)``, ``del db[key]``, ``len(db)``,
    iteration, etc.).

    Reference: https://github.com/Dobatymo/lmdb-python-dbm/blob/master/lmdbm/lmdbm.py#L185

    Args:
        path: Filesystem path to the LMDB database directory. If the database
            does not yet exist and ``flag`` is ``"c"`` or ``"n"``, a new
            directory is created at this location.
        flag: Open mode, similar to the ``flag`` argument of ``dbm.open``.
            Accepted values:

            - ``"r"`` -- Open an existing database for **read-only** access
              (default). Raises if the database does not exist.
            - ``"w"`` -- Open an existing database for **read and write**
              access. Raises if the database does not exist.
            - ``"c"`` -- Open a database for **read and write**, **creating**
              it if it does not already exist.
            - ``"n"`` -- Always create a **new, empty** database for read and
              write access. Any existing database at ``path`` is removed first.

        mode: File permission bits used when creating the database directory
            and its files. Defaults to ``0o755``.
        map_size: Maximum size in bytes that the database may grow to.
            Defaults to ``1e12`` (approximately 1 TB). The value can be
            changed later via the :attr:`map_size` property.

    Raises:
        ValueError: If ``flag`` is not one of ``"r"``, ``"w"``, ``"c"``, or
            ``"n"``.
        lmdb.Error: If the underlying ``lmdb.open`` call fails (for example
            when opening a non-existent database with ``flag="r"``).

    Examples:
        Create a new database, write a key, and read it back::

            >>> from wtools.utils.io import LMDB
            >>> with LMDB("/tmp/mydb", flag="c") as db:
            ...     db["answer"] = b"42"
            ...     print(db["answer"])
            b'42'

        Open an existing database read-only and iterate over its items::

            >>> with LMDB("/tmp/mydb", flag="r") as db:
            ...     for key, value in db.items():
            ...         print(key, value)

        Re-create a database from scratch with ``flag="n"``::

            >>> with LMDB("/tmp/mydb", flag="n") as db:
            ...     db["fresh"] = b"data"
    """

    # reference: https://github.com/Dobatymo/lmdb-python-dbm/blob/master/lmdbm/lmdbm.py#L185
    def __init__(
        self,
        path: str,
        flag: str = "r",
        mode: int = 0o755,
        map_size: int = int(1e12),
    ) -> None:
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
        """int: The maximum size (in bytes) the database map can grow to."""
        return self.env.info()["map_size"]

    def _pre_key(self, key: KT) -> bytes:
        """Convert a user-facing key into bytes for LMDB storage.

        Args:
            key: A ``str`` or ``bytes`` key.

        Returns:
            The key encoded as ``bytes``.

        Raises:
            TypeError: If ``key`` is neither ``str`` nor ``bytes``.
        """
        if isinstance(key, bytes):
            return key
        elif isinstance(key, str):
            return key.encode("Latin-1")

        raise TypeError(
            f"LMDB key must be str or bytes, got {type(key).__name__}: "
            f"{key!r}"
        )

    def _post_key(self, key: bytes) -> Any:
        """Convert a raw ``bytes`` key from LMDB back to the user-facing type.

        Override in a subclass to customize the return type.

        Args:
            key: Raw ``bytes`` key read from LMDB.

        Returns:
            The key in the type expected by the caller (``bytes`` by default).
        """
        return key

    def _pre_value(self, value: VT) -> bytes:
        """Convert a user-facing value into bytes for LMDB storage.

        Args:
            value: A ``str`` or ``bytes`` value.

        Returns:
            The value encoded as ``bytes``.

        Raises:
            TypeError: If ``value`` is neither ``str`` nor ``bytes``.
        """
        if isinstance(value, bytes):
            return value
        elif isinstance(value, str):
            return value.encode("Latin-1")
        raise TypeError(
            f"LMDB value must be str or bytes, got {type(value).__name__}: "
            f"{value!r}"
        )

    def _post_value(self, value: bytes) -> Any:
        """Convert a raw ``bytes`` value from LMDB back to the user-facing type.

        Override in a subclass to customize the return type.

        Args:
            value: Raw ``bytes`` value read from LMDB.

        Returns:
            The value in the type expected by the caller (``bytes`` by
            default).
        """
        return value

    @map_size.setter  # type: ignore
    def map_size(self, value: int) -> None:
        self.env.set_mapsize(value)

    def __getitem__(self, key: KT) -> Any:
        if isinstance(key, str):
            bkey: Any = key.encode()
        else:
            bkey = key
        with self.env.begin() as txn:
            value = txn.get(self._pre_key(bkey))
        if value is None:
            raise KeyError(key)
        return self._post_value(value)

    def __setitem__(self, key: KT, value: VT) -> None:
        with self.env.begin(write=True) as txn:
            txn.put(self._pre_key(key), self._pre_value(value))

    def __delitem__(self, key: KT) -> None:
        with self.env.begin(write=True) as txn:
            txn.delete(self._pre_key(key))

    def update(self, *args: Any, **kwargs: Any) -> None:
        """Store key-value pairs, optimized for the single-pair fast path.

        When called as ``update(key, value)`` the pair is written in a single
        transaction. Otherwise the standard ``MutableMapping.update``
        behavior is used (accepting a mapping or an iterable of pairs).

        Args:
            *args: Either a single mapping/dict, or a ``(key, value)`` pair
                when exactly two positional arguments are given.
            **kwargs: Additional ``key=value`` pairs to store.

        Examples:
            >>> db.update("k1", "v1")  # fast single-pair write
            >>> db.update({"a": 1, "b": 2})
            >>> db.update(a=1, b=2)
        """
        if len(args) == 2 and not kwargs:
            with self.env.begin(write=True) as txn:
                txn.put(self._pre_key(args[0]), self._pre_value(args[1]))
        else:
            super().update(*args, **kwargs)

    def batch_put(self, items: Iterable[Tuple[KT, VT]]) -> None:
        """Write many key-value pairs in a single transaction.

        Opening one LMDB write transaction per item (as ``__setitem__`` does)
        is extremely slow for bulk inserts because each transaction incurs
        fsync and commit overhead.  This method amortizes that cost by
        committing all puts in one transaction.

        Args:
            items: An iterable of ``(key, value)`` pairs.

        Examples:
            >>> db.batch_put([("k1", b"v1"), ("k2", b"v2"), ("k3", b"v3")])
        """
        with self.env.begin(write=True) as txn:
            for key, value in items:
                txn.put(self._pre_key(key), self._pre_value(value))

    def batch_get(self, keys: Iterable[KT]) -> List[Any]:
        """Read many keys in a single read-only transaction.

        Like :meth:`batch_put`, this avoids the per-key transaction overhead
        of repeated ``__getitem__`` calls.

        Args:
            keys: An iterable of keys to look up.

        Returns:
            A list of values in the same order as *keys*.  Entries for
            missing keys are ``None``.

        Examples:
            >>> db.batch_get(["k1", "k2", "missing"])
            [b'v1', b'v2', None]
        """
        with self.env.begin() as txn:
            return [
                self._post_value(v) if v is not None else None
                for v in (txn.get(self._pre_key(k)) for k in keys)
            ]

    def keys(self) -> Iterator[KT]:  # type: ignore[override]
        """Iterate over all keys in the database.

        Yields:
            The next key in the database, in insertion order.
        """
        with self.env.begin() as txn:
            for key in txn.cursor().iternext(keys=True, values=False):
                yield self._post_key(key)

    def values(self) -> Iterator[VT]:  # type: ignore[override]
        """Iterate over all values in the database.

        Yields:
            The next value in the database, in insertion order.
        """
        with self.env.begin() as txn:
            for value in txn.cursor().iternext(keys=False, values=True):
                yield self._post_value(value)

    def items(self) -> Iterator[Tuple[KT, VT]]:  # type: ignore[override]
        """Iterate over all ``(key, value)`` pairs in the database.

        Yields:
            A tuple ``(key, value)`` for each entry, in insertion order.
        """
        with self.env.begin() as txn:
            for key, value in txn.cursor().iternext(keys=True, values=True):
                yield (self._post_key(key), self._post_value(value))

    def __len__(self) -> int:
        with self.env.begin() as txn:
            return txn.stat()["entries"]

    def __iter__(self) -> Iterator[KT]:
        return self.keys()

    def sync(self) -> None:
        """Flush pending writes from the OS buffer cache to disk."""
        self.env.sync()

    def close(self) -> None:
        """Close the underlying LMDB environment.

        After calling this method the instance can no longer be used. It is
        called automatically when used as a context manager (``with``).
        """
        self.env.close()

    def __enter__(self) -> "LMDB[KT, VT]":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


def load_pickle(path: str, verbose: bool = False) -> Any:
    """Load a Python object from a pickle file.

    .. warning::

        ``pickle.load`` can execute arbitrary code embedded in the pickle
        stream.  **Only load pickle files from trusted sources.**  Never
        use this function with data received from untrusted or unauthenticated
        parties -- a malicious pickle can achieve remote code execution
        during deserialization.

    The file is read in binary mode and deserialized with ``encoding="latin1"``
    to improve compatibility with objects pickled under Python 2.

    Args:
        path: Path to the pickle file.
        verbose: If ``True``, print progress messages before and after
            loading.

    Returns:
        The deserialized Python object.

    Raises:
        FileNotFoundError: If ``path`` does not exist.

    Examples:
        >>> data = load_pickle("data.pkl", verbose=True)
        Loading pickle file from data.pkl => Done.
    """
    if verbose:
        print(f"Loading pickle file from {path}", end="")
    # SECURITY: pickle.load is inherently unsafe -- a crafted pickle stream
    # can execute arbitrary Python code during unpickling.  Only use this
    # function with files from a trusted source.
    with open(path, "rb") as f:
        data = pkl.load(f, encoding="latin1")
    if verbose:
        print(" => Done.")
    return data


def dump_pickle(obj: Any, path: str, verbose: bool = False) -> None:
    """Serialize a Python object to a pickle file.

    Args:
        obj: The Python object to serialize.
        path: Destination file path.
        verbose: If ``True``, print progress messages before and after
            dumping.

    Examples:
        >>> dump_pickle({"a": 1}, "data.pkl", verbose=True)
        Dumping pickle file to data.pkl => Done.
    """
    if verbose:
        print(f"Dumping pickle file to {path}", end="")
    with open(path, "wb") as f:
        pkl.dump(obj, f)
    if verbose:
        print(" => Done.")


def load_json(path: str, verbose: bool = False) -> Any:
    """Load a JSON file into a Python object.

    Args:
        path: Path to the JSON file.
        verbose: If ``True``, print progress messages before and after
            loading.

    Returns:
        The deserialized Python object (typically a ``dict`` or ``list``).

    Examples:
        >>> config = load_json("config.json")
    """
    if verbose:
        print(f"Loading json file from {path}", end="")
    with open(path, "r") as f:
        data = json.load(f)
    if verbose:
        print(" => Done.")
    return data


def dump_json(obj: Any, path: str, verbose: bool = False, ensure_ascii: bool = False) -> None:
    """Serialize a Python object to a JSON file.

    Args:
        obj: The Python object to serialize (must be JSON-serializable).
        path: Destination file path.
        verbose: If ``True``, print progress messages before and after
            dumping.
        ensure_ascii: If ``True``, escape all non-ASCII characters (e.g.
            Chinese text becomes ``\\uXXXX``). If ``False`` (default),
            non-ASCII characters are written as-is, which produces more
            readable output for text containing CJK characters.

    Examples:
        >>> dump_json({"a": 1}, "data.json", verbose=True)
        Dumping json file to data.json => Done.
        >>> dump_json({"name": "中文"}, "data.json")  # Chinese preserved
    """
    if verbose:
        print(f"Dumping json file to {path}", end="")
    with open(path, "w") as f:
        json.dump(obj, f, ensure_ascii=ensure_ascii)
    if verbose:
        print(" => Done.")


def load_yaml(path: str, verbose: bool = False) -> Any:
    """Load a YAML file into a Python object.

    Uses ``yaml.safe_load`` so arbitrary Python object construction is
    disallowed, making it safe to use with untrusted YAML files.

    Args:
        path: Path to the YAML file.
        verbose: If ``True``, print progress messages before and after
            loading.

    Returns:
        The deserialized Python object (typically a ``dict`` or ``list``).

    Examples:
        >>> config = load_yaml("config.yaml", verbose=True)
        Loading yaml file from config.yaml => Done.
    """
    if verbose:
        print(f"Loading yaml file from {path}", end="")
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if verbose:
        print(" => Done.")
    return data


def dump_yaml(obj: Any, path: str, verbose: bool = False) -> None:
    """Serialize a Python object to a YAML file.

    Uses ``yaml.safe_dump`` so only standard YAML types are emitted,
    making the output compatible with ``yaml.safe_load`` (and thus
    :func:`load_yaml`).

    Args:
        obj: The Python object to serialize (must be YAML-serializable,
            e.g. ``dict``, ``list``, ``str``, ``int``, ``float``).
        path: Destination file path.
        verbose: If ``True``, print progress messages before and after
            dumping.

    Examples:
        >>> dump_yaml({"a": 1}, "config.yaml", verbose=True)
        Dumping yaml file to config.yaml => Done.
        >>> # Round-trip with load_yaml:
        >>> config = load_yaml("config.yaml")
    """
    if verbose:
        print(f"Dumping yaml file to {path}", end="")
    with open(path, "w") as f:
        yaml.safe_dump(obj, f)
    if verbose:
        print(" => Done.")


def load_pts(path: str, verbose: bool = False) -> np.ndarray:
    """Load a whitespace-separated points file into a NumPy array.

    Each line in the file is split on whitespace and interpreted as a row of
    float values (typically ``x y`` pairs or higher-dimensional points). All
    rows are stacked into a single ``np.float32`` array.

    Args:
        path: Path to the points file.
        verbose: If ``True``, print progress messages before and after
            loading.

    Returns:
        A ``np.ndarray`` of shape ``(N, D)`` with dtype ``np.float32``,
        where ``N`` is the number of lines and ``D`` is the number of values
        per line.

    Examples:
        >>> pts = load_pts("landmarks.pts")
        >>> pts.shape
        (68, 2)
    """
    if verbose:
        print(f"Loading pts file from {path}", end="")
    # Iterate over the file object directly instead of f.readlines() to avoid
    # materialising the entire file into a list before processing.
    with open(path, "r") as f:
        data: Any = [l.strip().split() for l in f if l.strip()]
    data = np.array(data, dtype=np.float32)
    if verbose:
        print(" => Done.")
    return data  # type: ignore[no-any-return]


def dump_pts(obj: np.ndarray, path: str, verbose: bool = False) -> None:
    """Write an array of points to a whitespace-separated text file.

    Each row of ``obj`` is written on its own line, with values separated by
    a single space.

    Args:
        obj: An iterable of point arrays (e.g. ``np.ndarray`` of shape
            ``(N, D)``). Each row must be convertible to ``str``.
        path: Destination file path.
        verbose: If ``True``, print progress messages before and after
            dumping.

    Examples:
        >>> import numpy as np
        >>> pts = np.array([[1.0, 2.0], [3.0, 4.0]])
        >>> dump_pts(pts, "out.pts", verbose=True)
        Dumping pts object to out.pts => Done.
    """
    if verbose:
        print(f"Dumping pts object to {path}", end="")
    with open(path, "w") as f:
        f.write("\n".join([" ".join(map(str, p)) for p in obj]))
    if verbose:
        print(" => Done.")


def dump_jsonlines(obj: Iterable[Any], path: str, verbose: bool = False) -> None:
    """Write an iterable of objects to a JSON Lines (``.jsonl``) file.

    Each element of ``obj`` is serialized as a single JSON object on its own
    line.

    Args:
        obj: An iterable of JSON-serializable objects.
        path: Destination file path.
        verbose: If ``True``, print progress messages before and after
            dumping.

    Examples:
        >>> records = [{"id": 1}, {"id": 2}]
        >>> dump_jsonlines(records, "data.jsonl", verbose=True)
        Dumping jsonlines file to data.jsonl => Done.
    """
    if verbose:
        print(f"Dumping jsonlines file to {path}", end="")
    with open(path, "w") as f:
        f.write("\n".join(json.dumps(line) for line in obj))
    if verbose:
        print(" => Done.")


def load_jsonlines(path: str, verbose: bool = False) -> List[Any]:
    """Load a JSON Lines (``.jsonl``) file into a list of Python objects.

    Each non-empty line is parsed as an independent JSON object.

    Args:
        path: Path to the JSON Lines file.
        verbose: If ``True``, print progress messages before and after
            loading.

    Returns:
        A ``list`` of deserialized Python objects, one per line.

    Examples:
        >>> records = load_jsonlines("data.jsonl")
        >>> len(records)
        2
    """
    if verbose:
        print(f"Loading jsonlines file from {path}", end="")
    # Iterate over the file object directly instead of f.readlines() to avoid
    # loading the entire file into memory before parsing.
    with open(path, "r") as f:
        data = [json.loads(l.strip()) for l in f if l.strip()]
    if verbose:
        print(" => Done.")
    return data
