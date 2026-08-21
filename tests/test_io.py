"""Tests for wtools.utils.io -- round-trip tests for every load/dump pair and the LMDB class."""

import json as _json
from pathlib import Path

import numpy as np
import pytest

from wtools.utils.io import (
    LMDB,
    MissingOk,
    dump_json,
    dump_jsonlines,
    dump_pickle,
    dump_pts,
    load_json,
    load_jsonlines,
    load_pickle,
    load_pts,
    load_yaml,
    remove_lmdbm,
)


# ---------------------------------------------------------------------------
# Pickle round-trip
# ---------------------------------------------------------------------------
class TestPickleRoundTrip:
    def test_dict_roundtrip(self, tmp_path):
        data = {"a": 1, "b": [2, 3], "c": (4, 5)}
        path = str(tmp_path / "data.pkl")
        dump_pickle(data, path)
        loaded = load_pickle(path)
        assert loaded == data

    def test_numpy_array_roundtrip(self, tmp_path):
        data = np.random.rand(10, 3).astype(np.float32)
        path = str(tmp_path / "array.pkl")
        dump_pickle(data, path)
        loaded = load_pickle(path)
        np.testing.assert_array_equal(loaded, data)

    def test_nested_structure_roundtrip(self, tmp_path):
        data = {"list": [1, 2, {"inner": "value"}], "tuple": (1, 2), "none": None}
        path = str(tmp_path / "nested.pkl")
        dump_pickle(data, path)
        loaded = load_pickle(path)
        assert loaded == data


# ---------------------------------------------------------------------------
# JSON round-trip
# ---------------------------------------------------------------------------
class TestJsonRoundTrip:
    def test_dict_roundtrip(self, tmp_path):
        data = {"name": "test", "value": 42, "list": [1, 2, 3]}
        path = str(tmp_path / "data.json")
        dump_json(data, path)
        loaded = load_json(path)
        assert loaded == data

    def test_list_roundtrip(self, tmp_path):
        data = [1, "two", {"three": 3}, [4, 5]]
        path = str(tmp_path / "list.json")
        dump_json(data, path)
        loaded = load_json(path)
        assert loaded == data


# ---------------------------------------------------------------------------
# YAML round-trip
# ---------------------------------------------------------------------------
class TestYamlRoundTrip:
    def test_dict_roundtrip(self, tmp_path):
        import yaml

        data = {"name": "test", "value": 42, "nested": {"a": 1, "b": 2}}
        path = str(tmp_path / "data.yaml")
        # yaml doesn't have a dump helper in io.py, so write manually
        with open(path, "w") as f:
            yaml.dump(data, f)
        loaded = load_yaml(path)
        assert loaded == data

    def test_list_roundtrip(self, tmp_path):
        import yaml

        data = [{"a": 1}, {"b": 2}, {"c": 3}]
        path = str(tmp_path / "list.yaml")
        with open(path, "w") as f:
            yaml.dump(data, f)
        loaded = load_yaml(path)
        assert loaded == data


# ---------------------------------------------------------------------------
# PTS round-trip
# ---------------------------------------------------------------------------
class TestPtsRoundTrip:
    def test_2d_points_roundtrip(self, tmp_path):
        data = np.array([[1.0, 2.0], [3.5, 4.2], [5.1, 6.9]], dtype=np.float32)
        path = str(tmp_path / "pts.pts")
        dump_pts(data, path)
        loaded = load_pts(path)
        np.testing.assert_array_equal(loaded, data)

    def test_3d_points_roundtrip(self, tmp_path):
        data = np.array([[1.0, 2.0, 3.0], [4.5, 5.5, 6.5]], dtype=np.float32)
        path = str(tmp_path / "pts3d.pts")
        dump_pts(data, path)
        loaded = load_pts(path)
        np.testing.assert_array_equal(loaded, data)


# ---------------------------------------------------------------------------
# JSON Lines round-trip
# ---------------------------------------------------------------------------
class TestJsonLinesRoundTrip:
    def test_roundtrip(self, tmp_path):
        data = [
            {"id": 1, "name": "alice"},
            {"id": 2, "name": "bob"},
            {"id": 3, "name": "charlie"},
        ]
        path = str(tmp_path / "data.jsonl")
        dump_jsonlines(data, path)
        loaded = load_jsonlines(path)
        assert loaded == data

    def test_empty_lines_handled(self, tmp_path):
        data = [{"a": 1}]
        path = str(tmp_path / "single.jsonl")
        dump_jsonlines(data, path)
        loaded = load_jsonlines(path)
        assert loaded == data


# ---------------------------------------------------------------------------
# LMDB class tests
# ---------------------------------------------------------------------------
class TestLMDB:
    def test_create_and_write_read(self, tmp_path):
        db_path = str(tmp_path / "lmdb_db")
        with LMDB(db_path, flag="c") as db:
            db["key1"] = b"value1"
            db["key2"] = b"value2"
            assert db["key1"] == b"value1"
            assert db["key2"] == b"value2"

    def test_len(self, tmp_path):
        db_path = str(tmp_path / "lmdb_db")
        with LMDB(db_path, flag="c") as db:
            db["a"] = b"1"
            db["b"] = b"2"
            db["c"] = b"3"
            assert len(db) == 3

    def test_delete(self, tmp_path):
        db_path = str(tmp_path / "lmdb_db")
        with LMDB(db_path, flag="c") as db:
            db["a"] = b"1"
            db["b"] = b"2"
            del db["a"]
            assert len(db) == 1
            assert db["b"] == b"2"
            with pytest.raises(KeyError):
                _ = db["a"]

    def test_get_method(self, tmp_path):
        db_path = str(tmp_path / "lmdb_db")
        with LMDB(db_path, flag="c") as db:
            db["present"] = b"yes"
            assert db.get("present") == b"yes"
            assert db.get("missing") is None
            assert db.get("missing", b"default") == b"default"

    def test_iteration_keys(self, tmp_path):
        db_path = str(tmp_path / "lmdb_db")
        with LMDB(db_path, flag="c") as db:
            db["x"] = b"1"
            db["y"] = b"2"
            db["z"] = b"3"
            keys = list(db.keys())
            assert set(keys) == {b"x", b"y", b"z"}

    def test_iteration_items(self, tmp_path):
        db_path = str(tmp_path / "lmdb_db")
        with LMDB(db_path, flag="c") as db:
            db["a"] = b"1"
            db["b"] = b"2"
            pairs = list(db.items())
            assert len(pairs) == 2
            assert (b"a", b"1") in pairs
            assert (b"b", b"2") in pairs

    def test_iteration_values(self, tmp_path):
        db_path = str(tmp_path / "lmdb_db")
        with LMDB(db_path, flag="c") as db:
            db["a"] = b"1"
            db["b"] = b"2"
            vals = list(db.values())
            assert set(vals) == {b"1", b"2"}

    def test_contains(self, tmp_path):
        db_path = str(tmp_path / "lmdb_db")
        with LMDB(db_path, flag="c") as db:
            db["exists"] = b"yes"
            assert b"exists" in db
            assert b"missing" not in db

    def test_update_single_pair(self, tmp_path):
        db_path = str(tmp_path / "lmdb_db")
        with LMDB(db_path, flag="c") as db:
            db.update("key", b"value")
            assert db["key"] == b"value"

    def test_update_dict(self, tmp_path):
        db_path = str(tmp_path / "lmdb_db")
        with LMDB(db_path, flag="c") as db:
            db.update({"a": b"1", "b": b"2"})
            assert db["a"] == b"1"
            assert db["b"] == b"2"
            assert len(db) == 2

    def test_str_keys_and_values(self, tmp_path):
        db_path = str(tmp_path / "lmdb_db")
        with LMDB(db_path, flag="c") as db:
            db["str_key"] = b"str_value"
            assert db["str_key"] == b"str_value"

    def test_flag_r_readonly(self, tmp_path):
        db_path = str(tmp_path / "lmdb_db")
        # First create and populate
        with LMDB(db_path, flag="c") as db:
            db["key"] = b"value"
        # Then open read-only
        with LMDB(db_path, flag="r") as db:
            assert db["key"] == b"value"

    def test_flag_w_readwrite(self, tmp_path):
        db_path = str(tmp_path / "lmdb_db")
        with LMDB(db_path, flag="c") as db:
            db["k1"] = b"v1"
        with LMDB(db_path, flag="w") as db:
            db["k2"] = b"v2"
            assert db["k1"] == b"v1"
            assert db["k2"] == b"v2"

    def test_flag_n_recreates(self, tmp_path):
        db_path = str(tmp_path / "lmdb_db")
        with LMDB(db_path, flag="c") as db:
            db["old"] = b"data"
        # flag="n" should wipe and create new
        with LMDB(db_path, flag="n") as db:
            db["new"] = b"data"
            assert len(db) == 1
            assert db["new"] == b"data"
            with pytest.raises(KeyError):
                _ = db["old"]

    def test_invalid_flag_raises(self, tmp_path):
        db_path = str(tmp_path / "lmdb_db")
        with pytest.raises(ValueError, match="Invalid flag"):
            LMDB(db_path, flag="x")

    def test_context_manager_closes(self, tmp_path):
        db_path = str(tmp_path / "lmdb_db")
        db = LMDB(db_path, flag="c")
        db["k"] = b"v"
        db.__exit__(None, None, None)
        # After close, using db should fail
        with pytest.raises(Exception):
            _ = db["k"]

    def test_map_size_property(self, tmp_path):
        db_path = str(tmp_path / "lmdb_db")
        with LMDB(db_path, flag="c", map_size=1 << 20) as db:
            assert db.map_size == 1 << 20
            db.map_size = 2 << 20
            assert db.map_size == 2 << 20

    def test_sync(self, tmp_path):
        db_path = str(tmp_path / "lmdb_db")
        with LMDB(db_path, flag="c") as db:
            db["k"] = b"v"
            db.sync()  # should not raise

    def test_keyerror_on_missing(self, tmp_path):
        db_path = str(tmp_path / "lmdb_db")
        with LMDB(db_path, flag="c") as db:
            with pytest.raises(KeyError):
                _ = db["nonexistent"]

    def test_pre_value_invalid_type(self, tmp_path):
        db_path = str(tmp_path / "lmdb_db")
        with LMDB(db_path, flag="c") as db:
            with pytest.raises(TypeError):
                db["key"] = 12345  # type: ignore  # only str/bytes are valid

    def test_pre_key_invalid_type(self, tmp_path):
        db_path = str(tmp_path / "lmdb_db")
        with LMDB(db_path, flag="c") as db:
            with pytest.raises(TypeError):
                _ = db[12345]  # type: ignore  # only str/bytes keys are valid by _pre_key


# ---------------------------------------------------------------------------
# MissingOk / remove_lmdbm tests
# ---------------------------------------------------------------------------
class TestMissingOk:
    def test_missing_ok_suppresses(self):
        with MissingOk(True):
            Path("/nonexistent/path/that/does/not/exist").unlink()

    def test_missing_ok_false_raises(self):
        with pytest.raises(FileNotFoundError):
            with MissingOk(False):
                Path("/nonexistent/path/that/does/not/exist").unlink()

    def test_remove_lmdbm_missing_ok(self, tmp_path):
        # Should not raise even if the directory doesn't exist
        remove_lmdbm(str(tmp_path / "nonexistent_db"), missing_ok=True)

    def test_remove_lmdbm_existing(self, tmp_path):
        db_path = str(tmp_path / "lmdb_db")
        with LMDB(db_path, flag="c") as db:
            db["k"] = b"v"
        remove_lmdbm(db_path, missing_ok=True)
        # The directory should no longer exist
        assert not Path(db_path).exists()
