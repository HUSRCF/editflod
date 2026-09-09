import json

import numpy as np

from ospedit.data import json_safe


def test_json_safe_converts_nonfinite_values_to_null():
    payload = json_safe({"nan": float("nan"), "inf": float("inf"), "nested": [1.0, float("-inf")]})
    encoded = json.dumps(payload, allow_nan=False)
    assert "NaN" not in encoded
    assert json.loads(encoded) == {"nan": None, "inf": None, "nested": [1.0, None]}


def test_json_safe_converts_numpy_scalars_and_arrays():
    payload = json_safe({"flag": np.bool_(True), "values": np.array([np.float32(1.5), np.float32(np.nan)])})
    assert payload == {"flag": True, "values": [1.5, None]}


def test_json_safe_converts_numpy_dict_keys():
    payload = json_safe({np.int64(7): np.float32(1.25)})
    assert json.dumps(payload, allow_nan=False) == '{"7": 1.25}'


def test_json_safe_converts_sets_deterministically():
    payload = json_safe({"values": {"beta", "alpha"}})
    assert payload == {"values": ["alpha", "beta"]}
