"""Tests for lume_pva.runner configuration generation.

Runner.__init__ starts PVA/CA servers, so these tests only exercise the pure
configuration logic (Runner.generate_config) using a stub model object — no
servers are started and no network calls are made.
"""

import numpy as np
import pytest
from lume.variables import NDVariable, ScalarVariable, Variable

from lume_pva.runner import Runner


class StubModel:
    """Minimal stand-in for a LUMEModel: generate_config only reads
    supported_variables."""

    def __init__(self, variables: dict[str, Variable]) -> None:
        self.supported_variables = variables


@pytest.fixture
def model() -> StubModel:
    return StubModel(
        {
            "input_a": ScalarVariable(name="input_a"),
            "output_b": ScalarVariable(name="output_b", read_only=True),
            "image": NDVariable(
                name="image", shape=(4, 4), dtype=np.float64, read_only=True
            ),
        }
    )


def test_runner_defaults(model: StubModel) -> None:
    config = Runner.generate_config(model)

    for name, var_config in config["variables"].items():
        assert var_config["name"] == name
        assert var_config["pv"] == name

    assert set(config["variables"].keys()) == {"input_a", "output_b", "image"}
    # only one read write
    assert config["variables"]["input_a"]["mode"] == "rw"
    assert config["variables"]["output_b"]["mode"] == "ro"
    assert config["variables"]["image"]["mode"] == "ro"

    # continuous mode is default
    assert config["remote_model_mode"] == "continuous"

    # No prefix
    assert config["prefix"] == ""


def test_set_prefix(model: StubModel) -> None:
    config = Runner.generate_config(model, prefix="TEST:")

    assert config["prefix"] == "TEST:"


def test_mark_rw_variables_ro_remote(model: StubModel) -> None:
    config = Runner.generate_config(model, remote_inputs=True)

    assert config["variables"]["input_a"]["mode"] == "remote"
    # Read-only variables stay served by the runner
    assert config["variables"]["output_b"]["mode"] == "ro"


def test_pv_name_transformer(model: StubModel) -> None:
    config = Runner.generate_config(
        model, name_transformer=lambda var, name: f"XFORM:{name.upper()}"
    )
    assert config["variables"]["input_a"]["pv"] == "XFORM:INPUT_A"
    # Variable names must remain untouched — only the PV name changes
    assert config["variables"]["input_a"]["name"] == "input_a"


def test_no_variables() -> None:
    empty_model = StubModel({})
    config = Runner.generate_config(empty_model)

    assert config["variables"] == {}


# ---------------------------------------------------------------------------
# Tests for None variable value retrieval during simulation execution
# ---------------------------------------------------------------------------


class SimModel:
    """Model stub that simulates get() returning None for some variables,
    mimicking a simulation where certain outputs are not computed."""

    def __init__(self, variables: dict[str, Variable], outputs: dict[str, object]):
        self.supported_variables = variables
        self._outputs = outputs

    def set(self, values: dict[str, object]) -> None:
        pass

    def get(self, variables: dict[str, Variable]) -> dict[str, object]:
        return {k: self._outputs.get(k) for k in variables}


@pytest.mark.parametrize(
    ("variable", "expected_default"),
    [
        pytest.param(
            ScalarVariable(name="x", default_value=5.0),
            5.0,
            id="scalar_with_default",
        ),
        pytest.param(
            ScalarVariable(name="x"),
            0.0,
            id="scalar_no_default",
        ),
        pytest.param(
            NDVariable(name="img", shape=(2, 3), dtype=np.float64, default_value=np.ones((2, 3))),
            np.ones((2, 3)),
            id="ndarray_with_default",
        ),
        pytest.param(
            NDVariable(name="img", shape=(2, 3), dtype=np.float64),
            np.zeros((2, 3), dtype=np.float64),
            id="ndarray_no_default",
        ),
    ],
)
def test_none_value_from_model_get_uses_default(variable, expected_default):
    """Verify that when model.get() returns None for a variable (e.g. the
    simulation did not compute that output), pack_value falls back to the
    variable's default value — matching what Runner._generate_value does."""
    from lume_pva.variables import find_variable_handler

    handler = find_variable_handler(type(variable))
    assert handler is not None

    type_ = handler.create_type(variable)

    # Simulate the Runner._generate_value path: value comes in as None
    packed = handler.pack_value(variable, type_, None)
    unpacked = handler.unpack_value(variable, packed)

    if isinstance(expected_default, np.ndarray):
        np.testing.assert_array_equal(unpacked, expected_default)
    else:
        assert unpacked == expected_default


def test_none_values_during_simulation_execution():
    """End-to-end test: model.get() returns None for some outputs.
    Verify that the handler pipeline (as used by Runner._generate_value)
    produces valid packed values with defaults instead of raising."""
    from lume_pva.variables import find_variable_handler

    variables = {
        "input_x": ScalarVariable(name="input_x", default_value=1.0),
        "output_y": ScalarVariable(name="output_y", default_value=42.0, read_only=True),
        "output_img": NDVariable(
            name="output_img", shape=(4, 4), dtype=np.float64, read_only=True
        ),
    }

    # Model returns None for output_y and output_img (simulation didn't compute them)
    sim = SimModel(variables, outputs={"input_x": 1.0, "output_y": None, "output_img": None})

    out_values = sim.get(variables)

    # Walk through each output the way Runner._run does
    for var_name, value in out_values.items():
        var = variables[var_name]
        handler = find_variable_handler(type(var))
        assert handler is not None

        type_ = handler.create_type(var)

        # This must not raise — None must be handled gracefully
        packed = handler.pack_value(var, type_, value)
        unpacked = handler.unpack_value(var, packed)

        # None values should resolve to defaults
        if value is None:
            expected = handler.default_value(var)
            if isinstance(expected, np.ndarray):
                np.testing.assert_array_equal(unpacked, expected)
            else:
                assert unpacked == expected
        else:
            assert unpacked == value
