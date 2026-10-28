"""Tests for the xtb wrapper."""

import logging
import re
import subprocess

import numpy
import pytest

from openff.fragmenter import _xtb
from openff.fragmenter._xtb import (
    XTBError,
    XTBNotFoundError,
    compute_wiberg_bond_orders,
    find_xtb,
    parse_wbo_file,
    xtb_version,
)

ETHANE_SYMBOLS = ["C", "C", "H", "H", "H", "H", "H", "H"]
ETHANE_COORDINATES = numpy.array(
    [
        [0.000, 0.000, 0.762],
        [0.000, 0.000, -0.762],
        [1.018, 0.000, 1.155],
        [-0.509, 0.882, 1.155],
        [-0.509, -0.882, 1.155],
        [-1.018, 0.000, -1.155],
        [0.509, -0.882, -1.155],
        [0.509, 0.882, -1.155],
    ]
)


def test_parse_wbo_file():
    contents = (
        "           1           2   1.0311837093841154     \n           3           1  0.98893631125628267     \n\n"
    )

    assert parse_wbo_file(contents) == {(0, 1): 1.0311837093841154, (0, 2): 0.98893631125628267}


def test_parse_wbo_file_bad_line():
    with pytest.raises(XTBError, match="Unexpected line"):
        parse_wbo_file("1 2\n")


def test_find_xtb_missing(monkeypatch):
    monkeypatch.setattr(_xtb.shutil, "which", lambda *_: None)

    with pytest.raises(XTBNotFoundError, match="could not be found"):
        find_xtb()


def test_xtb_version():
    assert re.match(r"\d+\.\d+\.\d+", xtb_version())


def test_compute_wiberg_bond_orders():
    bond_orders = compute_wiberg_bond_orders(ETHANE_SYMBOLS, ETHANE_COORDINATES, charge=0)

    # The seven bonds of ethane, and no non-bonded pairs.
    assert set(bond_orders) == {(0, 1), (0, 2), (0, 3), (0, 4), (1, 5), (1, 6), (1, 7)}
    assert bond_orders[(0, 1)] == pytest.approx(1.03, abs=0.02)
    assert all(bond_orders[pair] == pytest.approx(0.99, abs=0.02) for pair in bond_orders if pair != (0, 1))


def test_compute_wiberg_bond_orders_solvent():
    gas = compute_wiberg_bond_orders(ETHANE_SYMBOLS, ETHANE_COORDINATES, charge=0)
    water = compute_wiberg_bond_orders(ETHANE_SYMBOLS, ETHANE_COORDINATES, charge=0, solvent="water")

    assert set(gas) == set(water)
    assert gas != water


@pytest.mark.parametrize("solvent", [None, "water"])
def test_compute_wiberg_bond_orders_optimized(solvent, caplog):
    # Stretch the C-C bond so the optimization has real work to do.
    coordinates = ETHANE_COORDINATES.copy()
    coordinates[[0, 2, 3, 4], 2] += 0.3

    with caplog.at_level(logging.WARNING):
        single_point = compute_wiberg_bond_orders(ETHANE_SYMBOLS, coordinates, charge=0, solvent=solvent)
        optimized = compute_wiberg_bond_orders(
            ETHANE_SYMBOLS, coordinates, charge=0, solvent=solvent, optimization_level="normal"
        )

    assert "did not converge" not in caplog.text

    assert set(optimized) == set(single_point)
    # The stretched bond is weaker than the relaxed one.
    assert single_point[(0, 1)] < optimized[(0, 1)] - 0.02
    assert optimized[(0, 1)] == pytest.approx(1.03, abs=0.02)


def test_compute_wiberg_bond_orders_unconverged(monkeypatch, caplog):
    run = subprocess.run

    def run_one_cycle(command, **kwargs):
        return run([*command, "--cycles", "1"], **kwargs)

    monkeypatch.setattr(_xtb.subprocess, "run", run_one_cycle)

    coordinates = ETHANE_COORDINATES.copy()
    coordinates[[0, 2, 3, 4], 2] += 0.3

    with caplog.at_level(logging.WARNING):
        bond_orders = compute_wiberg_bond_orders(ETHANE_SYMBOLS, coordinates, charge=0, optimization_level="normal")

    assert "did not converge" in caplog.text
    assert len(bond_orders) == 7


def test_compute_wiberg_bond_orders_failure():
    with pytest.raises(XTBError, match="not parametrized"):
        compute_wiberg_bond_orders(ETHANE_SYMBOLS, ETHANE_COORDINATES, charge=0, solvent="not-a-solvent")


def test_compute_wiberg_bond_orders_bad_shape():
    with pytest.raises(ValueError, match="Expected coordinates"):
        compute_wiberg_bond_orders(ETHANE_SYMBOLS, ETHANE_COORDINATES[:-1], charge=0)
