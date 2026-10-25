"""A thin wrapper around the ``xtb`` command-line program for computing GFN2-xTB
Wiberg bond orders.

``xtb`` is run as a subprocess in a scratch directory. A single point writes a
``wbo`` file next to its input, with one ``i j wbo`` line (1-based atom indices,
``i < j``) for every atom pair whose bond order clears a small cutoff (about 0.1).
"""

import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy

_XTB_EXECUTABLE = "xtb"


class XTBNotFoundError(Exception):
    """Raised when the ``xtb`` executable cannot be found.

    This deliberately does not subclass ``RuntimeError``: fragment growth treats a
    ``RuntimeError`` from a WBO calculation as "this fragment is not sensible yet,
    keep growing", and a missing program must not be mistaken for that.
    """


class XTBError(RuntimeError):
    """Raised when an ``xtb`` calculation fails or its output cannot be read."""


def find_xtb() -> str:
    """Returns the path to the ``xtb`` executable, raising ``XTBNotFoundError`` if it
    is not on the ``PATH``."""

    path = shutil.which(_XTB_EXECUTABLE)

    if path is None:
        raise XTBNotFoundError(
            "The `xtb` executable could not be found on the PATH. It is required to compute "
            "Wiberg bond orders and can be installed with `mamba install -c conda-forge xtb`."
        )

    return path


def xtb_version() -> str:
    """Returns the version of the ``xtb`` executable, e.g. ``'6.7.1 (edcfbbe)'``."""

    result = subprocess.run(
        [find_xtb(), "--version"],
        capture_output=True,
        text=True,
        timeout=60,
        env=_xtb_environment(),
    )

    match = re.search(r"xtb version\s+(\S+(?:\s+\([0-9a-f]+\))?)", result.stdout + result.stderr)

    if match is None:
        raise XTBError(f"Could not parse the xtb version from:\n{result.stdout[-2000:]}")

    return match.group(1)


def _xtb_environment() -> dict[str, str]:
    """The environment to run ``xtb`` in.

    xtb is OpenMP parallel. Default it to a single thread so that running many
    fragmentations side by side (e.g. under ``pytest-xdist``) does not oversubscribe
    the machine; an explicit ``OMP_NUM_THREADS`` set by the user still wins.
    """

    environment = os.environ.copy()
    environment.setdefault("OMP_NUM_THREADS", "1")
    environment.setdefault("MKL_NUM_THREADS", "1")
    environment.setdefault("OPENBLAS_NUM_THREADS", "1")

    return environment


def _write_xyz(symbols: Sequence[str], coordinates: numpy.ndarray, path: Path):
    lines = [str(len(symbols)), ""]
    lines.extend(f"{symbol:<2s} {x:20.12f} {y:20.12f} {z:20.12f}" for symbol, (x, y, z) in zip(symbols, coordinates))

    path.write_text("\n".join(lines) + "\n")


def parse_wbo_file(contents: str) -> dict[tuple[int, int], float]:
    """Parses the contents of the ``wbo`` file written by ``xtb``.

    Returns
    -------
        The bond order of each listed atom pair, keyed by **0-based** atom indices
        with the smaller index first.
    """

    bond_orders = {}

    for line in contents.splitlines():
        tokens = line.split()

        if len(tokens) == 0:
            continue

        if len(tokens) != 3:
            raise XTBError(f"Unexpected line in the xtb wbo file: {line!r}")

        index_a, index_b = int(tokens[0]) - 1, int(tokens[1]) - 1
        bond_orders[(min(index_a, index_b), max(index_a, index_b))] = float(tokens[2])

    return bond_orders


def compute_wiberg_bond_orders(
    symbols: Sequence[str],
    coordinates: numpy.ndarray,
    charge: int,
    n_unpaired: int = 0,
    solvent: str | None = None,
    timeout: float = 3600.0,
) -> dict[tuple[int, int], float]:
    """Computes GFN2-xTB Wiberg bond orders for a single geometry.

    Parameters
    ----------
    symbols
        The element symbol of each atom.
    coordinates
        The coordinates [Angstrom] of each atom with shape ``(n_atoms, 3)``.
    charge
        The total charge of the molecule.
    n_unpaired
        The number of unpaired electrons.
    solvent
        The name of an ALPB implicit solvent known to xtb (e.g. ``'water'``), or
        ``None`` for the gas phase.
    timeout
        The maximum time [s] to let ``xtb`` run for.

    Returns
    -------
        The bond order of each atom pair that ``xtb`` reports, keyed by 0-based atom
        indices with the smaller index first. Pairs below xtb's print cutoff are
        absent.
    """

    coordinates = numpy.asarray(coordinates, dtype=float)

    if coordinates.shape != (len(symbols), 3):
        raise ValueError(f"Expected coordinates with shape ({len(symbols)}, 3), got {coordinates.shape}.")

    executable = find_xtb()

    with tempfile.TemporaryDirectory(prefix="openff_fragmenter_xtb_") as directory:
        work_dir = Path(directory)
        _write_xyz(symbols, coordinates, work_dir / "input.xyz")

        command = [
            executable,
            "input.xyz",
            "--sp",
            "--gfn",
            "2",
            "--chrg",
            str(charge),
            "--uhf",
            str(n_unpaired),
        ]

        if solvent is not None:
            command.extend(["--alpb", solvent])

        try:
            result = subprocess.run(
                command,
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=_xtb_environment(),
            )
        except subprocess.TimeoutExpired as error:
            raise XTBError(f"xtb did not finish within {timeout} s.") from error

        if result.returncode != 0:
            # xtb reports why it stopped on stdout, in the lines after '[ERROR]'.
            raise XTBError(
                f"xtb failed (exit code {result.returncode}) running `{' '.join(command[1:])}`:\n"
                f"{result.stdout[-2000:]}\n{result.stderr[-1000:]}"
            )

        wbo_path = work_dir / "wbo"

        if not wbo_path.exists():
            raise XTBError(f"xtb did not write a wbo file:\n{result.stdout[-2000:]}")

        return parse_wbo_file(wbo_path.read_text())
