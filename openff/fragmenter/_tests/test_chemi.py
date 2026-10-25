"""Test chemi module"""

import warnings

import numpy
import pytest
from openff.toolkit.topology import Molecule
from openff.toolkit.utils.exceptions import AtomMappingWarning, RadicalsNotSupportedError
from openff.units import unit

from openff.fragmenter import chemi
from openff.fragmenter._xtb import XTBError
from openff.fragmenter.chemi import (
    _extract_rd_fragment,
    _find_rd_stereocenters,
    assign_elf10_xtb_bond_orders,
    extract_fragment,
    find_ring_systems,
    find_stereocenters,
    smiles_to_molecule,
)


def test_assign_elf10_xtb_bond_orders():
    molecule = assign_elf10_xtb_bond_orders(Molecule.from_smiles("CCCC"))

    for bond in molecule.bonds:
        assert bond.fractional_bond_order is not None

        # Every bond in butane is a single bond.
        assert 0.9 < bond.fractional_bond_order < 1.1


def test_assign_elf10_xtb_bond_orders_aromatic():
    molecule = assign_elf10_xtb_bond_orders(Molecule.from_smiles("c1ccccc1"))

    carbon_carbon = [
        bond.fractional_bond_order
        for bond in molecule.bonds
        if bond.atom1.atomic_number == 6 and bond.atom2.atomic_number == 6
    ]

    assert len(carbon_carbon) == 6
    assert all(1.3 < value < 1.5 for value in carbon_carbon)
    # The WBOs come from single points on unminimized RDKit conformers, so the
    # six bonds are equal only to within the geometry noise.
    assert numpy.ptp(carbon_carbon) < 0.03


def test_assign_elf10_xtb_bond_orders_keeps_atom_map():
    molecule = smiles_to_molecule("CCO", add_atom_map=True)
    atom_map = dict(molecule.properties["atom_map"])

    molecule = assign_elf10_xtb_bond_orders(molecule, max_confs=1)

    assert molecule.properties["atom_map"] == atom_map


def test_assign_elf10_xtb_bond_orders_charged():
    molecule = assign_elf10_xtb_bond_orders(Molecule.from_smiles("C[NH3+]"), max_confs=1)

    assert all(bond.fractional_bond_order is not None for bond in molecule.bonds)


def test_assign_elf10_xtb_bond_orders_solvent():
    gas = assign_elf10_xtb_bond_orders(Molecule.from_smiles("CC(=O)[O-]"), max_confs=1)
    water = assign_elf10_xtb_bond_orders(Molecule.from_smiles("CC(=O)[O-]"), max_confs=1, solvent="water")

    gas_orders = numpy.array([bond.fractional_bond_order for bond in gas.bonds])
    water_orders = numpy.array([bond.fractional_bond_order for bond in water.bonds])

    assert numpy.allclose(gas_orders, water_orders, atol=0.1)
    assert not numpy.allclose(gas_orders, water_orders, atol=1.0e-4)


def test_assign_elf10_xtb_bond_orders_bad_solvent():
    with pytest.raises(XTBError, match="not parametrized"):
        assign_elf10_xtb_bond_orders(Molecule.from_smiles("C"), max_confs=1, solvent="not-a-solvent")


def test_assign_elf10_xtb_bond_orders_open_shell():
    # A methyl radical: built by hand as SMILES parsing would add a fourth H.
    molecule = Molecule()
    carbon = molecule.add_atom(6, 0, False)

    for _ in range(3):
        hydrogen = molecule.add_atom(1, 0, False)
        molecule.add_bond(carbon, hydrogen, 1, False)

    with pytest.raises(ValueError, match="odd number of electrons"):
        assign_elf10_xtb_bond_orders(molecule)


def test_assign_elf10_xtb_bond_orders_missing_bond(monkeypatch):
    monkeypatch.setattr(chemi, "compute_wiberg_bond_orders", lambda *args, **kwargs: {})

    with pytest.raises(XTBError, match="no Wiberg bond order"):
        assign_elf10_xtb_bond_orders(Molecule.from_smiles("C"), max_confs=1)


@pytest.mark.parametrize("smiles, max_confs", [("CCCCCCC", 1), ("CCCCCCC", 3)])
def test_generate_conformers(smiles, max_confs):
    returned_molecule = chemi._generate_conformers(Molecule.from_smiles(smiles), max_confs=max_confs)

    assert 0 < returned_molecule.n_conformers <= max_confs


def test_generate_conformers_ordering():
    original_molecule = Molecule.from_smiles("CCCC")

    returned_molecule = chemi._generate_conformers(original_molecule, max_confs=1)
    assert returned_molecule.n_conformers == 1

    # Make sure the atom ordering did not change.
    _, atom_map = Molecule.are_isomorphic(original_molecule, returned_molecule, return_atom_map=True)

    assert all(i == j for i, j in atom_map.items())


def test_generate_conformers_canonical_check():
    original_molecule = Molecule.from_smiles("CCCCl").canonical_order_atoms()
    original_molecule = chemi._generate_conformers(original_molecule, max_confs=1)

    # Generate a conformer using a molecule with permuted atom orderings.
    atom_map = numpy.arange(original_molecule.n_atoms)
    rng = numpy.random.default_rng(1337)
    rng.shuffle(atom_map)

    remapped_molecule = original_molecule.remap({int(i): int(j) for i, j in enumerate(atom_map)})
    remapped_molecule = chemi._generate_conformers(remapped_molecule, max_confs=1)

    original_conformer = original_molecule.conformers[0].m_as(unit.angstrom)
    remapped_conformer = remapped_molecule.conformers[0].m_as(unit.angstrom)

    assert numpy.allclose(original_conformer[:4], remapped_conformer[atom_map][:4])


@pytest.mark.parametrize(
    "smiles, expected_sizes",
    [
        ("c1ccccc1", [6]),
        ("C1CC2CCC1C2", [7]),
        ("C12C3C4C1C5C2C3C45", [8]),
        ("c1ccc2ccccc2c1", [10]),
        ("c1ccc(cc1)C2CCCC2", [5, 6]),
        ("c1ccc(cc1)C2CCC3C2CC3", [6, 7]),
        ("c1cc2ccc3cccc4c3c2c(c1)cc4", [16]),
        # Spiro rings share an atom and so form a single ring system.
        ("C1CCC2(CC1)CCCCC2", [11]),
        ("c1ccc(cc1)Cc2ccccc2", [6, 6]),
        ("CCCC", []),
    ],
)
def test_find_ring_systems(smiles, expected_sizes):
    molecule = Molecule.from_smiles(smiles, allow_undefined_stereo=True)

    ring_systems = find_ring_systems(molecule)

    assert all(molecule.atoms[index].is_in_ring() for index in ring_systems)
    assert sum(atom.is_in_ring() for atom in molecule.atoms) == len(ring_systems)

    sizes = sorted(list(ring_systems.values()).count(system) for system in set(ring_systems.values()))
    assert sizes == expected_sizes


@pytest.mark.parametrize("add_atom_map", [True, False])
def test_smiles_to_molecule(add_atom_map):
    molecule = smiles_to_molecule("CCCC", add_atom_map=add_atom_map)

    assert isinstance(molecule, Molecule)
    assert ("atom_map" in molecule.properties) == add_atom_map


@pytest.mark.parametrize(
    "smiles, expected_atoms, expected_bonds",
    [
        ("[C:1]([H:6])([H:7])([H:8])[C:5]([Br:3])([Cl:4])([H:2])", [4], []),
        ("[C:3]([Cl:1])([H:2])=[C:6]([H:4])([Cl:5])", [], [(2, 5)]),
        (
            "[C:3]([Cl:1])([H:2])=[C:6]([H:4])[C:5]([Br:7])([Cl:8])([H:9])",
            [4],
            [(2, 5)],
        ),
    ],
)
@pytest.mark.parametrize("find_method", [_find_rd_stereocenters, find_stereocenters])
def test_find_stereocenters(smiles, expected_atoms, expected_bonds, find_method):
    molecule = Molecule.from_mapped_smiles(smiles, allow_undefined_stereo=True)

    stereogenic_atoms, stereogenic_bonds = find_method(molecule)

    assert stereogenic_atoms == expected_atoms
    assert stereogenic_bonds == expected_bonds


@pytest.mark.parametrize(
    "smiles, atoms, bonds, expected",
    [
        (
            "[C:1]([H:4])([H:5])([H:6])[C:2]([H:7])([H:8])[C:3]([H:9])([H:10])([H:11])",
            {1, 2},
            {(1, 2)},
            "CC",
        ),
        (
            "[C:1]([H:4])([H:5])([H:6])[C:2]([H:7])([H:8])[C:3]([H:9])([H:10])([H:11])",
            {1, 2, 4},
            {(1, 2), (1, 4)},
            "CC",
        ),
        (
            r"[H:6]/[C:1](=[C:2](\[C:3]([H:7])([H:8])[H:9])/[Cl:5])/[Cl:4]",
            {1, 2, 4, 5},
            {(1, 2), (1, 4), (2, 5)},
            r"Cl\C=C/Cl",
        ),
        (
            "[H:7][C:1]([H:8])([C@:2]([F:3])([Cl:5])[Br:6])[Cl:4]",
            {1, 2, 3, 5, 6},
            {(1, 2), (2, 3), (2, 5), (2, 6)},
            r"C[C@](F)(Cl)Br",
        ),
    ],
)
@pytest.mark.parametrize("extract_method", [_extract_rd_fragment, extract_fragment])
def test_extract_fragment(smiles, atoms, bonds, expected, extract_method):
    molecule = Molecule.from_mapped_smiles(smiles)
    molecule.properties["atom_map"] = {i: i + 1 for i in range(molecule.n_atoms)}

    fragment = extract_method(molecule, atoms, bonds)

    expected_fragment = Molecule.from_smiles(expected)

    assert Molecule.are_isomorphic(fragment, expected_fragment, bond_stereochemistry_matching=False)[0]


def test_extract_fragment_bonds_in_atoms():
    """Tests that an exception is raised when the bonds set contains atoms not in the
    atoms set."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AtomMappingWarning)

        molecule = Molecule.from_smiles("[H:1][C:2]#[C:3][H:4]")

    with pytest.raises(ValueError, match="set includes atoms not in the"):
        extract_fragment(molecule, {2}, {(2, 3)})


def test_extract_fragment_disconnected_fragment_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AtomMappingWarning)

        molecule = Molecule.from_smiles("[C:1]([H:3])([H:4])([H:5])[C:2]([H:6])([H:7])([H:8])")

    # Dropping the C-C bond leaves two methyl radicals, which the RDKit path rejects
    # before the disconnected-fragment check is reached.
    with pytest.raises(RadicalsNotSupportedError):
        extract_fragment(molecule, {1, 2}, set())
