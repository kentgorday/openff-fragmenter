"""Functions for computing bond orders of, and manipulating, OpenFF molecules."""

import copy
import logging

import networkx
import numpy
from openff.toolkit.topology import Molecule
from openff.toolkit.utils import RDKitToolkitWrapper
from openff.units import unit

from openff.fragmenter._xtb import OptimizationLevel, XTBError, compute_wiberg_bond_orders
from openff.fragmenter.utils import get_atom_index, get_map_index

logger = logging.getLogger(__name__)


def assign_elf10_xtb_bond_orders(
    molecule: Molecule,
    max_confs: int = 800,
    rms_threshold: float = 1.0,
    solvent: str | None = None,
    optimization_level: OptimizationLevel | None = None,
) -> Molecule:
    """Generate ELF10 GFN2-xTB Wiberg bond orders (WBOs) for a molecule.

    Conformers are generated with the RDKit and the ELF10 subset of them is selected.
    Each selected conformer is optionally optimized with GFN2-xTB, a GFN2-xTB WBO
    calculation is run on it with the ``xtb`` program, and the WBO of each bond is
    averaged over those conformers.

    Parameters
    ----------
    molecule
        The molecule to compute WBOs for.
    max_confs
        Max number of conformers to generate.
    rms_threshold
        The minimum RMS value [Angstrom] at which two conformers are considered redundant
        and one is deleted.
    solvent
        The name of an ALPB implicit solvent known to ``xtb`` (e.g. ``'water'``) to
        compute the WBOs in, or ``None`` to compute them in the gas phase.
    optimization_level
        If not ``None``, each ELF10 conformer is optimized with ``xtb --opt`` at
        this level (e.g. ``'normal'``) before its WBOs are taken. Otherwise the WBOs
        are computed for the unminimized RDKit conformers.

    Returns
    -------
        A new molecule which contains ELF10 GFN2-xTB Wiberg bond orders.
    """

    molecule = copy.deepcopy(molecule)

    # Store the atom map separately in case it gets removed by a TK.
    atom_map = molecule.properties.pop("atom_map", None)

    charge = int(numpy.rint(molecule.total_charge.m_as(unit.elementary_charge)))
    n_electrons = sum(atom.atomic_number for atom in molecule.atoms) - charge

    if n_electrons % 2 != 0:
        raise ValueError(
            f"{molecule.to_smiles()} has an odd number of electrons. Only closed-shell molecules are supported."
        )

    # Generate a set of ELF10 conformers. The RDKit implementation of ELF ranks the
    # conformers using MMFF94 charges, so no semi-empirical charges are needed.
    molecule = _generate_conformers(molecule, max_confs, rms_threshold)
    molecule.apply_elf_conformer_selection(toolkit_registry=RDKitToolkitWrapper())

    symbols = [atom.symbol for atom in molecule.atoms]
    bond_keys = [tuple(sorted((bond.atom1_index, bond.atom2_index))) for bond in molecule.bonds]

    per_conformer_bond_orders = []

    for conformer in molecule.conformers:
        bond_orders = compute_wiberg_bond_orders(
            symbols,
            conformer.m_as(unit.angstrom),
            charge=charge,
            solvent=solvent,
            optimization_level=optimization_level,
        )

        # xtb only lists pairs above a small cutoff (~0.1), which every real bond in
        # a closed-shell molecule clears. A missing bond means the calculation went
        # wrong, so treat it as a failure rather than as a zero bond order.
        missing = [key for key in bond_keys if key not in bond_orders]

        if missing:
            raise XTBError(f"xtb reported no Wiberg bond order for bonds {missing} of {molecule.to_smiles()}.")

        per_conformer_bond_orders.append([bond_orders[key] for key in bond_keys])

    bond_orders = [*numpy.mean(per_conformer_bond_orders, axis=0)]

    for bond, bond_order in zip(molecule.bonds, bond_orders):
        bond.fractional_bond_order = float(bond_order)

    # Restore the atom map.
    if atom_map is not None:
        molecule.properties["atom_map"] = atom_map

    return molecule


def _generate_conformers(molecule: Molecule, max_confs: int = 800, rms_threshold: float = 1.0) -> Molecule:
    """Generate conformations for the supplied molecule.

    Parameters
    ----------
    molecule
        Molecule for which to generate conformers
    max_confs
        Max number of conformers to generate.
    rms_threshold
        The minimum RMS value [Angstrom] at which two conformers are considered redundant
        and one is deleted.

    Returns
    -------
        A new multi-conformer molecule with up to max_confs conformers.
    """

    molecule = copy.deepcopy(molecule)

    # Store the atom map separately in case it gets removed / mangled by a TK.
    atom_map = molecule.properties.pop("atom_map", None)

    # Canonically order the atoms in the molecule before generating the conformer.
    # This helps ensure the same conformers are generated for the same molecules
    # independently of their atom order.
    canonical_molecule = molecule.canonical_order_atoms()

    canonical_molecule.generate_conformers(n_conformers=max_confs, rms_cutoff=rms_threshold * unit.angstrom)

    _, canonical_map = Molecule.are_isomorphic(canonical_molecule, molecule, return_atom_map=True)

    molecule = canonical_molecule.remap(canonical_map)

    # Restore the atom map.
    if atom_map is not None:
        molecule.properties["atom_map"] = atom_map

    return molecule


def find_ring_systems(molecule: Molecule) -> dict[int, int]:
    """This function attempts to find all ring systems (see [1] for more details) in
    a given molecule.

    The method first attempts to determine which atoms and bonds are part of rings
    by matching the `[*:1]@[*:2]` SMIRKS pattern against the molecule.

    The matched bonds are then used to construct a graph (using ``networkx``), from which
    the ring systems are identified as those sets of atoms which are 'connected'
    together (using ``connected_components``) by at least one path.

    Parameters
    ----------
    molecule:
        The molecule to search for ring systems.

    Notes
    -----
    * Two molecular rings with only one common atom (i.e. spiro compounds) are
      considered to be part of the same ring system.

    References
    ----------
    [1] `Ring Perception <https://docs.eyesopen.com/toolkits/python/oechemtk/ring.html>`_

    Returns
    -------
        The index of which ring system each atom belongs to. Only ring atoms are
        included in the returned dictionary.
    """

    # Find the ring atoms
    ring_atom_index_pairs = {tuple(sorted(pair)) for pair in molecule.chemical_environment_matches("[*:1]@[*:2]")}

    # Construct a networkx graph from the found ring bonds.
    graph = networkx.Graph()

    for atom_index_pair in ring_atom_index_pairs:
        graph.add_edge(*atom_index_pair)

    ring_systems = {}

    for i, ring_system in enumerate(networkx.connected_components(graph)):
        for atom_index in ring_system:
            assert atom_index not in ring_systems
            ring_systems[atom_index] = i + 1

    return ring_systems


def _find_rd_stereocenters(
    molecule: Molecule,
) -> tuple[list[int], list[tuple[int, int]]]:
    """A method which returns the of the stereogenic atom and bonds of a molecule
    using the RDKit.

    Parameters
    ----------
    molecule
        The molecule whose stereocenters should be returned.

    Notes
    -----
    * This method currently deals with RDKit directly and should be removed once
      the API points suggested in openff-toolkit issue #903 have been added.

    Returns
    -------
        The indices of the stereogenic atoms in the molecule and the indices of the
        atoms involved in stereogenic bonds in the molecule.
    """

    from rdkit import Chem

    rd_molecule = molecule.to_rdkit()

    Chem.AssignStereochemistry(rd_molecule, cleanIt=True, force=True, flagPossibleStereoCenters=True)

    stereogenic_atoms = {atom.GetIdx() for atom in rd_molecule.GetAtoms() if atom.HasProp("_ChiralityPossible")}

    # Clear any previous assignments on the bonds, since FindPotentialStereo
    # may not overwrite it
    for bond in rd_molecule.GetBonds():
        bond.SetStereo(Chem.BondStereo.STEREONONE)

    Chem.FindPotentialStereoBonds(rd_molecule, cleanIt=True)

    stereogenic_bonds = {
        (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        for bond in rd_molecule.GetBonds()
        if bond.GetStereo() == Chem.BondStereo.STEREOANY
    }

    return sorted(stereogenic_atoms), sorted(stereogenic_bonds)


def find_stereocenters(molecule: Molecule) -> tuple[list[int], list[tuple[int, int]]]:
    """A method which returns the of the stereogenic atom and bonds of a molecule.

    Parameters
    ----------
    molecule
        The molecule whose stereocenters should be returned.

    Returns
    -------
        The indices of the stereogenic atoms in the molecule and the indices of the
        atoms involved in stereogenic bonds in the molecule.
    """

    return _find_rd_stereocenters(molecule)


def _extract_rd_fragment(molecule: Molecule, atom_indices: set[int], bond_indices: set[tuple[int, int]]) -> Molecule:
    from rdkit import Chem

    rd_molecule = Chem.RWMol(molecule.to_rdkit())
    rd_atoms_by_map: dict[int, Chem.Atom] = {}

    # Restore the map indices as to_rdkit does not automatically add them.
    for atom in rd_molecule.GetAtoms():
        atom.SetAtomMapNum(get_map_index(molecule, atom.GetIdx()))

        rd_atoms_by_map[atom.GetAtomMapNum()] = atom

    atoms_to_use = [get_atom_index(molecule, i) for i in atom_indices]
    bonds_to_use = [
        rd_molecule.GetBondBetweenAtoms(get_atom_index(molecule, pair[0]), get_atom_index(molecule, pair[1])).GetIdx()
        for pair in bond_indices
    ]

    # Make sure to include any Hs bonded to the included atom set otherwise radicals
    # will form.
    for map_index in atom_indices:
        for neighbour in rd_atoms_by_map[map_index].GetNeighbors():
            if (
                neighbour.GetAtomicNum() != 1
                or neighbour.GetAtomMapNum() < 1
                or neighbour.GetAtomMapNum() in atom_indices
            ):
                continue

            atoms_to_use.append(neighbour.GetIdx())
            bonds_to_use.append(
                rd_molecule.GetBondBetweenAtoms(rd_atoms_by_map[map_index].GetIdx(), neighbour.GetIdx()).GetIdx()
            )

    # Add additional hydrogens to atoms where the total valence will change likewise to
    # ensure the valence does not change.
    rd_atoms_by_index = {atom.GetIdx(): atom for atom in rd_molecule.GetAtoms()}

    for atom_index in [*atoms_to_use]:
        atom = rd_atoms_by_index[atom_index]

        old_valence = atom.GetTotalValence()
        new_valence = atom.GetTotalValence()

        for neighbour_bond in rd_atoms_by_index[atom_index].GetBonds():
            if neighbour_bond.GetBeginAtomIdx() in atoms_to_use and neighbour_bond.GetEndAtomIdx() in atoms_to_use:
                continue

            new_valence -= neighbour_bond.GetValenceContrib(atom)

        if numpy.isclose(old_valence, new_valence):
            # Skip the cases where the valence won't change
            continue

        if (
            atom.GetAtomicNum() == 6
            and atom.GetIsAromatic()
            and sum(1 for bond_tuple in bond_indices if atom.GetAtomMapNum() in bond_tuple) == 1
        ):
            # This is likely a cap carbon which was retained from an existing ring. It's
            # aromaticity needs to be cleared before calling ``MolFragmentToSmiles``
            # otherwise will (understandably) be confused and throw an exception.
            atom.SetIsAromatic(False)

        # Add a hydrogen to the atom whose valence will change.
        for _ in range(int(numpy.rint(old_valence - new_valence))):
            new_atom = Chem.Atom(1)
            new_atom_index = rd_molecule.AddAtom(new_atom)

            rd_molecule.AddBond(atom_index, new_atom_index)

            new_bond = rd_molecule.GetBondBetweenAtoms(atom_index, new_atom_index)
            new_bond.SetBondType(Chem.BondType.SINGLE)
            new_bond.SetIsAromatic(False)

            atoms_to_use.append(new_atom_index)
            bonds_to_use.append(new_bond.GetIdx())

    fragment_smiles = Chem.MolFragmentToSmiles(rd_molecule, atoms_to_use, bonds_to_use)
    fragment = Molecule.from_smiles(fragment_smiles, allow_undefined_stereo=True)

    return fragment


def extract_fragment(molecule: Molecule, atom_indices: set[int], bond_indices: set[tuple[int, int]]) -> Molecule:
    """Returns a fragment which contains the specified atoms and bonds of a parent
    molecule

    Parameters
    ----------
    molecule
        The parent molecule being fragmented.
    atom_indices
        The *map* indices of the subset of atoms to retain.
    bond_indices
        The *map* indices of the subset of bonds to retain.
    """

    # Make sure that the bond and atom indices are self consistent and that there
    # are no disconnected bonds.
    if not all(i in atom_indices for map_tuple in bond_indices for i in map_tuple):
        raise ValueError("The ``bond_indices`` set includes atoms not in the ``atom_indices`` set.")

    fragment = _extract_rd_fragment(molecule, atom_indices, bond_indices)

    # Sanity check that all atoms are still bonded
    fragment_smiles = fragment.to_smiles()

    assert "." not in fragment_smiles, (
        "An atom that is not bonded to any other atom in the fragment. "
        "You probably ran into a bug. Please report the input molecule to the "
        "issue tracker"
    )

    return fragment


def smiles_to_molecule(smiles, add_atom_map: bool = False) -> Molecule:
    """Create a molecule object from an input SMILES pattern.

    Parameters
    ----------
    smiles : str
        SMILES representation of desired molecule.
    add_atom_map
        Whether to create a canonical atom map for the molecule.

    Returns
    -------
        A normalized molecule with desired smiles string.
    """

    molecule = Molecule.from_smiles(smiles, allow_undefined_stereo=True)

    # Add canonical ordered atom maps
    if add_atom_map:
        molecule = molecule.canonical_order_atoms()
        molecule.properties["atom_map"] = {i: i + 1 for i in range(molecule.n_atoms)}

    return molecule
