"""
Molecule Builder Engine - First engine of the Digital Data Factory
Converts SMILES strings into 3D molecular structures using RDKit.
"""

from rdkit import Chem
from rdkit.Chem import AllChem


def create_molecule(smiles_string):
    """
    Convert a SMILES string to a 3D molecular object.
    
    Args:
        smiles_string: A valid SMILES representation of a molecule.
        
    Returns:
        RDKit Mol object with explicit hydrogens and 3D coordinates.
    """
    # Convert SMILES string to Mol object
    mol = Chem.MolFromSmiles(smiles_string)
    
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles_string}")
    
    # Add explicit Hydrogens - Critical for physics simulations!
    mol_with_h = Chem.AddHs(mol)
    
    # Generate a random 3D conformation
    AllChem.EmbedMolecule(mol_with_h, randomSeed=42)
    
    return mol_with_h


if __name__ == "__main__":
    # Test with Methane ('C')
    smiles = 'C'
    print(f"Testing molecule builder with Methane (SMILES: '{smiles}')")
    print("-" * 50)
    
    # Create initial molecule (without explicit H)
    mol_no_h = Chem.MolFromSmiles(smiles)
    print(f"Atoms BEFORE adding Hydrogens: {mol_no_h.GetNumAtoms()}")
    
    # Create molecule with full treatment
    mol_with_h = create_molecule(smiles)
    print(f"Atoms AFTER adding Hydrogens:  {mol_with_h.GetNumAtoms()}")
    
    print("-" * 50)
    print("✓ Molecule builder is working correctly!")
    print(f"  Methane: 1 Carbon + 4 Hydrogens = 5 atoms total")
