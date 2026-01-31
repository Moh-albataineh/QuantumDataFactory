"""
Feature Extractor - AI Data Enrichment Module for Digital Data Factory
Extracts molecular descriptors for machine learning applications.
"""

from rdkit import Chem
from rdkit.Chem import Descriptors


def extract_features(mol):
    """
    Extract molecular features/descriptors for AI/ML training.
    
    Args:
        mol: RDKit Mol object (with or without explicit hydrogens).
        
    Returns:
        Dictionary of molecular features.
    """
    features = {
        'Mol_Weight': Descriptors.MolWt(mol),
        'Num_Atoms': mol.GetNumAtoms(),
        'Num_Rings': Descriptors.RingCount(mol),
        'Num_Valence_Electrons': Descriptors.NumValenceElectrons(mol),
    }
    
    return features


if __name__ == "__main__":
    from molecule_builder import create_molecule
    
    print("=" * 50)
    print("   Feature Extractor - Test Module")
    print("=" * 50)
    
    # Test molecules
    test_molecules = [
        ('C', 'Methane'),
        ('c1ccccc1', 'Benzene'),
        ('CCO', 'Ethanol'),
    ]
    
    for smiles, name in test_molecules:
        print(f"\n{name} (SMILES: '{smiles}')")
        print("-" * 30)
        
        mol = create_molecule(smiles)
        features = extract_features(mol)
        
        for key, value in features.items():
            print(f"  {key}: {value}")
    
    print("\n✅ Feature extractor test complete!")
