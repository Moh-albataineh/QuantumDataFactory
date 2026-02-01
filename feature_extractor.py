"""
Feature Extractor - AI Data Enrichment Module for Digital Data Factory
Extracts molecular descriptors for machine learning applications.
Includes deep chemical descriptors for drug-likeness analysis.
"""

from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen
from rdkit.Chem.QED import qed


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


def extract_deep_features(mol):
    """
    Extract deep chemical descriptors for drug-likeness analysis.
    
    Args:
        mol: RDKit Mol object.
        
    Returns:
        Dictionary of deep molecular features including:
        - TPSA: Topological Polar Surface Area (drug absorption)
        - QED: Quantitative Estimation of Drug-likeness
        - Frac_CSP3: Fraction of sp3 carbons (3D complexity)
        - MolLogP: Partition coefficient (lipophilicity)
    """
    # Basic features
    features = extract_features(mol)
    
    # Deep chemical descriptors
    deep_features = {
        'TPSA': Descriptors.TPSA(mol),           # Topological Polar Surface Area
        'QED': qed(mol),                          # Drug-likeness score (0-1)
        'Frac_CSP3': Descriptors.FractionCSP3(mol),  # Carbon saturation
        'MolLogP': Crippen.MolLogP(mol),          # Lipophilicity
        'NumHDonors': Descriptors.NumHDonors(mol),    # H-bond donors
        'NumHAcceptors': Descriptors.NumHAcceptors(mol),  # H-bond acceptors
        'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),  # Flexibility
    }
    
    # Merge all features
    features.update(deep_features)
    
    return features


if __name__ == "__main__":
    from molecule_builder import create_molecule
    
    print("=" * 60)
    print("   Feature Extractor - Deep Descriptors Test")
    print("=" * 60)
    
    # Test molecules
    test_molecules = [
        ('C', 'Methane'),
        ('c1ccccc1', 'Benzene'),
        ('CCO', 'Ethanol'),
        ('CC(=O)Oc1ccccc1C(=O)O', 'Aspirin'),
    ]
    
    for smiles, name in test_molecules:
        print(f"\n{name} (SMILES: '{smiles}')")
        print("-" * 40)
        
        mol = create_molecule(smiles)
        features = extract_deep_features(mol)
        
        for key, value in features.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")
    
    print("\n✅ Deep feature extractor test complete!")
