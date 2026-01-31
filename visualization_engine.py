"""
Visualization Engine - Quality Control for the Digital Data Factory
Generates 2D images and 3D coordinate files for molecules.
"""

import os
from rdkit import Chem
from rdkit.Chem import Draw

from molecule_builder import create_molecule


def generate_visualization(mol, mol_name, images_dir=None, structures_dir=None):
    """
    Generate 2D image and 3D XYZ file for a molecule.
    
    Args:
        mol: RDKit Mol object with 3D coordinates.
        mol_name: Name for the output files.
        images_dir: Optional custom directory for 2D images.
        structures_dir: Optional custom directory for 3D XYZ files.
        
    Returns:
        Tuple of (png_path, xyz_path) for the generated files.
    """
    # Use default output folder if no custom paths provided
    if images_dir is None:
        images_dir = "output"
    if structures_dir is None:
        structures_dir = "output"
    
    # Ensure output folders exist
    for dir_path in [images_dir, structures_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    
    # === 2D Image Generation ===
    png_path = os.path.join(images_dir, f"{mol_name}.png")
    
    # Remove Hs for cleaner 2D visualization
    mol_2d = Chem.RemoveHs(mol)
    img = Draw.MolToImage(mol_2d, size=(400, 400))
    img.save(png_path)
    
    # === 3D XYZ File Generation ===
    xyz_path = os.path.join(structures_dir, f"{mol_name}.xyz")
    
    # Get XYZ block from RDKit
    xyz_content = Chem.MolToXYZBlock(mol)
    
    with open(xyz_path, 'w') as f:
        f.write(xyz_content)
    
    return png_path, xyz_path


if __name__ == "__main__":
    print("=" * 50)
    print("   Visualization Engine - Quality Control Test")
    print("=" * 50)
    
    # Test with Methane ('C')
    smiles = 'C'
    mol_name = 'methane'
    
    print(f"\nTesting with: {mol_name.capitalize()} (SMILES: '{smiles}')")
    print("-" * 50)
    
    # Build the molecule
    mol = create_molecule(smiles)
    print(f"✓ Molecule built with {mol.GetNumAtoms()} atoms")
    
    # Generate visualizations
    png_path, xyz_path = generate_visualization(mol, mol_name)
    
    # Confirm files exist
    print("-" * 50)
    if os.path.exists(png_path) and os.path.exists(xyz_path):
        print("✅ Visualization engine test complete!")
        print(f"   PNG size: {os.path.getsize(png_path)} bytes")
        print(f"   XYZ size: {os.path.getsize(xyz_path)} bytes")
    else:
        print("❌ Error: Files not created properly")
