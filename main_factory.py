"""
Main Factory Driver - Digital Data Factory Pipeline
Processes molecules from SMILES through quantum energy calculations.
"""

import pandas as pd
from rdkit import Chem

from molecule_builder import create_molecule
from energy_calculator import calculate_energy


def mol_to_psi4_geometry(mol):
    """
    Convert an RDKit molecule with 3D coordinates to Psi4 geometry string.
    
    Args:
        mol: RDKit Mol object with 3D coordinates and explicit hydrogens.
        
    Returns:
        Psi4-compatible XYZ geometry string.
    """
    # Get XYZ block from RDKit (skip first two header lines)
    xyz_block = Chem.MolToXYZBlock(mol)
    lines = xyz_block.strip().split('\n')
    
    # Skip atom count and comment lines (first 2 lines of XYZ format)
    atom_lines = lines[2:]
    
    # Format for Psi4
    geometry_str = '\n'.join(atom_lines)
    return geometry_str


def main():
    print("=" * 60)
    print("       DIGITAL DATA FACTORY - Main Pipeline")
    print("=" * 60)
    
    # Raw materials to process
    raw_materials = ['C', 'O', 'N', 'C=C']  # Methane, Water, Ammonia, Ethylene
    molecule_names = ['Methane', 'Water', 'Ammonia', 'Ethylene']
    
    # Results storage
    results = []
    
    for smiles, name in zip(raw_materials, molecule_names):
        print(f"\n▶ Processing: {name} (SMILES: '{smiles}')")
        print("-" * 40)
        
        try:
            # Step 1: Build the molecule
            print("  Step 1: Building 3D molecule...")
            mol = create_molecule(smiles)
            num_atoms = mol.GetNumAtoms()
            print(f"          ✓ Built with {num_atoms} atoms")
            
            # Step 2: Get geometry string (XYZ block)
            print("  Step 2: Extracting geometry...")
            geometry = mol_to_psi4_geometry(mol)
            print("          ✓ Geometry extracted")
            
            # Step 3: Calculate energy
            print("  Step 3: Running quantum calculation...")
            energy = calculate_energy(geometry)
            print(f"          ✓ Energy: {energy:.6f} Hartrees")
            
            status = "Success"
            
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            energy = None
            status = f"Error: {str(e)}"
        
        # Store result
        results.append({
            'SMILES': smiles,
            'Name': name,
            'Energy_Hartrees': energy,
            'Status': status
        })
    
    # Create DataFrame
    print("\n" + "=" * 60)
    print("                    RESULTS")
    print("=" * 60)
    
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    
    # Save to CSV
    output_file = 'factory_data.csv'
    df.to_csv(output_file, index=False)
    print(f"\n💾 Data saved to: {output_file}")
    print("✅ Factory pipeline complete!")


if __name__ == "__main__":
    main()
