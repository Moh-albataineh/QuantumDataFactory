"""
Main Factory Driver - Digital Data Factory Pipeline V2
Reads input from CSV, processes molecules, outputs to organized folders.
"""

import os
import pandas as pd
from rdkit import Chem

from molecule_builder import create_molecule
from energy_calculator import calculate_energy
from visualization_engine import generate_visualization
from feature_extractor import extract_features


# === OUTPUT DIRECTORY STRUCTURE ===
OUTPUT_BASE = "factory_output"
IMAGES_DIR = os.path.join(OUTPUT_BASE, "2D_Images")
STRUCTURES_DIR = os.path.join(OUTPUT_BASE, "3D_Structures")
REPORTS_DIR = os.path.join(OUTPUT_BASE, "reports")


def setup_output_folders():
    """Create organized output folder structure if it doesn't exist."""
    folders = [IMAGES_DIR, STRUCTURES_DIR, REPORTS_DIR]
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"📁 Created: {folder}")


def mol_to_psi4_geometry(mol):
    """
    Convert an RDKit molecule with 3D coordinates to Psi4 geometry string.
    
    Args:
        mol: RDKit Mol object with 3D coordinates and explicit hydrogens.
        
    Returns:
        Psi4-compatible XYZ geometry string.
    """
    xyz_block = Chem.MolToXYZBlock(mol)
    lines = xyz_block.strip().split('\n')
    atom_lines = lines[2:]  # Skip header lines
    return '\n'.join(atom_lines)


def main():
    print("=" * 60)
    print("       DIGITAL DATA FACTORY V2 - Production Pipeline")
    print("=" * 60)
    
    # === Step 0: Setup output folders ===
    print("\n📂 Setting up output structure...")
    setup_output_folders()
    
    # === Step 1: Read input data ===
    input_file = "inputs/chemicals_list.csv"
    print(f"\n📥 Reading input from: {input_file}")
    
    if not os.path.exists(input_file):
        print(f"❌ Error: Input file not found: {input_file}")
        print("   Run setup_data.py first to create input data.")
        return
    
    input_df = pd.read_csv(input_file)
    print(f"   Found {len(input_df)} molecules to process")
    
    # === Step 2: Process each molecule ===
    results = []
    
    for idx, row in input_df.iterrows():
        smiles = row['SMILES']
        name = row['Name']
        
        print(f"\n▶ [{idx+1}/{len(input_df)}] Processing: {name}")
        print("-" * 40)
        
        try:
            # Build 3D molecule
            print("  Step 1: Building 3D molecule...")
            mol = create_molecule(smiles)
            num_atoms = mol.GetNumAtoms()
            print(f"          ✓ Built with {num_atoms} atoms")
            
            # Extract molecular features for AI
            print("  Step 2: Extracting features...")
            features = extract_features(mol)
            print(f"          ✓ Features: MW={features['Mol_Weight']:.2f}, Rings={features['Num_Rings']}")
            
            # Extract geometry
            print("  Step 3: Extracting geometry...")
            geometry = mol_to_psi4_geometry(mol)
            print("          ✓ Geometry extracted")
            
            # Calculate energy
            print("  Step 4: Running quantum calculation...")
            energy = calculate_energy(geometry)
            print(f"          ✓ Energy: {energy:.6f} Hartrees")
            
            # Generate visualizations
            print("  Step 5: Generating visualizations...")
            generate_visualization(mol, name, 
                                   images_dir=IMAGES_DIR, 
                                   structures_dir=STRUCTURES_DIR)
            print(f"          -> 📸 Images generated for {name}")
            
            status = "Success"
            
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            energy = None
            features = {'Mol_Weight': None, 'Num_Atoms': None, 'Num_Rings': None, 'Num_Valence_Electrons': None}
            status = f"Error: {str(e)}"
        
        # Store result with features
        results.append({
            'SMILES': smiles,
            'Name': name,
            'Mol_Weight': features['Mol_Weight'],
            'Num_Atoms': features['Num_Atoms'],
            'Num_Rings': features['Num_Rings'],
            'Num_Valence_Electrons': features['Num_Valence_Electrons'],
            'Energy_Hartrees': energy,
            'Status': status
        })
    
    # === Step 3: Generate Report ===
    print("\n" + "=" * 60)
    print("                    PRODUCTION REPORT")
    print("=" * 60)
    
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    # Save to reports folder
    report_path = os.path.join(REPORTS_DIR, "factory_data.csv")
    results_df.to_csv(report_path, index=False)
    
    # Save AI-ready dataset
    ai_data_path = os.path.join(REPORTS_DIR, "ai_ready_data.csv")
    results_df.to_csv(ai_data_path, index=False)
    print(f"\n🤖 AI-ready dataset saved to: {ai_data_path}")
    
    # Summary
    success_count = len([r for r in results if r['Status'] == 'Success'])
    error_count = len(results) - success_count
    
    print("\n" + "-" * 60)
    print("📊 SUMMARY:")
    print(f"   Total molecules:  {len(results)}")
    print(f"   Successful:       {success_count} ✓")
    print(f"   Errors:           {error_count}")
    print(f"\n💾 Report saved to: {report_path}")
    print(f"🖼️  2D Images in:    {IMAGES_DIR}/")
    print(f"📐 3D Structures in: {STRUCTURES_DIR}/")
    print("\n✅ Factory pipeline V2 complete!")


if __name__ == "__main__":
    main()
