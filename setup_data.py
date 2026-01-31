"""
Setup Data Script - Initializes the Digital Data Factory input structure.
Creates input folder and chemicals list for processing.
"""

import os
import csv


def setup_input_data():
    """Create the inputs folder and chemicals list CSV."""
    
    # Create inputs folder
    inputs_dir = "inputs"
    if not os.path.exists(inputs_dir):
        os.makedirs(inputs_dir)
        print(f"📁 Created folder: {inputs_dir}/")
    else:
        print(f"📁 Folder already exists: {inputs_dir}/")
    
    # Define diverse molecules list
    chemicals = [
        ("C", "Methane"),
        ("O", "Water"),
        ("N", "Ammonia"),
        ("C=C", "Ethylene"),
        ("c1ccccc1", "Benzene"),
        ("C1CCCCC1", "Cyclohexane"),
        ("CCO", "Ethanol"),
        ("CC(=O)O", "Acetic_Acid"),
        ("CC(=O)Oc1ccccc1C(=O)O", "Aspirin"),
        ("C1=CC=C(C=C1)O", "Phenol"),
    ]
    
    # Create chemicals list CSV
    csv_path = os.path.join(inputs_dir, "chemicals_list.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["SMILES", "Name"])  # Header
        writer.writerows(chemicals)
    
    print(f"📄 Created chemicals list: {csv_path}")
    print(f"   Contains {len(chemicals)} molecules:")
    for smiles, name in chemicals:
        print(f"   - {name}: '{smiles}'")
    
    print("\n✅ Input data setup complete!")


if __name__ == "__main__":
    print("=" * 50)
    print("   Digital Data Factory - Input Data Setup")
    print("=" * 50)
    setup_input_data()
