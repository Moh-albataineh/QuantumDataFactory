"""
Predict New - Deployment Script for Digital Data Factory
Uses trained ML model to instantly predict molecular energies.
"""

import joblib
import pandas as pd

from molecule_builder import create_molecule
from feature_extractor import extract_features


# Load trained model once at startup
print("🔄 Loading trained model...")
MODEL = joblib.load('energy_predictor_model.pkl')
print("✓ Model loaded successfully!\n")


def predict_energy(smiles):
    """
    Predict molecular energy from SMILES string using trained ML model.
    
    Args:
        smiles: SMILES string representation of the molecule.
        
    Returns:
        Predicted energy in Hartrees.
    """
    # Generate molecule from SMILES
    mol = create_molecule(smiles)
    
    # Extract features
    features = extract_features(mol)
    
    # Create DataFrame matching training columns
    feature_df = pd.DataFrame([{
        'Mol_Weight': features['Mol_Weight'],
        'Num_Atoms': features['Num_Atoms'],
        'Num_Rings': features['Num_Rings'],
        'Num_Valence_Electrons': features['Num_Valence_Electrons']
    }])
    
    # Predict energy
    predicted_energy = MODEL.predict(feature_df)[0]
    
    return predicted_energy, features


def main():
    print("=" * 60)
    print("    DIGITAL DATA FACTORY - AI Energy Predictor")
    print("=" * 60)
    print("Predict molecular energies instantly using ML!")
    print("Type 'quit' or 'exit' to stop.\n")
    
    while True:
        try:
            # Get user input
            smiles = input("Enter a SMILES string (e.g., CCO): ").strip()
            
            # Check for exit commands
            if smiles.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if not smiles:
                print("⚠️  Please enter a valid SMILES string.\n")
                continue
            
            # Predict energy
            print(f"\n🔬 Analyzing: {smiles}")
            energy, features = predict_energy(smiles)
            
            # Display results
            print("-" * 40)
            print(f"📊 Molecular Features:")
            print(f"   Weight:     {features['Mol_Weight']:.2f} g/mol")
            print(f"   Atoms:      {features['Num_Atoms']}")
            print(f"   Rings:      {features['Num_Rings']}")
            print(f"   Valence e⁻: {features['Num_Valence_Electrons']}")
            print(f"\n⚡ Predicted Energy: {energy:.6f} Hartrees")
            print(f"                     {energy * 627.509:.2f} kcal/mol")
            print("-" * 40 + "\n")
            
        except ValueError as e:
            print(f"❌ Error: {e}\n")
        except Exception as e:
            print(f"❌ Unexpected error: {e}\n")


if __name__ == "__main__":
    main()
