"""
Energy Calculator Engine - Quantum Engine of the Digital Data Factory
Performs quantum mechanical energy calculations using Psi4.
"""

import psi4

# Configure Psi4 for fast local execution
psi4.set_memory('4 GB')
psi4.set_num_threads(4)

# Suppress excessive output
psi4.core.set_output_file('psi4_output.dat', False)


def calculate_energy(mol_geometry_string):
    """
    Calculate the quantum mechanical energy of a molecule.
    
    Args:
        mol_geometry_string: Molecular geometry in Psi4 Z-matrix or XYZ format.
        
    Returns:
        Energy value in Hartrees.
    """
    # Create Psi4 molecule from geometry string
    molecule = psi4.geometry(mol_geometry_string)
    
    # Run Hartree-Fock calculation with minimal basis set (fast!)
    energy = psi4.energy('hf/sto-3g')
    
    return energy


if __name__ == "__main__":
    print("Quantum Engine - Energy Calculator")
    print("=" * 50)
    
    # Define water molecule geometry (Psi4 Z-matrix format)
    water_geometry = """
    O
    H 1 0.96
    H 1 0.96 2 104.5
    """
    
    print("Molecule: Water (H2O)")
    print("Method: Hartree-Fock / STO-3G basis set")
    print("-" * 50)
    print("Running quantum calculation...")
    
    # Calculate energy
    energy = calculate_energy(water_geometry)
    
    print(f"\nEnergy: {energy:.10f} Hartrees")
    print(f"Energy: {energy * 627.509:.4f} kcal/mol")
    print("-" * 50)
    print("✅ Quantum calculation complete!")
