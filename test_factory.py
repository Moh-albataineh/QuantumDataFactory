"""
Digital Data Factory - System Verification Script
Tests that all required computational chemistry libraries are properly installed.
"""

import sys

# Fix Windows console encoding for Unicode characters
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')


def main():
    print("=" * 50)
    print("Digital Data Factory - System Verification")
    print("=" * 50)
    print()

    all_passed = True

    # Test Psi4 import
    try:
        import psi4
        print(f"[OK] Psi4 imported successfully")
        print(f"     Version: {psi4.__version__}")
    except ImportError as e:
        print(f"[FAIL] Psi4 import failed: {e}")
        all_passed = False

    print()

    # Test RDKit import
    try:
        import rdkit
        from rdkit import Chem
        print(f"[OK] RDKit imported successfully")
        print(f"     Version: {rdkit.__version__}")
    except ImportError as e:
        print(f"[FAIL] RDKit import failed: {e}")
        all_passed = False

    print()
    print("=" * 50)
    
    if all_passed:
        print("SUCCESS: The Digital Data Factory engines are online!")
    else:
        print("WARNING: Some dependencies are missing. Please install them.")
    
    print("=" * 50)


if __name__ == "__main__":
    main()
