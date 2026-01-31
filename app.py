"""
QuantumData Factory - Streamlit Web Application v1.0
Clean UI with strict Input/Output separation and proper state management.
"""

import os
import glob
import streamlit as st
import pandas as pd
import joblib
import py3Dmol
from stmol import showmol
from rdkit import Chem

from molecule_builder import create_molecule
from feature_extractor import extract_features
from energy_calculator import calculate_energy


# === PAGE CONFIG ===
st.set_page_config(
    page_title="QuantumData Factory",
    page_icon="⚛️",
    layout="wide"
)


# === INITIALIZE SESSION STATE ===
if 'last_result' not in st.session_state:
    st.session_state.last_result = None


# === MOLECULE LIBRARY ===
MOLECULE_LIBRARY = {
    'Benzene': 'c1ccccc1',
    'Ethanol': 'CCO',
    'Aspirin': 'CC(=O)Oc1ccccc1C(=O)O',
    'Caffeine': 'Cn1cnc2c1c(=O)n(c(=O)n2C)C',
    'Phenol': 'c1ccccc1O',
    'Acetone': 'CC(=O)C',
    'Water': 'O',
    'Methanol': 'CO',
    'Pyridine': 'n1ccccc1'
}


# === LOAD MODEL ===
@st.cache_resource
def load_model():
    return joblib.load('energy_predictor_model.pkl')

MODEL = load_model()


# === HELPER FUNCTIONS ===
def get_3d_view(mol):
    xyz_block = Chem.MolToXYZBlock(mol)
    view = py3Dmol.view(width=800, height=500)
    view.addModel(xyz_block, 'xyz')
    view.setStyle({'stick': {'colorscheme': 'Jmol', 'radius': 0.2}})
    view.addStyle({'sphere': {'colorscheme': 'Jmol', 'scale': 0.3}})
    view.setBackgroundColor('#0e1117')
    view.zoomTo()
    return view


def mol_to_psi4_geometry(mol):
    xyz_block = Chem.MolToXYZBlock(mol)
    lines = xyz_block.strip().split('\n')
    return '\n'.join(lines[2:])


def cleanup_psi4_files():
    for pattern in ['psi4_output.dat', 'timer.dat']:
        for f in glob.glob(pattern):
            try:
                os.remove(f)
            except:
                pass


# === SIDEBAR (INPUT SECTION) ===
with st.sidebar:
    st.title("🧪 Input Panel")
    st.markdown("---")
    
    # SMILES INPUT (TOP OF SIDEBAR)
    st.markdown("### 📝 SMILES Code")
    smiles_input = st.text_input(
        "Enter SMILES:",
        value="c1ccccc1",
        placeholder="e.g., CCO",
        key="smiles_text"
    )
    
    st.markdown("---")
    
    # QUICK EXAMPLES
    st.markdown("### 🧬 Quick Examples")
    example_choice = st.selectbox(
        "Load from library:",
        options=["-- Select --"] + list(MOLECULE_LIBRARY.keys()),
        key="example_select"
    )
    
    # Update text input if example selected
    if example_choice != "-- Select --":
        smiles_input = MOLECULE_LIBRARY[example_choice]
        st.info(f"Loaded: `{smiles_input}`")
    
    st.markdown("---")
    
    # CALCULATION ENGINE
    st.markdown("### ⚙️ Calculation Engine")
    engine = st.radio(
        "Method:",
        ["🚀 AI (Instant)", "⚛️ Quantum (1-2 min)"],
        key="engine_radio"
    )
    
    st.markdown("---")
    
    # ANALYZE BUTTON (PROMINENT)
    st.markdown("### 🎯 Execute")
    analyze_clicked = st.button(
        "🔬 ANALYZE & CALCULATE",
        type="primary",
        use_container_width=True
    )


# === MAIN AREA (OUTPUT SECTION) ===
st.title("⚛️ QuantumData Factory")

# Process calculation when button clicked
if analyze_clicked:
    try:
        # Determine spinner message based on engine
        if "Quantum" in engine:
            spinner_msg = "⚛️ Quantum Engine Running... Please wait (1-2 mins)..."
        else:
            spinner_msg = "🧠 AI Prediction in progress..."
        
        with st.spinner(spinner_msg):
            # Build molecule
            mol = create_molecule(smiles_input)
            features = extract_features(mol)
            
            # Prepare features for AI
            feature_df = pd.DataFrame([{
                'Mol_Weight': features['Mol_Weight'],
                'Num_Atoms': features['Num_Atoms'],
                'Num_Rings': features['Num_Rings'],
                'Num_Valence_Electrons': features['Num_Valence_Electrons']
            }])
            ai_energy = MODEL.predict(feature_df)[0]
            
            # Calculate based on engine
            if "Quantum" in engine:
                geometry = mol_to_psi4_geometry(mol)
                quantum_energy = calculate_energy(geometry)
                cleanup_psi4_files()
                
                result = {
                    'smiles': smiles_input,
                    'name': example_choice if example_choice != "-- Select --" else "Custom",
                    'mol': mol,
                    'features': features,
                    'energy': quantum_energy,
                    'ai_energy': ai_energy,
                    'quantum_energy': quantum_energy,
                    'engine': 'Quantum',
                    'source': 'Hartree-Fock / STO-3G'
                }
            else:
                result = {
                    'smiles': smiles_input,
                    'name': example_choice if example_choice != "-- Select --" else "Custom",
                    'mol': mol,
                    'features': features,
                    'energy': ai_energy,
                    'ai_energy': ai_energy,
                    'quantum_energy': None,
                    'engine': 'AI',
                    'source': 'Machine Learning Model'
                }
            
            # Store in session state (FREEZE LOGIC)
            st.session_state.last_result = result
            
    except ValueError as e:
        st.error(f"❌ Invalid SMILES: {e}")
    except Exception as e:
        st.error(f"❌ Error: {e}")
        cleanup_psi4_files()


# === DISPLAY RESULTS (FROM SESSION STATE ONLY) ===
if st.session_state.last_result:
    r = st.session_state.last_result
    
    # HEADER WITH MOLECULE INFO
    st.markdown("---")
    st.header(f"📊 Result for: {r['name']} (`{r['smiles']}`)")
    st.caption(f"Calculated using: {r['source']}")
    
    # TWO COLUMN LAYOUT
    col_left, col_right = st.columns([1.5, 1])
    
    # LEFT: 3D VISUALIZATION
    with col_left:
        st.markdown("### 🔮 3D Structure")
        view = get_3d_view(r['mol'])
        showmol(view, height=500, width=800)
    
    # RIGHT: METRICS
    with col_right:
        # ENERGY
        icon = "⚛️" if r['engine'] == 'Quantum' else "🚀"
        st.markdown(f"### {icon} Energy")
        st.metric(
            label="Calculated Energy",
            value=f"{r['energy']:.6f} Ha",
            delta=f"{r['energy'] * 627.509:.2f} kcal/mol"
        )
        
        # AI VS QUANTUM COMPARISON
        if r['quantum_energy'] is not None:
            st.markdown("---")
            st.markdown("### 🔄 AI vs Reality")
            
            c1, c2 = st.columns(2)
            with c1:
                st.metric("🚀 AI", f"{r['ai_energy']:.4f} Ha")
            with c2:
                st.metric("⚛️ Quantum", f"{r['quantum_energy']:.4f} Ha")
            
            error = abs(r['ai_energy'] - r['quantum_energy'])
            error_pct = (error / abs(r['quantum_energy'])) * 100
            
            if error_pct < 1:
                st.success(f"✅ Error: {error_pct:.2f}%")
            elif error_pct < 5:
                st.info(f"📊 Error: {error_pct:.2f}%")
            else:
                st.warning(f"⚠️ Error: {error_pct:.2f}%")
        
        # FEATURES
        st.markdown("---")
        st.markdown("### 📊 Features")
        
        f1, f2 = st.columns(2)
        with f1:
            st.metric("Weight", f"{r['features']['Mol_Weight']:.1f} g/mol")
            st.metric("Rings", r['features']['Num_Rings'])
        with f2:
            st.metric("Atoms", r['features']['Num_Atoms'])
            st.metric("Valence e⁻", r['features']['Num_Valence_Electrons'])
    
    st.success(f"✅ Analysis complete using {r['engine']} engine!")

else:
    # WELCOME MESSAGE (FIRST LOAD)
    st.markdown("---")
    st.info("👈 Enter a SMILES code or select an example, then click **ANALYZE & CALCULATE**")
    
    st.markdown("### 🎯 Choose Your Engine")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        #### 🚀 AI Prediction
        - Instant results
        - ~99% accuracy
        - Best for screening
        """)
    with c2:
        st.markdown("""
        #### ⚛️ Quantum Simulation
        - 1-2 minute runtime
        - Exact calculation
        - Best for validation
        """)
    
    st.markdown("---")
    st.markdown("### 🧬 Molecule Library")
    st.dataframe(
        pd.DataFrame(list(MOLECULE_LIBRARY.items()), columns=['Name', 'SMILES']),
        use_container_width=True,
        hide_index=True
    )


# === FOOTER ===
st.markdown("---")
st.caption("Built with ❤️ using RDKit, Psi4 & Scikit-Learn | QuantumData Factory v1.0")
