"""
QuantumData Factory - Streamlit Web Application v2.0
Dual-mode support: Single Molecule Analysis & Batch Factory Processing.
"""

import os
import glob
import streamlit as st
import pandas as pd
import joblib
import py3Dmol
from stmol import showmol
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

from molecule_builder import create_molecule
from feature_extractor import extract_features, extract_deep_features
from energy_calculator import calculate_energy
from report_generator import generate_pdf_report


# === PAGE CONFIG ===
st.set_page_config(
    page_title="QuantumData Factory",
    page_icon="⚛️",
    layout="wide"
)


# === INITIALIZE SESSION STATE ===
if 'last_result' not in st.session_state:
    st.session_state.last_result = None
if 'batch_result' not in st.session_state:
    st.session_state.batch_result = None


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
def get_3d_view(mol, style="ball_stick", auto_rotate=False):
    """Generate 3D molecular visualization with configurable style."""
    xyz_block = Chem.MolToXYZBlock(mol)
    view = py3Dmol.view(width=800, height=500)
    view.addModel(xyz_block, 'xyz')
    
    # Apply style based on selection
    if style == "vdw_surface":
        view.setStyle({'stick': {'colorscheme': 'Jmol', 'radius': 0.1}})
        view.addSurface(py3Dmol.VDW, {'opacity': 0.7, 'colorscheme': 'Jmol'})
    elif style == "stick_only":
        view.setStyle({'stick': {'colorscheme': 'Jmol', 'radius': 0.15}})
    else:  # ball_stick (default)
        view.setStyle({'stick': {'colorscheme': 'Jmol', 'radius': 0.2}})
        view.addStyle({'sphere': {'colorscheme': 'Jmol', 'scale': 0.3}})
    
    view.setBackgroundColor('#0e1117')
    view.zoomTo()
    
    # Add rotation animation if enabled
    if auto_rotate:
        view.spin({'x': 1, 'y': 0.5, 'z': 0}, 1)
    
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


def predict_energy_for_smiles(smiles):
    """Predict energy for a single SMILES using AI model with deep features."""
    try:
        mol = create_molecule(smiles)
        features = extract_deep_features(mol)  # Use deep features
        feature_df = pd.DataFrame([{
            'Mol_Weight': features['Mol_Weight'],
            'Num_Atoms': features['Num_Atoms'],
            'Num_Rings': features['Num_Rings'],
            'Num_Valence_Electrons': features['Num_Valence_Electrons']
        }])
        energy = MODEL.predict(feature_df)[0]
        return energy, features, None
    except Exception as e:
        return None, None, str(e)


# === SIDEBAR ===
with st.sidebar:
    st.title("⚛️ QuantumData Factory")
    st.markdown("---")
    
    # MODE SELECTOR (TOP OF SIDEBAR)
    mode = st.radio(
        "Select Mode",
        ["🧪 Single Molecule", "🏭 Batch Factory"],
        key="mode_selector"
    )
    
    st.markdown("---")


# ============================================================
# MODE 1: SINGLE MOLECULE ANALYSIS
# ============================================================
if mode == "🧪 Single Molecule":
    
    # === SIDEBAR INPUTS FOR SINGLE MODE ===
    with st.sidebar:
        st.markdown("### 📝 SMILES Code")
        smiles_input = st.text_input(
            "Enter SMILES:",
            value="c1ccccc1",
            placeholder="e.g., CCO",
            key="smiles_text"
        )
        
        st.markdown("---")
        
        st.markdown("### 🧬 Quick Examples")
        example_choice = st.selectbox(
            "Load from library:",
            options=["-- Select --"] + list(MOLECULE_LIBRARY.keys()),
            key="example_select"
        )
        
        if example_choice != "-- Select --":
            smiles_input = MOLECULE_LIBRARY[example_choice]
            st.info(f"Loaded: `{smiles_input}`")
        
        st.markdown("---")
        
        st.markdown("### ⚙️ Calculation Engine")
        engine = st.radio(
            "Method:",
            ["🚀 AI (Instant)", "⚛️ Quantum (1-2 min)"],
            key="engine_radio"
        )
        
        st.markdown("---")
        
        st.markdown("### 🎯 Execute")
        analyze_clicked = st.button(
            "🔬 ANALYZE & CALCULATE",
            type="primary",
            use_container_width=True
        )
    
    # === MAIN AREA FOR SINGLE MODE ===
    st.title("🧪 Single Molecule Analysis")
    st.caption("Analyze individual molecules with AI or Quantum calculations")
    
    # Process calculation when button clicked
    if analyze_clicked:
        try:
            if "Quantum" in engine:
                spinner_msg = "⚛️ Quantum Engine Running... Please wait (1-2 mins)..."
            else:
                spinner_msg = "🧠 AI Prediction in progress..."
            
            with st.spinner(spinner_msg):
                mol = create_molecule(smiles_input)
                features = extract_features(mol)
                
                feature_df = pd.DataFrame([{
                    'Mol_Weight': features['Mol_Weight'],
                    'Num_Atoms': features['Num_Atoms'],
                    'Num_Rings': features['Num_Rings'],
                    'Num_Valence_Electrons': features['Num_Valence_Electrons']
                }])
                ai_energy = MODEL.predict(feature_df)[0]
                
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
                
                st.session_state.last_result = result
                
        except ValueError as e:
            st.error(f"❌ Invalid SMILES: {e}")
        except Exception as e:
            st.error(f"❌ Error: {e}")
            cleanup_psi4_files()
    
    # Display results
    if st.session_state.last_result:
        r = st.session_state.last_result
        
        st.markdown("---")
        st.header(f"📊 Result for: {r['name']} (`{r['smiles']}`)")
        st.caption(f"Calculated using: {r['source']}")
        
        col_left, col_right = st.columns([1.5, 1])
        
        with col_left:
            st.markdown("### 🔮 3D Structure")
            
            # Style controls
            style_col1, style_col2 = st.columns([2, 1])
            with style_col1:
                viz_style = st.radio(
                    "Visualization Style",
                    ["🏐 Ball & Stick", "☁️ VdW Surface", "✨ Stick Only"],
                    horizontal=True,
                    key="viz_style"
                )
            with style_col2:
                auto_rotate = st.checkbox("🔄 Auto-Rotate", key="auto_rotate")
            
            # Map style selection to function parameter
            style_map = {
                "🏐 Ball & Stick": "ball_stick",
                "☁️ VdW Surface": "vdw_surface",
                "✨ Stick Only": "stick_only"
            }
            selected_style = style_map.get(viz_style, "ball_stick")
            
            # Render 3D view with selected options
            view = get_3d_view(r['mol'], style=selected_style, auto_rotate=auto_rotate)
            showmol(view, height=500, width=800)
        
        with col_right:
            icon = "⚛️" if r['engine'] == 'Quantum' else "🚀"
            st.markdown(f"### {icon} Energy")
            st.metric(
                label="Calculated Energy",
                value=f"{r['energy']:.6f} Ha",
                delta=f"{r['energy'] * 627.509:.2f} kcal/mol"
            )
            
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
            
            st.markdown("---")
            st.markdown("### 📊 Features")
            
            f1, f2 = st.columns(2)
            with f1:
                st.metric("Weight", f"{r['features']['Mol_Weight']:.1f} g/mol")
                st.metric("Rings", r['features']['Num_Rings'])
            with f2:
                st.metric("Atoms", r['features']['Num_Atoms'])
                st.metric("Valence e⁻", r['features']['Num_Valence_Electrons'])
        
        # PDF Report Download Button
        st.markdown("---")
        st.markdown("### 📄 Research Report")
        
        try:
            pdf_bytes = generate_pdf_report(
                molecule_name=r['name'],
                smiles=r['smiles'],
                mol=r['mol'],
                properties=r['features'],
                energy=r['energy'],
                source=r['source']
            )
            
            st.download_button(
                label="📄 Download Research Report (PDF)",
                data=pdf_bytes,
                file_name=f"{r['name']}_report.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True
            )
            st.caption("Enterprise-grade PDF report with molecular visualization and AI analysis")
        except Exception as e:
            st.error(f"Error generating PDF: {e}")
        
        st.success(f"✅ Analysis complete using {r['engine']} engine!")
    
    else:
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


# ============================================================
# MODE 2: BATCH FACTORY
# ============================================================
elif mode == "🏭 Batch Factory":
    
    # === SIDEBAR FOR BATCH MODE ===
    with st.sidebar:
        st.markdown("### 📁 Upload Data")
        uploaded_file = st.file_uploader(
            "Upload CSV file",
            type=['csv'],
            help="CSV must contain a 'SMILES' column"
        )
        
        st.markdown("---")
        
        if uploaded_file is not None:
            process_clicked = st.button(
                "🚀 PROCESS ALL",
                type="primary",
                use_container_width=True
            )
        else:
            st.info("📤 Upload a CSV to begin")
    
    # === MAIN AREA FOR BATCH MODE ===
    st.title("🏭 High-Throughput AI Screening")
    st.caption("Process hundreds of molecules instantly using trained AI model")
    
    if uploaded_file is not None:
        # Read and validate CSV
        try:
            df = pd.read_csv(uploaded_file)
            
            # Find SMILES column (case-insensitive)
            smiles_col = None
            for col in df.columns:
                if col.lower() == 'smiles':
                    smiles_col = col
                    break
            
            if smiles_col is None:
                st.error("❌ CSV must contain a 'SMILES' column!")
                st.stop()
            
            st.markdown("### 📋 Input Data Preview")
            st.dataframe(df.head(10), use_container_width=True, hide_index=True)
            st.caption(f"Total rows: {len(df)}")
            
            # Process when button clicked
            if 'process_clicked' in dir() and process_clicked:
                st.markdown("---")
                st.markdown("### ⚡ Processing...")
                
                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, row in df.iterrows():
                    smiles = row[smiles_col]
                    status_text.text(f"Processing {idx + 1}/{len(df)}: {smiles[:30]}...")
                    
                    energy, features, error = predict_energy_for_smiles(smiles)
                    
                    if error:
                        results.append({
                            'SMILES': smiles,
                            'AI_Energy_Hartree': None,
                            'Mol_Weight': None,
                            'Num_Atoms': None,
                            'Num_Rings': None,
                            'Num_Valence_Electrons': None,
                            'TPSA': None,
                            'QED': None,
                            'Frac_CSP3': None,
                            'MolLogP': None,
                            'Status': f'Error: {error}'
                        })
                    else:
                        results.append({
                            'SMILES': smiles,
                            'AI_Energy_Hartree': energy,
                            'Mol_Weight': features['Mol_Weight'],
                            'Num_Atoms': features['Num_Atoms'],
                            'Num_Rings': features['Num_Rings'],
                            'Num_Valence_Electrons': features['Num_Valence_Electrons'],
                            'TPSA': features.get('TPSA', 0),
                            'QED': features.get('QED', 0),
                            'Frac_CSP3': features.get('Frac_CSP3', 0),
                            'MolLogP': features.get('MolLogP', 0),
                            'Status': 'Success'
                        })
                    
                    progress_bar.progress((idx + 1) / len(df))
                
                status_text.empty()
                progress_bar.empty()
                
                # Create result DataFrame
                result_df = pd.DataFrame(results)
                
                # Merge with original data
                if 'Name' in df.columns or 'name' in df.columns:
                    name_col = 'Name' if 'Name' in df.columns else 'name'
                    result_df.insert(0, 'Name', df[name_col].values)
                
                st.session_state.batch_result = result_df
        
        except Exception as e:
            st.error(f"❌ Error reading CSV: {e}")
    
    # Display batch results
    if st.session_state.batch_result is not None:
        result_df = st.session_state.batch_result
        
        st.markdown("---")
        st.markdown("### 📊 Results")
        
        # Summary metrics
        success_count = len(result_df[result_df['Status'] == 'Success'])
        error_count = len(result_df) - success_count
        
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Total Processed", len(result_df))
        with m2:
            st.metric("Successful", success_count, delta=None)
        with m3:
            st.metric("Errors", error_count, delta=None if error_count == 0 else f"-{error_count}")
        
        st.markdown("---")
        
        # Display result table
        st.dataframe(result_df, use_container_width=True, hide_index=True)
        
        # Download button
        csv_data = result_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results CSV",
            data=csv_data,
            file_name="batch_results.csv",
            mime="text/csv",
            type="primary"
        )
        
        # === VISUAL INSIGHTS SECTION ===
        st.markdown("---")
        st.markdown("### 📊 Visual Insights")
        
        # Filter successful results for visualization
        viz_df = result_df[result_df['Status'] == 'Success'].copy()
        
        if len(viz_df) > 0:
            chart_col1, chart_col2 = st.columns(2)
            
            # Chart 1: Scatter Plot - Weight vs Energy
            with chart_col1:
                st.markdown("#### Weight vs Energy Correlation")
                st.caption("As molecular weight increases, energy becomes more negative")
                
                # Prepare data for scatter chart
                scatter_data = viz_df[['Mol_Weight', 'AI_Energy_Hartree']].copy()
                scatter_data = scatter_data.rename(columns={
                    'Mol_Weight': 'Molecular Weight (g/mol)',
                    'AI_Energy_Hartree': 'Energy (Hartree)'
                })
                
                st.scatter_chart(
                    scatter_data,
                    x='Molecular Weight (g/mol)',
                    y='Energy (Hartree)',
                    height=350
                )
            
            # Chart 2: Bar Chart - Molecule Complexity (Horizontal)
            with chart_col2:
                st.markdown("#### Molecule Complexity Comparison")
                st.caption("Number of atoms per molecule")
                
                import altair as alt
                
                # Prepare data for horizontal bar chart
                if 'Name' in viz_df.columns:
                    bar_data = viz_df[['Name', 'Num_Atoms']].copy()
                    name_col = 'Name'
                else:
                    bar_data = viz_df[['SMILES', 'Num_Atoms']].copy()
                    bar_data['SMILES'] = bar_data['SMILES'].str[:20] + '...'
                    name_col = 'SMILES'
                
                # Create horizontal bar chart with Altair
                chart = alt.Chart(bar_data).mark_bar(
                    color='#4C78A8',
                    cornerRadiusEnd=4
                ).encode(
                    y=alt.Y(f'{name_col}:N', sort='-x', title='Molecule'),
                    x=alt.X('Num_Atoms:Q', title='Atom Count'),
                    tooltip=[
                        alt.Tooltip(f'{name_col}:N', title='Molecule'),
                        alt.Tooltip('Num_Atoms:Q', title='Atoms')
                    ]
                ).properties(
                    height=350
                ).configure_axis(
                    labelFontSize=12,
                    titleFontSize=13
                )
                
                st.altair_chart(chart, use_container_width=True)
            
            # === DEEP ANALYSIS TAB ===
            st.markdown("---")
            st.markdown("### 📈 Deep Analysis (Drug-Likeness)")
            st.caption("Advanced chemical descriptors for pharmaceutical research")
            
            # Check if deep features exist
            if 'TPSA' in viz_df.columns and 'MolLogP' in viz_df.columns:
                deep_col1, deep_col2 = st.columns(2)
                
                # Chart: MolLogP vs TPSA (Boiled Egg context)
                with deep_col1:
                    st.markdown("#### 🥚 Lipophilicity vs Polarity")
                    st.caption("MolLogP vs TPSA - Key for drug absorption")
                    
                    if 'Name' in viz_df.columns:
                        boiled_data = viz_df[['Name', 'MolLogP', 'TPSA']].copy()
                        lbl = 'Name'
                    else:
                        boiled_data = viz_df[['SMILES', 'MolLogP', 'TPSA']].copy()
                        lbl = 'SMILES'
                    
                    boiled_chart = alt.Chart(boiled_data).mark_circle(size=100).encode(
                        x=alt.X('MolLogP:Q', title='MolLogP'),
                        y=alt.Y('TPSA:Q', title='TPSA'),
                        color=alt.Color('TPSA:Q', scale=alt.Scale(scheme='viridis'), legend=None),
                        tooltip=[lbl, 'MolLogP', 'TPSA']
                    ).properties(height=300).interactive()
                    
                    st.altair_chart(boiled_chart, use_container_width=True)
                
                # QED Distribution
                with deep_col2:
                    st.markdown("#### 💊 Drug-Likeness (QED)")
                    st.caption("QED Score: 0=poor, 1=excellent")
                    
                    if 'Name' in viz_df.columns:
                        qed_data = viz_df[['Name', 'QED']].copy()
                        qed_lbl = 'Name'
                    else:
                        qed_data = viz_df[['SMILES', 'QED']].copy()
                        qed_lbl = 'SMILES'
                    
                    qed_chart = alt.Chart(qed_data).mark_bar(cornerRadiusEnd=4).encode(
                        y=alt.Y(f'{qed_lbl}:N', sort='-x', title='Molecule'),
                        x=alt.X('QED:Q', title='QED', scale=alt.Scale(domain=[0, 1])),
                        color=alt.Color('QED:Q', scale=alt.Scale(scheme='redyellowgreen'), legend=None),
                        tooltip=[qed_lbl, 'QED']
                    ).properties(height=300)
                    
                    st.altair_chart(qed_chart, use_container_width=True)
                
                # Summary metrics
                dm1, dm2, dm3, dm4 = st.columns(4)
                with dm1:
                    st.metric("Avg TPSA", f"{viz_df['TPSA'].mean():.1f} Ų")
                with dm2:
                    st.metric("Avg QED", f"{viz_df['QED'].mean():.3f}")
                with dm3:
                    st.metric("Avg CSP3", f"{viz_df['Frac_CSP3'].mean():.2f}")
                with dm4:
                    st.metric("Avg LogP", f"{viz_df['MolLogP'].mean():.2f}")
            else:
                st.info("Deep features not available. Re-process batch.")
                
        else:
            st.warning("No successful results to visualize.")
        
        # === MOLECULAR SIMILARITY SEARCH ===
        st.markdown("---")
        st.markdown("### 🧬 Molecular Similarity Search")
        st.caption("Find structurally similar molecules using Morgan Fingerprints & Tanimoto Similarity")
        
        # Get successful molecules for similarity search
        sim_df = result_df[result_df['Status'] == 'Success'].copy()
        
        if len(sim_df) >= 2:
            # Get molecule names or SMILES for selection
            if 'Name' in sim_df.columns:
                molecule_options = sim_df['Name'].tolist()
                id_col = 'Name'
            else:
                molecule_options = sim_df['SMILES'].tolist()
                id_col = 'SMILES'
            
            # Controls
            sim_col1, sim_col2 = st.columns([2, 1])
            with sim_col1:
                ref_molecule = st.selectbox(
                    "Select Reference Molecule",
                    options=molecule_options,
                    key="ref_mol_select"
                )
            with sim_col2:
                sim_threshold = st.slider(
                    "Similarity Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.05,
                    key="sim_threshold"
                )
            
            # Calculate similarities
            if ref_molecule:
                with st.spinner("🧬 Calculating molecular fingerprints..."):
                    # Get reference SMILES
                    ref_smiles = sim_df[sim_df[id_col] == ref_molecule]['SMILES'].values[0]
                    
                    # Calculate Morgan fingerprints
                    try:
                        ref_mol = Chem.MolFromSmiles(ref_smiles)
                        ref_fp = AllChem.GetMorganFingerprintAsBitVect(ref_mol, 2, nBits=2048)
                        
                        similarities = []
                        for idx, row in sim_df.iterrows():
                            try:
                                mol = Chem.MolFromSmiles(row['SMILES'])
                                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                                similarity = DataStructs.TanimotoSimilarity(ref_fp, fp)
                                similarities.append(similarity)
                            except:
                                similarities.append(0.0)
                        
                        sim_df['Similarity_Score'] = similarities
                        
                        # Filter by threshold (exclude reference itself for clarity)
                        filtered_df = sim_df[sim_df['Similarity_Score'] >= sim_threshold].copy()
                        filtered_df = filtered_df.sort_values('Similarity_Score', ascending=False)
                        
                        # Display results
                        st.markdown(f"**Found {len(filtered_df)} molecules with similarity ≥ {sim_threshold}**")
                        
                        if len(filtered_df) > 0:
                            # Select columns to display
                            display_cols = [id_col, 'SMILES', 'Similarity_Score', 'AI_Energy_Hartree'] if id_col == 'Name' else ['SMILES', 'Similarity_Score', 'AI_Energy_Hartree']
                            display_df = filtered_df[display_cols].copy()
                            display_df['Similarity_Score'] = display_df['Similarity_Score'].apply(lambda x: f"{x:.3f}")
                            
                            st.dataframe(
                                display_df,
                                use_container_width=True,
                                hide_index=True,
                                column_config={
                                    "Similarity_Score": st.column_config.TextColumn(
                                        "🎯 Similarity",
                                        help="Tanimoto similarity (1.0 = identical)"
                                    )
                                }
                            )
                        else:
                            st.info("No molecules found above the similarity threshold.")
                    
                    except Exception as e:
                        st.error(f"Error calculating similarity: {e}")
        else:
            st.info("Need at least 2 successful molecules for similarity search.")
        
        st.success(f"✅ Batch processing complete! {success_count}/{len(result_df)} molecules processed successfully.")
    
    else:
        st.markdown("---")
        st.info("👈 Upload a CSV file with a 'SMILES' column to begin batch processing")
        
        st.markdown("### 📝 CSV Format Example")
        example_df = pd.DataFrame({
            'Name': ['Benzene', 'Ethanol', 'Aspirin'],
            'SMILES': ['c1ccccc1', 'CCO', 'CC(=O)Oc1ccccc1C(=O)O']
        })
        st.dataframe(example_df, use_container_width=True, hide_index=True)
        
        st.markdown("### ⚡ Features")
        st.markdown("""
        - **Instant Processing**: AI predictions in milliseconds per molecule
        - **Batch Support**: Process hundreds of molecules at once
        - **Feature Extraction**: Get molecular weight, atoms, rings, valence electrons
        - **Export Results**: Download results as CSV for further analysis
        """)


# === FOOTER ===
st.markdown("---")
st.caption("Built with ❤️ using RDKit, Psi4 & Scikit-Learn | QuantumData Factory v2.4")
