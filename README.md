# ⚛️ QuantumData Factory v2.4: AI-Powered Deep Chemical Analysis

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![AI](https://img.shields.io/badge/AI-Machine%20Learning-green.svg)
![Science](https://img.shields.io/badge/Science-Quantum%20Chemistry-purple.svg)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)

A **Hybrid AI + Quantum Chemistry** platform for molecular energy prediction, drug-likeness analysis, and interactive 3D visualization.

---

## 🎯 Overview

QuantumData Factory combines the speed of Machine Learning with the precision of Quantum Mechanical calculations. Analyze molecules instantly with AI or validate with high-precision quantum simulations.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🚀 **AI Prediction** | Instant energy prediction using trained ML model |
| ⚛️ **Quantum Engine** | Hartree-Fock calculations via Psi4 |
| 🔮 **3D Visualization** | Interactive molecular viewer with multiple styles |
| 🏭 **Batch Factory** | Process hundreds of molecules via CSV upload |
| 💊 **Drug Discovery Mode** | Auto-calculation of Lipinski Rules, QED, and TPSA |
| 📈 **Deep Visualizations** | Interactive scatter plots for Structure-Property relationships |
| 🧬 **Similarity Search** | Find chemical analogs using Tanimoto Similarity |
| 📄 **PDF Reports** | Enterprise-grade research reports with AI conclusions |

---

## 🛠️ Installation

### Step 1: Create Conda Environment
```bash
conda create -n ddf python=3.9 -y
conda activate ddf
```

### Step 2: Install Quantum Chemistry Packages (via Conda)
```bash
conda install -c conda-forge rdkit psi4 -y
```

### Step 3: Install Python Dependencies (via pip)
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Web Application (Recommended)
```bash
streamlit run app.py
```
Then open http://localhost:8501 in your browser.

### Batch Processing Demo
Upload `batch_sample.csv` in the Batch Factory mode to test with example molecules.

### Command Line - Interactive Prediction
```bash
python predict_new.py
```

### Train New Model
```bash
python train_model.py
```

---

## 📁 Project Structure

```
DigitalDataFactory/
├── app.py                  # Streamlit web application
├── main_factory.py         # Batch processing pipeline
├── molecule_builder.py     # SMILES → 3D molecule conversion
├── energy_calculator.py    # Quantum energy calculations (Psi4)
├── feature_extractor.py    # Molecular feature extraction (Deep)
├── visualization_engine.py # 2D/3D visualization generation
├── report_generator.py     # PDF report generation
├── train_model.py          # ML model training
├── predict_new.py          # CLI predictor
├── batch_sample.csv        # Demo batch file
├── energy_predictor_model.pkl
├── requirements.txt
└── README.md
```

---

## 🧪 Deep Chemical Descriptors

| Descriptor | Purpose |
|------------|---------|
| **TPSA** | Topological Polar Surface Area (drug absorption) |
| **QED** | Quantitative Estimation of Drug-likeness (0-1) |
| **Frac_CSP3** | Carbon saturation (3D complexity) |
| **MolLogP** | Partition coefficient (lipophilicity) |

---

## 🔧 Technologies

- **RDKit** - Molecular structure handling
- **Psi4** - Quantum chemistry engine
- **Scikit-Learn** - Machine learning
- **Streamlit** - Web interface
- **py3Dmol** - 3D molecular visualization
- **Altair** - Interactive charts
- **FPDF** - PDF report generation

---

## 📜 License

MIT License - Feel free to use, modify, and distribute.

---

Built with ❤️ for computational chemistry and AI-driven drug discovery.
