# ⚛️ QuantumData Factory v1.0

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![AI](https://img.shields.io/badge/AI-Machine%20Learning-green.svg)
![Science](https://img.shields.io/badge/Science-Quantum%20Chemistry-purple.svg)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)

A **Hybrid AI + Quantum Chemistry** platform for molecular energy prediction with interactive 3D visualization.

---

## 🎯 Overview

QuantumData Factory combines the speed of Machine Learning with the precision of Quantum Mechanical calculations. Enter any molecule as a SMILES string and get:

- **Instant AI predictions** trained on quantum data
- **High-precision Quantum simulations** using Hartree-Fock theory
- **Interactive 3D molecular visualization**
- **Feature extraction** for ML applications

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🚀 **AI Prediction** | Instant energy prediction using trained ML model |
| ⚛️ **Quantum Engine** | Hartree-Fock calculations via Psi4 |
| 🔮 **3D Visualization** | Interactive molecular viewer with py3Dmol |
| 🔄 **Hybrid Mode** | Compare AI vs Quantum results side-by-side |
| 📊 **Feature Extraction** | Molecular weight, atoms, rings, valence electrons |
| 📁 **Batch Processing** | Process multiple molecules from CSV |

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

### Command Line - Batch Processing
```bash
python main_factory.py
```

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
├── app.py                 # Streamlit web application
├── main_factory.py        # Batch processing pipeline
├── molecule_builder.py    # SMILES to 3D molecule conversion
├── energy_calculator.py   # Quantum energy calculations (Psi4)
├── feature_extractor.py   # Molecular feature extraction
├── visualization_engine.py # 2D/3D visualization generation
├── train_model.py         # ML model training
├── predict_new.py         # Interactive CLI predictor
├── setup_data.py          # Input data initialization
├── energy_predictor_model.pkl  # Trained ML model
├── inputs/
│   └── chemicals_list.csv # Input molecules
├── factory_output/
│   ├── 2D_Images/         # Generated PNG images
│   ├── 3D_Structures/     # Generated XYZ files
│   └── reports/           # CSV reports
└── requirements.txt       # Python dependencies
```

---

## 🧪 Example Molecules

| Name | SMILES |
|------|--------|
| Benzene | `c1ccccc1` |
| Ethanol | `CCO` |
| Aspirin | `CC(=O)Oc1ccccc1C(=O)O` |
| Caffeine | `Cn1cnc2c1c(=O)n(c(=O)n2C)C` |
| Phenol | `c1ccccc1O` |

---

## 📊 Model Performance

- **R² Score**: 99.97%
- **RMSE**: < 1 Hartree
- **Training Data**: 19 molecules with quantum-calculated energies

---

## 🔧 Technologies

- **RDKit** - Molecular structure handling
- **Psi4** - Quantum chemistry engine
- **Scikit-Learn** - Machine learning
- **Streamlit** - Web interface
- **py3Dmol** - 3D molecular visualization

---

## 📜 License

MIT License - Feel free to use, modify, and distribute.

---
---
## 👤 Author

**Mohammed Albataineh**
* 🐱 GitHub: [@Moh-albataineh](https://github.com/Moh-albataineh)
* 📧 Email: (hmoodx2006xbatayneh@gmail.com)

Built with ❤️ for Science & AI.
