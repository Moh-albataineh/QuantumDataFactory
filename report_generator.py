"""
PDF Report Generator for QuantumData Factory
Enterprise-grade research reports with molecular visualization.
"""

import os
import tempfile
from datetime import datetime
from fpdf import FPDF
from rdkit import Chem
from rdkit.Chem import Draw


class PDFReport(FPDF):
    """Professional PDF report generator for molecular analysis."""
    
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)
    
    def header(self):
        """Add header with title and date."""
        # Title
        self.set_font('Arial', 'B', 20)
        self.set_text_color(30, 60, 114)  # Dark blue
        self.cell(0, 10, 'QuantumData Factory v2.0', 0, 1, 'C')
        
        # Subtitle
        self.set_font('Arial', 'I', 12)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, 'Molecular Analysis Research Report', 0, 1, 'C')
        
        # Date
        self.set_font('Arial', '', 10)
        self.cell(0, 6, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1, 'C')
        
        # Separator line
        self.set_draw_color(30, 60, 114)
        self.set_line_width(0.5)
        self.line(10, 35, 200, 35)
        self.ln(10)
    
    def footer(self):
        """Add footer with page numbers."""
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}} | QuantumData Factory - Confidential', 0, 0, 'C')
    
    def add_section_title(self, title):
        """Add a section title."""
        self.set_font('Arial', 'B', 14)
        self.set_text_color(30, 60, 114)
        self.cell(0, 10, title, 0, 1, 'L')
        self.set_text_color(0, 0, 0)
    
    def add_molecule_image(self, mol, smiles):
        """Add 2D molecule structure image."""
        self.add_section_title('Molecular Structure')
        
        # Generate molecule image
        temp_dir = tempfile.gettempdir()
        img_path = os.path.join(temp_dir, 'mol_report.png')
        
        try:
            img = Draw.MolToImage(mol, size=(400, 300))
            img.save(img_path)
            
            # Center the image
            self.image(img_path, x=55, w=100)
            self.ln(5)
            
            # Add SMILES caption
            self.set_font('Arial', 'I', 10)
            self.set_text_color(80, 80, 80)
            self.cell(0, 6, f'SMILES: {smiles}', 0, 1, 'C')
            self.set_text_color(0, 0, 0)
            self.ln(5)
            
            # Cleanup
            if os.path.exists(img_path):
                os.remove(img_path)
        except Exception as e:
            self.set_font('Arial', 'I', 10)
            self.cell(0, 10, f'[Image generation failed: {e}]', 0, 1, 'C')
    
    def add_properties_table(self, properties):
        """Add a table of molecular properties."""
        self.add_section_title('Calculated Properties')
        
        # Table header
        self.set_font('Arial', 'B', 11)
        self.set_fill_color(30, 60, 114)
        self.set_text_color(255, 255, 255)
        self.cell(95, 10, 'Property', 1, 0, 'C', True)
        self.cell(95, 10, 'Value', 1, 1, 'C', True)
        
        # Table rows
        self.set_font('Arial', '', 10)
        self.set_text_color(0, 0, 0)
        
        row_colors = [(245, 245, 245), (255, 255, 255)]
        for i, (key, value) in enumerate(properties.items()):
            color = row_colors[i % 2]
            self.set_fill_color(*color)
            self.cell(95, 8, str(key), 1, 0, 'L', True)
            self.cell(95, 8, str(value), 1, 1, 'R', True)
        
        self.ln(5)
    
    def add_energy_result(self, energy, source):
        """Add energy calculation result."""
        self.add_section_title('Energy Calculation')
        
        self.set_font('Arial', '', 11)
        self.cell(0, 8, f'Calculation Method: {source}', 0, 1)
        
        # Energy in Hartrees
        self.set_font('Arial', 'B', 12)
        self.set_text_color(30, 60, 114)
        self.cell(0, 10, f'Energy: {energy:.6f} Hartree', 0, 1)
        
        # Energy in kcal/mol
        energy_kcal = energy * 627.509
        self.set_font('Arial', '', 11)
        self.set_text_color(0, 0, 0)
        self.cell(0, 8, f'Energy: {energy_kcal:.2f} kcal/mol', 0, 1)
        self.ln(5)
    
    def add_ai_conclusion(self, energy):
        """Add AI-generated conclusion based on energy."""
        self.add_section_title('AI Analysis Conclusion')
        
        # Generate conclusion based on energy value
        if energy < -300:
            stability = "Very High"
            conclusion = "This molecule exhibits exceptional thermodynamic stability, suggesting it is a large molecular system with extensive bonding networks. Such stability indicates potential applications in materials science or pharmaceutical development."
            color = (34, 139, 34)  # Green
        elif energy < -100:
            stability = "High"
            conclusion = "The molecule demonstrates high stability with a significantly negative energy. This suggests a well-structured compound with favorable intramolecular interactions. Suitable for further computational or experimental investigation."
            color = (46, 139, 87)  # Sea green
        elif energy < -50:
            stability = "Moderate"
            conclusion = "This molecule shows moderate stability. The energy profile suggests a balanced structure with standard bonding characteristics. Additional analysis may be needed to assess reactivity."
            color = (255, 165, 0)  # Orange
        else:
            stability = "Low"
            conclusion = "The molecule exhibits relatively low stability. This could indicate a small molecular system or potential reactivity. Consider validation with higher-level quantum calculations."
            color = (220, 20, 60)  # Crimson
        
        # Stability badge
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(*color)
        self.set_text_color(255, 255, 255)
        self.cell(60, 10, f'Stability: {stability}', 0, 1, 'L', True)
        self.ln(3)
        
        # Conclusion text
        self.set_font('Arial', '', 10)
        self.set_text_color(0, 0, 0)
        self.multi_cell(0, 6, conclusion)
        self.ln(5)
    
    def add_disclaimer(self):
        """Add legal disclaimer."""
        self.ln(10)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(128, 128, 128)
        disclaimer = (
            "DISCLAIMER: This report is generated for research purposes only. "
            "The AI predictions and quantum calculations are computational estimates "
            "and should be validated experimentally before use in any critical applications. "
            "QuantumData Factory is not liable for decisions made based on this report."
        )
        self.multi_cell(0, 4, disclaimer)


def generate_pdf_report(molecule_name, smiles, mol, properties, energy, source):
    """
    Generate a complete PDF research report.
    
    Args:
        molecule_name: Name of the molecule
        smiles: SMILES string
        mol: RDKit molecule object
        properties: Dictionary of molecular properties
        energy: Calculated energy in Hartree
        source: Calculation source (AI/Quantum)
    
    Returns:
        bytes: PDF file content
    """
    pdf = PDFReport()
    pdf.alias_nb_pages()
    pdf.add_page()
    
    # Title
    pdf.set_font('Arial', 'B', 16)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 15, f'Analysis Report: {molecule_name}', 0, 1, 'C')
    pdf.ln(5)
    
    # Add sections
    pdf.add_molecule_image(mol, smiles)
    
    # Format properties for table
    formatted_props = {
        'Molecular Weight': f"{properties['Mol_Weight']:.2f} g/mol",
        'Total Atoms': str(properties['Num_Atoms']),
        'Ring Count': str(properties['Num_Rings']),
        'Valence Electrons': str(properties['Num_Valence_Electrons'])
    }
    pdf.add_properties_table(formatted_props)
    
    pdf.add_energy_result(energy, source)
    pdf.add_ai_conclusion(energy)
    pdf.add_disclaimer()
    
    # Return PDF as bytes
    return pdf.output(dest='S').encode('latin-1')


if __name__ == "__main__":
    # Test the report generator
    from molecule_builder import create_molecule
    from feature_extractor import extract_features
    
    smiles = "c1ccccc1"
    mol = create_molecule(smiles)
    features = extract_features(mol)
    
    pdf_bytes = generate_pdf_report(
        molecule_name="Benzene",
        smiles=smiles,
        mol=mol,
        properties=features,
        energy=-232.456789,
        source="AI Prediction"
    )
    
    # Save test PDF
    with open("test_report.pdf", "wb") as f:
        f.write(pdf_bytes)
    
    print("Test report generated: test_report.pdf")
