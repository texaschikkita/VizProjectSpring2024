from nbconvert import HTMLExporter, PDFExporter
import nbformat

notebook_filename = "librosa_data_vizDUE_SATURDAY.ipynb"

# Load your Jupyter notebook
with open(notebook_filename) as f:
    notebook_node = nbformat.read(f, as_version=4)

# Convert to HTML
html_exporter = HTMLExporter()
html_data, resources = html_exporter.from_notebook_node(notebook_node)

# Write the HTML data to a file
with open("librosa_data_vizDUE_SATURDAY.ipynb", "w") as f:
    f.write(html_data)

# Convert to PDF (requires LaTeX)
pdf_exporter = PDFExporter()
pdf_data, resources = pdf_exporter.from_notebook_node(notebook_node)

# Write the PDF data to a file
with open("librosa_data_vizDUE_SATURDAY.ipynb", "wb") as f:
    f.write(pdf_data)
