"""Render and text-validate a one-page PDF in the project cache."""
from pathlib import Path

from pypdf import PdfReader
from weasyprint import HTML

root = Path(__file__).resolve().parents[1]
out = root / '.cache/pdf_toolchain.pdf'
out.parent.mkdir(exist_ok=True)
HTML(string='<html><body><h1>Agent3 PDF toolchain</h1><p>RNA + ATAC; histology + clinical.</p></body></html>').write_pdf(out)
reader = PdfReader(out)
assert len(reader.pages) == 1
assert 'Agent3 PDF toolchain' in reader.pages[0].extract_text()
print('PDF rendering and text extraction passed:', out)
