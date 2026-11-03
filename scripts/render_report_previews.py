"""Render every PDF page for visual review; rendering does not assert visual approval."""
import argparse

import fitz

from agent3_replearn.utils import ROOT, checksum, require_compute, write_json

require_compute()
parser = argparse.ArgumentParser()
parser.add_argument('--pdf', nargs='+', default=['reports/generated_report_1.pdf', 'reports/generated_report_2.pdf', 'reports/report.pdf'])
args = parser.parse_args()
records = []
for filename in args.pdf:
    source = ROOT / filename
    directory = ROOT / 'reports/qa_pages' / source.stem
    directory.mkdir(parents=True, exist_ok=True)
    pages = []
    with fitz.open(source) as document:
        for index, page in enumerate(document):
            image = directory / f'page_{index + 1:02}.png'
            page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False).save(image)
            pages.append(dict(page=index + 1, image=str(image), image_sha256=checksum(image),
                              text_characters=len(page.get_text().strip())))
    records.append(dict(pdf=str(source), pdf_sha256=checksum(source), pages=pages))
write_json(ROOT / 'reports/qa_preview_manifest.json',
           dict(reports=records, manual_visual_review='pending; previews alone do not establish visual approval'))
print('Rendered every requested PDF page; manual visual review remains explicit.')
