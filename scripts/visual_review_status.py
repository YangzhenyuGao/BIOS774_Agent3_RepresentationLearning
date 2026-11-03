"""Verify an explicit manual inspection record against the actual final PDFs/pages."""
from pathlib import Path

from agent3_replearn.utils import checksum, read_json


def visual_review_status(root):
    preview = read_json(root / 'reports/qa_preview_manifest.json')
    proof = root / 'reports/visual_review.json'
    reviews = read_json(proof)['reviews'] if proof.is_file() else []
    reports = []
    for report in preview['reports']:
        expected = [(page['page'], page['image_sha256']) for page in report['pages']]
        actual = (checksum(report['pdf']) == report['pdf_sha256']
                  and all(Path(page['image']).is_file() and checksum(page['image']) == page['image_sha256']
                          for page in report['pages']))
        matching = [review for review in reviews
                    if review.get('review_status') == 'completed_for_this_pdf_hash'
                    and review['pdf_sha256'] == report['pdf_sha256']
                    and [(page['page'], page['image_sha256']) for page in review['pages']] == expected]
        reports.append(dict(pdf=report['pdf'], pdf_sha256=report['pdf_sha256'],
                            passed=bool(actual and matching),
                            reviewed_utc=matching[-1]['reviewed_utc'] if matching else None))
    return dict(passed=len(reports) == 3 and all(report['passed'] for report in reports),
                manual_review_record_sha256=checksum(proof) if proof.is_file() else None,
                reports=reports, scope='Checks existing human/agent inspection records; never grants visual approval automatically.')
