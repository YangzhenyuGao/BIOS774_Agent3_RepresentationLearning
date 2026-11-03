"""Check presentation changes on copied accepted fixtures without retraining."""
import shutil

from agent3_replearn import project_report
from agent3_replearn.config import Config
from agent3_replearn.reporting import render_report
from agent3_replearn.schemas import ComputeBudget, ObjectiveSpec
from agent3_replearn.utils import ROOT, checksum, read_json, require_compute, write_json

require_compute()
gate = ROOT / 'logs/smoke_acceptance.json'
original_gate_hash = checksum(gate)
fixtures = []
for kind in ['counts', 'image_bags']:
    source = ROOT / 'outputs/smoke' / kind
    destination = ROOT / 'outputs/report_qa' / kind
    shutil.copytree(source, destination, dirs_exist_ok=True)
    if kind == 'image_bags':
        # Exercise WSI presentation fields with explicitly synthetic metadata only.
        feature = destination / 'features'
        coordinate = feature / 'synthetic_coordinate.json'
        write_json(coordinate, dict(tile_size=128, target_mpp=.5, fixture_only=True))
        write_json(feature / 'tiling_manifest.json', dict(slides_total=1, slides_success=1,
            coordinate_manifests=[str(coordinate)]))
        write_json(feature / 'encoder_selection.json', dict(tile_cap=16, measured_slides=1, safety_factor=2,
            projected_hours_by_cap={'16': .125}, extraction_hours_envelope=8.,
            throughput_tiles_per_second={'synthetic_render_fixture': 10.},
            operational_constraints=dict(original_budget_only_tile_cap=32, array_parallel=1,
                startup_seconds_per_task=90, wall_hours_per_task=2., partition='synthetic_render_fixture')))
    config = Config(dataset_id='report_qa_' + kind, modalities=[], output=str(destination.relative_to(ROOT)),
                    report_name='qa_' + kind, smoke=True,
                    budget=ComputeBudget(pilot_epochs=3, final_epochs=4, pilot_seconds=60, seeds=[774]))
    objective = ObjectiveSpec.model_validate(read_json(destination / 'objective_spec.json'))
    render_report(config, objective)
    fixtures.append(config)
    print(kind, read_json(destination / 'evidence_bundle/pdf_validation.json'), flush=True)
assert checksum(gate) == original_gate_hash
layout_root = ROOT / 'outputs/report_qa/project_layout'
shutil.copytree(ROOT / 'templates', layout_root / 'templates', dirs_exist_ok=True)
(layout_root / 'docs').mkdir(exist_ok=True)
shutil.copy2(ROOT / 'docs/REFERENCES.json', layout_root / 'docs/REFERENCES.json')
shutil.copytree(ROOT / 'outputs/robustness', layout_root / 'outputs/robustness', dirs_exist_ok=True)
template = layout_root / 'templates/project_report.html.j2'
template.write_text(template.read_text().replace('<h1>', '<h1>SYNTHETIC LAYOUT FIXTURE: ', 1))
original_root, original_loader = project_report.ROOT, project_report.load_config
try:
    project_report.ROOT = layout_root
    mapping = dict(zip(['pbmc_multiome.yaml', 'tcga_brca.yaml'], fixtures))
    project_report.load_config = lambda path: mapping[path.name]
    project_report.build_project_report()
    print('Synthetic project layout:', read_json(layout_root / 'reports/project_report_validation.json'), flush=True)
finally:
    project_report.ROOT, project_report.load_config = original_root, original_loader
print('Report-only checks passed; original smoke gate and accepted fixtures were preserved.')
