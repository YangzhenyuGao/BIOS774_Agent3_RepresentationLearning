"""Cheap objective/profile swaps; no representation fitting or held-out metrics."""
import yaml

from agent3_replearn.config import load_config
from agent3_replearn.inspector import read_profile
from agent3_replearn.objective_parser import interpret
from agent3_replearn.planner import applicability, preprocessing_plan
from agent3_replearn.utils import ROOT, write_json

out = ROOT / 'outputs/robustness'
examples = ROOT / 'objectives/examples'
examples.mkdir(exist_ok=True)
rows = []
for config_name, objective_name in [('pbmc_multiome.yaml', 'pbmc_multiome_joint.yaml'),
                                     ('tcga_brca.yaml', 'tcga_brca_pam50.yaml')]:
    cfg = load_config(ROOT / 'config' / config_name)
    profile = read_profile(cfg)
    primary = interpret(ROOT / 'objectives' / objective_name)
    alternate = primary.model_copy(deep=True)
    if profile.pairing.pairing_level == 'cell':
        alternate.objective_id = 'label_informed_structure_plan_demo'
        alternate.original_text = 'Label-informed reference objective: training cell-type labels may be used; plan only.'
        alternate.allowed_supervision = 'train_labels'
        alternate.primary_metrics = ['macro_f1', 'balanced_accuracy']
    else:
        alternate.objective_id = 'survival_primary_plan_demo'
        alternate.original_text = 'Plan a survival-primary representation workflow; outcomes are targets only; plan only.'
        alternate.objective_types = ['survival_risk']
        alternate.target_variable = 'overall_survival'
        alternate.primary_metrics = ['c_index']
        alternate.downstream_tasks = ['survival', 'retrieval']
    (examples / (alternate.objective_id + '.yaml')).write_text(yaml.safe_dump(alternate.model_dump(mode='json'), sort_keys=False))
    before = [r.model_dump() for r in applicability(profile, primary)]
    after = [r.model_dump() for r in applicability(profile, alternate)]
    changes = [dict(model_id=a['model_id'], before_status=a['status'], after_status=b['status'],
                    before_loss=a['training_objective'], after_loss=b['training_objective'])
               for a, b in zip(before, after) if a['status'] != b['status'] or a['training_objective'] != b['training_objective']]
    assert changes
    rows.append(dict(dataset_id=cfg.dataset_id, objective_before=primary.objective_id,
                     objective_after=alternate.objective_id, changes=changes,
                     primary_metrics_before=primary.primary_metrics, primary_metrics_after=alternate.primary_metrics,
                     data_profile_unchanged=True, execution='plan only; no training or held-out evaluation claimed'))
    write_json(out / (alternate.objective_id + '.json'), dict(objective=alternate, applicability=after,
                                                              preprocessing=preprocessing_plan(profile, alternate)))
write_json(out / 'objective_swap_summary.json', rows)
print('Saved two objective-swap demonstrations; no real models trained.')
