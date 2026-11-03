"""Agent3's own clinical/cohort table from GDC records and primary-sample Xena rows."""
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..utils import checksum, read_json, write_json

MISSING = {'', 'nan', 'na', 'n/a', 'null', 'none', '[not available]', '[not applicable]',
           'not reported', 'unknown', 'not available', '--'}
TARGETS = {'LumA', 'LumB', 'Basal', 'Her2'}
INPUTS = ['age', 'stage', 'histological_type', 'er', 'pr', 'her2']


def clean(value):
    if value is None or str(value).strip().lower() in MISSING:
        return None
    return str(value).strip()


def numeric(value):
    try:
        value = float(value)
        return value if np.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def consistent(series):
    values = sorted({clean(v) for v in series if clean(v) is not None})
    return (values[0] if len(values) == 1 else None), values


def stage_group(value):
    value = clean(value)
    if value is None:
        return None
    match = re.match(r'^(?:STAGE\s*)?(IV|III|II|I)(?:[ABC]|\b)', value.upper())
    return match.group(1) if match else None


def receptor(value):
    value = clean(value)
    if value is None:
        return 'missing'
    value = value.lower()
    if value in {'positive', 'negative', 'equivocal', 'indeterminate'}:
        return value
    if 'not eval' in value or 'not perform' in value:
        return 'not_evaluated'
    return 'missing'


def primary_diagnosis(case):
    diagnoses = case.get('diagnoses', [])
    explicit = [d for d in diagnoses if d.get('diagnosis_is_primary_disease') is True]
    if explicit:
        chosen = sorted(explicit, key=lambda d: (abs(numeric(d.get('days_to_diagnosis')) or 0),
                                                str(d.get('diagnosis_id', ''))))[0]
        return chosen, 'explicit_primary', len(explicit)
    if len(diagnoses) == 1 and 'diagnosis_is_primary_disease' not in diagnoses[0]:
        return diagnoses[0], 'legacy_single_diagnosis', 1
    return {}, 'unresolved_primary', 0


def endpoint(case, diagnosis, xena_values):
    demographic = case.get('demographic', {})
    vital = clean(demographic.get('vital_status'))
    death = numeric(demographic.get('days_to_death'))
    source = 'GDC demographic days_to_death'
    if death is None:
        death = numeric(diagnosis.get('days_to_death'))
        source = 'GDC primary diagnosis days_to_death'
    if death is None:
        death = numeric(xena_values.get('days_to_death'))
        source = 'Xena primary-sample days_to_death'
    follow = [numeric(diagnosis.get('days_to_last_follow_up'))]
    follow += [numeric(d.get('days_to_follow_up')) for d in case.get('follow_ups', [])]
    follow += [numeric(xena_values.get('days_to_last_followup')),
               numeric(xena_values.get('days_to_last_known_alive'))]
    follow = [v for v in follow if v is not None and v > 0]
    if vital and vital.lower() == 'dead':
        return death, 1, source, bool(death is not None and death > 0)
    if vital and vital.lower() == 'alive':
        return max(follow) if follow else None, 0, 'max GDC/Xena followup; alive', bool(follow)
    return None, None, 'unresolved vital status', False


def build_tables(config):
    """No reads of another project's phenotype or splits; audit is count-only cross-check."""
    wsi = next(m for m in config.modalities if m.modality_type == 'wsi_image')
    clinical = next(m for m in config.modalities if m.modality_type == 'tabular_clinical')
    slides = pd.read_csv(wsi.parameters['manifest'], sep='\t', dtype=str)
    slides['patient_id'] = slides.filename.str.extract(r'(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})', expand=False)
    slides['sample_code'] = slides.filename.str.extract(r'TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-(\d{2})', expand=False)
    slides['slide_id'] = slides['id']
    slides['path'] = [str(Path(wsi.path) / uid / name) for uid, name in zip(slides.id, slides.filename)]
    slides['primary_eligible'] = slides['sample_code'].eq('01') & slides.patient_id.notna()
    slides['exclusion_reason'] = np.where(slides.primary_eligible, '', 'nonprimary_or_invalid_barcode')
    slides['size_matches'] = [Path(p).is_file() and Path(p).stat().st_size == int(n)
                              for p, n in zip(slides.path, slides['size'])]
    if not slides.size_matches.all():
        raise ValueError('Slide manifest missing-file or size mismatch; inspect before proceeding')
    cases = read_json(clinical.path)['cases']
    records = {c['submitter_id']: c for c in cases}
    xena = pd.read_csv(clinical.parameters['xena'], sep='\t', dtype=str, keep_default_na=False)
    xena = xena[xena.sampleID.str.slice(13, 15).eq('01')].copy()
    xena['patient_id'] = xena['_PATIENT'].str.slice(0, 12)
    fields = ['PAM50Call_RNAseq', 'age_at_initial_pathologic_diagnosis', 'pathologic_stage',
              'histological_type', 'breast_carcinoma_estrogen_receptor_status',
              'breast_carcinoma_progesterone_receptor_status',
              'lab_proc_her2_neu_immunohistochemistry_receptor_status',
              'days_to_death', 'days_to_last_followup', 'days_to_last_known_alive']
    by_patient, conflicts = {}, []
    for patient, frame in xena.groupby('patient_id', sort=True):
        values = {}
        for field in fields:
            value, observed = consistent(frame[field])
            values[field] = value
            if len(observed) > 1:
                conflicts.append(dict(patient_id=patient, field=field, conflicting_values=observed,
                                      resolution='set missing rather than choose a favorable value'))
        by_patient[patient] = values
    rows, lineage = [], []
    primary_patients = set(slides.loc[slides.primary_eligible, 'patient_id'])
    for patient in sorted(records):
        case, xv = records[patient], by_patient.get(patient, {})
        diagnosis, policy, nprimary = primary_diagnosis(case)
        age_days = numeric(diagnosis.get('age_at_diagnosis'))
        age = age_days / 365.25 if age_days is not None else numeric(xv.get('age_at_initial_pathologic_diagnosis'))
        stage = stage_group(diagnosis.get('ajcc_pathologic_stage')) or stage_group(xv.get('pathologic_stage'))
        pam50 = xv.get('PAM50Call_RNAseq')
        time, event, os_source, eligible_os = endpoint(case, diagnosis, xv)
        rows.append(dict(patient_id=patient, site=patient.split('-')[1], age=age, stage=stage,
                         histological_type=clean(xv.get('histological_type')),
                         er=receptor(xv.get('breast_carcinoma_estrogen_receptor_status')),
                         pr=receptor(xv.get('breast_carcinoma_progesterone_receptor_status')),
                         her2=receptor(xv.get('lab_proc_her2_neu_immunohistochemistry_receptor_status')),
                         PAM50_raw=pam50, label=pam50 if pam50 in TARGETS else '__unlabeled__',
                         os_days=time, os_event=event, os_eligible=eligible_os,
                         has_primary_slide=patient in primary_patients))
        lineage.append(dict(patient_id=patient, diagnosis_policy=policy, n_primary_diagnoses=nprimary,
                            selected_diagnosis_id=diagnosis.get('diagnosis_id'),
                            age_source='GDC primary diagnosis (days / 365.25)' if age_days is not None else 'Xena',
                            stage_source='GDC primary diagnosis' if stage_group(diagnosis.get('ajcc_pathologic_stage')) else 'Xena',
                            histology_and_receptor_source='Xena primary-sample clinical fields',
                            pam50_source='Xena PAM50Call_RNAseq, sample code 01', os_source=os_source))
    all_clinical = pd.DataFrame(rows).set_index('patient_id')
    if primary_patients - set(all_clinical.index):
        raise ValueError('Primary-slide patients missing from GDC clinical records')
    cohort = all_clinical.loc[sorted(primary_patients)].copy()
    complete = cohort[INPUTS].notna().all(axis=1) & ~cohort[['er', 'pr', 'her2']].eq('missing').any(axis=1)
    flow = dict(manifest_slides=len(slides), primary_slides=int(slides.primary_eligible.sum()),
                excluded_slides=int((~slides.primary_eligible).sum()), primary_slide_patients=len(cohort),
                with_pam50_including_normal=int(cohort.PAM50_raw.notna().sum()),
                normal_like_excluded_primary=int(cohort.PAM50_raw.eq('Normal').sum()),
                labeled_four_class=int(cohort.label.isin(TARGETS).sum()),
                complete_clinical=int(complete.sum()),
                labeled_and_complete=int((complete & cohort.label.isin(TARGETS)).sum()),
                labeled_classes=cohort.loc[cohort.label.isin(TARGETS), 'label'].value_counts().to_dict(),
                all_clinical_pam50=all_clinical.PAM50_raw.dropna().value_counts().to_dict(),
                all_clinical_os_eligible=int(all_clinical.os_eligible.sum()),
                all_clinical_deaths=int(all_clinical.loc[all_clinical.os_eligible, 'os_event'].sum()),
                clinical_missingness={c: int(cohort[c].isna().sum() + cohort[c].eq('missing').sum()) for c in INPUTS})
    audit = read_json(clinical.parameters['audit'])['after_clinical']
    comparison = dict(reference_pam50=audit['pam50'], agent3_pam50=flow['all_clinical_pam50'],
                      reference_os_eligible=audit['os_eligible'], agent3_os_eligible=flow['all_clinical_os_eligible'],
                      reference_deaths=audit['deaths'], agent3_deaths=flow['all_clinical_deaths'],
                      other_project_tables_or_splits_adopted=False)
    out = config.out / 'preprocess'
    out.mkdir(parents=True, exist_ok=True)
    cohort.to_parquet(out / 'clinical_cohort.parquet')
    slides.to_parquet(out / 'slide_inventory.parquet', index=False)
    all_clinical.to_parquet(out / 'all_clinical_crosscheck.parquet')
    write_json(out / 'clinical_lineage.json', dict(records=lineage, conflicts=conflicts,
                                                   input_columns=INPUTS,
                                                   source_checksums={str(p): checksum(p) for p in
                                                                     [Path(clinical.path), Path(clinical.parameters['xena'])]}))
    write_json(out / 'cohort_flow.json', flow)
    write_json(out / 'clinical_audit_crosscheck.json', comparison)
    return cohort, slides, flow
