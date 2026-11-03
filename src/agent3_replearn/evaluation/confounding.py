import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def confounding(train_z, test_z, train_obs, test_obs):
    results = {}
    for col in ['rna_umis', 'atac_fragments', 'tss_fragment_fraction', 'mito_fraction']:
        if col not in train_obs:
            continue
        a = train_obs[col].to_numpy(dtype=float)
        b = test_obs[col].to_numpy(dtype=float)
        if col in {'rna_umis', 'atac_fragments'}:
            a, b = np.log1p(a), np.log1p(b)
        ta, tb = np.isfinite(a), np.isfinite(b)
        results[col] = dict(applicable=False, n_train=int(ta.sum()), n_test=int(tb.sum()),
                            transform='log1p' if col in {'rna_umis', 'atac_fragments'} else 'identity')
        if ta.sum() <= 10 or tb.sum() <= 10:
            results[col]['reason'] = 'Fewer than 11 finite observations in training or evaluation'
            continue
        if np.ptp(a[ta]) == 0 or np.ptp(b[tb]) == 0:
            results[col]['reason'] = 'QC target is constant in training or evaluation; R-squared is not informative'
            continue
        model = make_pipeline(StandardScaler(), Ridge(alpha=10)).fit(train_z[ta], a[ta])
        score = float(r2_score(b[tb], model.predict(test_z[tb]), force_finite=False))
        if not np.isfinite(score):
            results[col]['reason'] = 'R-squared is non-finite; no finite score substituted'
            continue
        results[col].update(applicable=True, test_r_squared=score)
        correlations = [np.corrcoef(test_z[tb, j], b[tb])[0, 1] for j in range(test_z.shape[1])
                        if np.std(test_z[tb, j]) > 1e-8 and np.std(b[tb]) > 1e-8]
        results[col]['maximum_absolute_coordinate_correlation'] = float(np.max(np.abs(correlations))) if correlations else None
    if 'site' in train_obs:
        known = test_obs.site.isin(train_obs.site).to_numpy()
        results['site'] = dict(applicable=False,
                n_known_site_test=int(known.sum()), n_unseen_site_test=int((~known).sum()),
                training_sites=int(train_obs.site.nunique()))
        if train_obs.site.nunique() < 2 or test_obs.loc[known, 'site'].nunique() < 2:
            results['site']['reason'] = 'At least two training and two known evaluation sites are required'
        else:
            model = make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=2000)).fit(train_z, train_obs.site)
            results['site'].update(applicable=True, balanced_accuracy=float(balanced_accuracy_score(
                test_obs.loc[known, 'site'], model.predict(test_z[known]))),
                chance_reference=1 / train_obs.site.nunique(),
                chance_definition='Uniform independent prediction over training-site classes')
    return results

