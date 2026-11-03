"""Exploratory penalized Cox with train-defined risk groups and held-out C-index."""
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def survival(train_z, test_z, train_obs, test_obs, output, bootstrap_replicates=500):
    import matplotlib.pyplot as plt
    tr = train_obs.os_eligible.to_numpy() & train_obs.os_days.gt(0).to_numpy()
    te = test_obs.os_eligible.to_numpy() & test_obs.os_days.gt(0).to_numpy()
    if train_obs.loc[tr, 'os_event'].sum() < 10 or test_obs.loc[te, 'os_event'].sum() < 2:
        return dict(status='insufficient_events', training_events=int(train_obs.loc[tr, 'os_event'].sum()),
                    test_events=int(test_obs.loc[te, 'os_event'].sum()))
    scaler = StandardScaler().fit(train_z[tr])
    reducer = PCA(n_components=min(8, train_z.shape[1], int(tr.sum()) - 1), random_state=774).fit(scaler.transform(train_z[tr]))
    a, b = reducer.transform(scaler.transform(train_z[tr])), reducer.transform(scaler.transform(test_z[te]))
    columns = [f'z{i}' for i in range(a.shape[1])]
    frame = pd.DataFrame(a, columns=columns)
    frame['time'] = train_obs.loc[tr, 'os_days'].to_numpy()
    frame['event'] = train_obs.loc[tr, 'os_event'].to_numpy()
    model = CoxPHFitter(penalizer=1.).fit(frame, 'time', 'event')
    risk = model.predict_partial_hazard(pd.DataFrame(b, columns=columns)).to_numpy()
    train_risk = model.predict_partial_hazard(frame[columns]).to_numpy()
    cutoff = float(np.median(train_risk))
    times, events = test_obs.loc[te, 'os_days'].to_numpy(), test_obs.loc[te, 'os_event'].to_numpy()
    cindex = float(concordance_index(times, -risk, events))
    rng, draws = np.random.default_rng(774), []
    for _ in range(bootstrap_replicates):
        ix = rng.integers(0, len(times), len(times))
        try:
            draws.append(concordance_index(times[ix], -risk[ix], events[ix]))
        except ZeroDivisionError:
            pass
    output.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for label, mask in [('Low risk', risk <= cutoff), ('High risk', risk > cutoff)]:
        if mask.any():
            KaplanMeierFitter().fit(times[mask], events[mask], label=label).plot_survival_function(ax=ax, ci_show=True)
    ax.set(xlabel='Days from diagnosis', ylabel='Overall survival', title='Exploratory held-out risk groups')
    fig.tight_layout()
    fig.savefig(output / 'survival_km.png', dpi=160)
    plt.close(fig)
    pd.DataFrame(dict(patient_id=test_obs.index[te], time=times, event=events, risk=risk,
                      risk_group=np.where(risk > cutoff, 'high', 'low'))).to_csv(output / 'risk_predictions.csv', index=False)
    return dict(status='success', c_index=cindex, lower=float(np.quantile(draws, .025)) if draws else None,
                upper=float(np.quantile(draws, .975)) if draws else None,
                training_events=int(frame.event.sum()), test_events=int(events.sum()),
                test_patients=len(times), risk_threshold_train_median=cutoff,
                penalizer=1., embedding_pca_dimensions=len(columns),
                caveat='Retrospective exploratory survival analysis with few held-out events; no clinical claim')
