"""Common callable adapter contract and explicit unsupported capabilities."""
from pathlib import Path

import joblib


class CapabilityError(NotImplementedError):
    pass


class RepresentationAdapter:
    @property
    def family(self):
        from ..registry import REGISTRY
        return REGISTRY[self.model_id].family

    def assess_applicability(self, data_profile, objective_spec, budget):
        from ..planner import applicability
        return next(r for r in applicability(data_profile, objective_spec) if r.model_id == self.model_id)

    def estimate_cost(self, data_profile, params):
        import math
        n = data_profile.pairing.n_paired
        return dict(dense_input_bytes_upper_bound=sum(m.n_samples * m.n_features * 4 for m in data_profile.modalities),
                    optimization_steps_upper_bound=math.ceil(n / params.get('batch_size', 128)) * params.get('epochs', 50),
                    fit_seconds_cap=params.get('seconds', 600),
                    calibration='WSI throughput measured separately; this is a planning bound, not measured runtime')

    def capabilities(self):
        return dict(new_sample_encoding=True, cross_modal_prediction=False,
                    attention=False, missing_modality=False, supervised=False)

    def validate_input(self, train_data, val_data):
        if not len(train_data['ids']) or not len(val_data['ids']):
            raise ValueError('Training and validation splits must be nonempty')
        if set(train_data['ids']) & set(val_data['ids']):
            raise ValueError('Training/validation ID overlap')

    def predict_cross_modal(self, data, source, target):
        raise CapabilityError('No native cross-modal decoder; evaluate a separately named train-fitted probe')

    def predict(self, data):
        raise CapabilityError('Adapter has no supervised head')

    def attention(self, data):
        raise CapabilityError('Adapter has no attention mechanism')

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path):
        return joblib.load(path)

