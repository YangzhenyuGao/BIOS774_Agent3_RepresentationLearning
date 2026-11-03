"""scvi-tools MULTIVI with raw count likelihoods and explicit external splits."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from ..utils import seed_all
from .base import RepresentationAdapter


def validation_checkpoint(patience):
    import copy
    from lightning.pytorch.callbacks import Callback

    class BestValidation(Callback):
        def __init__(self):
            self.best, self.state, self.stale = float('inf'), None, 0

        def on_validation_epoch_end(self, trainer, pl_module):
            if trainer.sanity_checking:
                return
            metric = trainer.callback_metrics.get('reconstruction_loss_validation')
            if metric is None:
                return
            value = float(metric)
            if value < self.best:
                self.best, self.state, self.stale = value, copy.deepcopy(pl_module.module.state_dict()), 0
            else:
                self.stale += 1
            if self.stale >= patience:
                trainer.should_stop = True

        def on_train_end(self, trainer, pl_module):
            if self.state is not None:
                pl_module.module.load_state_dict(self.state, strict=True)

    return BestValidation()


class MultiVIAdapter(RepresentationAdapter):
    def __init__(self, params):
        self.model_id, self.params = 'multivi', params
        self.history = []

    def capabilities(self):
        return dict(new_sample_encoding=True, cross_modal_prediction=True,
                    attention=False, missing_modality=True, supervised=False)

    def adata(self, data, source=None):
        import anndata as ad
        import mudata as md
        a, b = data['counts']
        if source == 0:
            b = sparse.csr_matrix(b.shape, dtype=np.float32)
        elif source == 1:
            a = sparse.csr_matrix(a.shape, dtype=np.float32)
        # Accessibility likelihood is Bernoulli; expression retains integer counts.
        x = sparse.hstack([a, (b > 0).astype(np.float32)], format='csr')
        if x.data.size and (np.any(x.data < 0) or not np.equal(x.data, np.floor(x.data)).all()):
            raise ValueError('MULTIVI requires nonnegative raw integer counts')
        obs = pd.DataFrame(index=np.asarray(data['ids'], dtype=str))
        rna = ad.AnnData(a.tocsr(), obs=obs.copy(), var=pd.DataFrame(index=[f'gene_{i}' for i in range(a.shape[1])]))
        atac = ad.AnnData((b > 0).astype(np.float32).tocsr(), obs=obs.copy(),
                         var=pd.DataFrame(index=[f'peak_{i}' for i in range(b.shape[1])]))
        result = md.MuData({'rna': rna, 'atac': atac})
        result.obs['_indices'] = np.arange(len(obs))
        return result

    def fit(self, train_data, val_data, params=None, seed=774):
        import anndata as ad
        import mudata as md
        import scvi
        from lightning.pytorch.callbacks import Timer
        self.validate_input(train_data, val_data)
        seed_all(seed)
        scvi.settings.seed = seed
        self.params.update(params or {})
        p = self.params
        train, val = self.adata(train_data), self.adata(val_data)
        a = md.MuData({m: ad.concat([train.mod[m], val.mod[m]], join='inner', merge='first')
                       for m in ['rna', 'atac']})
        scvi.model.MULTIVI.setup_mudata(a, modalities={'rna_layer': 'rna', 'atac_layer': 'atac'})
        self.model = scvi.model.MULTIVI(a, n_genes=train_data['counts'][0].shape[1],
                                       n_regions=train_data['counts'][1].shape[1],
                                       n_latent=p.get('latent_dim', 32), n_hidden=p.get('hidden', 128),
                                       n_layers_encoder=2, n_layers_decoder=2,
                                       gene_likelihood='nb', fully_paired=False,
                                       modality_weights='equal')
        nt, nv = len(train_data['ids']), len(val_data['ids'])
        self.model.train(max_epochs=p.get('epochs', 50), batch_size=p.get('batch_size', 128),
                         accelerator='auto', devices=1, early_stopping=False, check_val_every_n_epoch=1,
                         n_epochs_kl_warmup=max(1, min(50, p.get('epochs', 50) // 2)),
                         datasplitter_kwargs={'external_indexing': [np.arange(nt), np.arange(nt, nt + nv), np.array([], dtype=int)]},
                         plan_kwargs={'lr': p.get('lr', 1e-3)},
                         callbacks=[Timer(duration=dict(seconds=p.get('seconds', 600))),
                                    validation_checkpoint(p.get('patience', 10))],
                         enable_progress_bar=False, logger=False, enable_checkpointing=False)
        self.history = [{key: float(frame.iloc[i, 0]) for key, frame in self.model.history.items()
                         if len(frame) > i and frame.shape[1]} for i in range(
                             max((len(v) for v in self.model.history.values()), default=0))]
        self.fit_ids = list(train_data['ids'])
        self.validation_ids = list(val_data['ids'])
        self.parameter_count = sum(v.numel() for v in self.model.module.parameters())
        return self

    def encode(self, data, modality=None):
        a = self.adata(data)
        return self.model.get_latent_representation(a, modality='joint' if modality is None else
                                                     ['expression', 'accessibility'][int(modality)])

    def predict_cross_modal(self, data, source, target):
        a = self.adata(data, source=int(source))
        if int(target) == 1:
            return self.model.get_normalized_accessibility(a, return_numpy=True)
        return self.model.get_normalized_expression(a, return_numpy=True)

    def save(self, path):
        # Official checkpoint serialization avoids pickling a Lightning trainer.
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path.with_suffix('')), overwrite=True, save_anndata=True)
        model = self.model
        del self.model
        try:
            super().save(path)
        finally:
            self.model = model

    @classmethod
    def load(cls, path):
        import scvi
        instance = super().load(path)
        instance.model = scvi.model.MULTIVI.load(str(Path(path).with_suffix('')), accelerator='auto', device='auto')
        return instance

