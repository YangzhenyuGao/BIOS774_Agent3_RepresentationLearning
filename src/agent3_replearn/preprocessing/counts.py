"""Split-aware count QC, RNA HVGs/PCA and paired count exports."""

import joblib
import numpy as np
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ..data.multiome import load_multiome
from ..splits import assert_train_scope
from ..utils import digest, read_json, write_json
from .atac import ATACTransform


def normalized_log(counts):
    sizes = np.asarray(counts.sum(1)).ravel().clip(1)
    x = (sparse.diags(1e4 / sizes) @ counts).tocsr().astype(np.float32)
    x.data = np.log1p(x.data)
    return x


class RNATransform:
    def __init__(self, n_hvg=3000, n_components=32, seed=774):
        self.n_hvg, self.n_components, self.seed = n_hvg, n_components, seed

    def fit(self, counts, gene_names):
        import anndata as ad
        import scanpy as sc

        eligible = np.flatnonzero(np.asarray((counts > 0).sum(0)).ravel() >= 3)
        a = ad.AnnData(normalized_log(counts[:, eligible]))
        a.var_names = np.asarray(gene_names)[eligible]
        a.var_names_make_unique()
        sc.pp.highly_variable_genes(a, n_top_genes=min(self.n_hvg, len(eligible)), flavor="seurat")
        self.selected = eligible[a.var.highly_variable.to_numpy()]
        log = normalized_log(counts)[:, self.selected].toarray()
        self.scaler = StandardScaler().fit(log)
        standardized = np.clip(self.scaler.transform(log), -10, 10)
        self.pca = PCA(
            n_components=min(self.n_components, standardized.shape[0] - 1, standardized.shape[1] - 1),
            random_state=self.seed,
            svd_solver="randomized",
        ).fit(standardized)
        self.latent_scaler = StandardScaler().fit(self.pca.transform(standardized))
        return self

    def model_input(self, counts):
        return np.clip(
            self.scaler.transform(normalized_log(counts)[:, self.selected].toarray()), -10, 10
        ).astype(np.float32)

    def encode(self, counts):
        return self.latent_scaler.transform(self.pca.transform(self.model_input(counts))).astype(np.float32)


def mad_bounds(values, lower_floor=None, upper_ceiling=None, n_mad=4):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    median = np.median(values)
    mad = np.median(np.abs(values - median)) * 1.4826
    lower = median - n_mad * max(mad, 1e-6)
    upper = median + n_mad * max(mad, 1e-6)
    if lower_floor is not None:
        lower = max(lower, lower_floor)
    if upper_ceiling is not None:
        upper = min(upper, upper_ceiling)
    return [float(lower), float(upper)]


def preprocess_counts(config):
    matrices, obs, features = load_multiome(config)
    manifest = read_json(config.out / "splits/split_manifest.json")
    gene_id = next(m.modality_id for m in config.modalities if m.modality_type == "count_matrix_gene")
    peak_id = next(m.modality_id for m in config.modalities if m.modality_type == "count_matrix_peak")
    rna, atac = matrices[gene_id], matrices[peak_id]
    train = obs.index.isin(manifest["groups"]["train"])
    obs["rna_umis"] = np.asarray(rna.sum(1)).ravel()
    obs["rna_genes"] = np.asarray((rna > 0).sum(1)).ravel()
    mito = features[gene_id]["name"].str.upper().str.startswith("MT-").to_numpy()
    obs["mito_fraction"] = np.asarray(rna[:, mito].sum(1)).ravel() / np.maximum(obs.rna_umis, 1)
    obs["atac_counts"] = np.asarray(atac.sum(1)).ravel()
    if "atac_fragments" not in obs:
        obs["atac_fragments"] = obs.atac_counts
    if "atac_TSS_fragments" in obs:
        obs["tss_fragment_fraction"] = obs.atac_TSS_fragments / obs.atac_fragments.clip(lower=1)
        obs["fraction_fragments_in_peaks"] = obs.atac_peak_region_fragments / obs.atac_fragments.clip(lower=1)
    nm = config.preprocessing.get("qc_mad", 4)
    thresholds = {}
    passing = np.ones(len(obs), dtype=bool)
    for c in ["rna_umis", "rna_genes", "atac_fragments"]:
        values = np.log1p(obs[c].to_numpy())
        lo, hi = mad_bounds(values[train], n_mad=nm)
        thresholds[c] = dict(transform="log1p", lower=lo, upper=hi)
        passing &= (values >= lo) & (values <= hi)
    mt = obs.loc[train, "mito_fraction"].to_numpy()
    _, upper = mad_bounds(mt, upper_ceiling=0.2, n_mad=nm)
    upper = max(0.05, upper)
    thresholds["mito_fraction"] = dict(lower=0, upper=upper)
    passing &= obs.mito_fraction.to_numpy() <= upper
    for c in ["tss_fragment_fraction", "fraction_fragments_in_peaks"]:
        if c in obs:
            lo, _ = mad_bounds(obs.loc[train, c], lower_floor=0, n_mad=nm)
            thresholds[c] = dict(lower=lo)
            passing &= obs[c].fillna(0).to_numpy() >= lo
    # Scrublet is diagnostic on the training split. Do not silently propagate a
    # transductive doublet detector or fit it to validation/test cells.
    doublet = dict(
        method="scrublet",
        fit_scope="train",
        exclusion=False,
        reason="Diagnostic only: no validated out-of-sample doublet scoring; QC exclusion uses count/QC thresholds.",
    )
    obs["scrublet_score"] = np.nan
    if config.preprocessing.get("doublet_method") == "scrublet":
        import scrublet as scr

        try:
            detector = scr.Scrublet(rna[train], expected_doublet_rate=0.06, random_state=config.seed)
            scores, predicted = detector.scrub_doublets(n_prin_comps=min(30, int(train.sum()) - 2))
            obs.loc[train, "scrublet_score"] = scores
            doublet.update(
                status="success",
                threshold=float(detector.threshold_),
                predicted_training_doublet_fraction=float(np.mean(predicted)),
                caveat="Residual doublets may remain; scores never used as labels or representation inputs.",
            )
        except Exception as exc:
            doublet.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    obs["qc_pass"] = passing
    obs["split"] = ""
    for key, ids in manifest["groups"].items():
        obs.loc[obs.index.isin(ids), "split"] = key
    fitted_ids = obs.index[train & passing].tolist()
    scope = assert_train_scope(fitted_ids, manifest)
    fit = train & passing
    rna_transform = RNATransform(
        config.preprocessing.get("n_hvg", 3000), config.preprocessing.get("latent_dim", 32), config.seed
    ).fit(rna[fit], features[gene_id]["name"])
    atac_transform = ATACTransform(
        config.preprocessing.get("n_peaks", 30000), config.preprocessing.get("latent_dim", 32), config.seed
    ).fit(atac[fit])
    out = config.out / "preprocess"
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(dict(rna=rna_transform, atac=atac_transform), out / "transforms.joblib")
    obs.to_parquet(out / "cell_qc.parquet")
    for key in manifest["groups"]:
        keep = obs["split"].eq(key).to_numpy() & passing
        directory = out / key
        directory.mkdir(exist_ok=True)
        obs.loc[keep].to_parquet(directory / "obs.parquet")
        sparse.save_npz(directory / "rna_counts.npz", rna[keep][:, rna_transform.selected])
        sparse.save_npz(directory / "atac_counts.npz", atac[keep][:, atac_transform.selected])
        np.save(directory / "rna_pca.npy", rna_transform.encode(rna[keep]))
        np.save(directory / "atac_lsi.npy", atac_transform.encode(atac[keep]))
        sparse.save_npz(directory / "rna_log.npz", normalized_log(rna[keep])[:, rna_transform.selected])
        sparse.save_npz(directory / "atac_tfidf.npz", atac_transform.tfidf(atac[keep]))
    features[gene_id].iloc[rna_transform.selected].to_csv(out / "selected_genes.csv", index=False)
    features[peak_id].iloc[atac_transform.selected].to_csv(out / "selected_peaks.csv", index=False)
    write_json(
        out / "preprocessing_manifest.json",
        dict(
            fit_scope=scope,
            thresholds=thresholds,
            doublet=doublet,
            modalities=[
                dict(
                    modality_id=gene_id,
                    method="Seurat HVGs on training log-normalized RNA; scaled/clipped PCA",
                    retained_features=len(rna_transform.selected),
                ),
                dict(
                    modality_id=peak_id,
                    method="train-frequency peak filter; binary TF-IDF (log1p 1e4 TF*IDF); LSI",
                    retained_features=len(atac_transform.selected),
                    first_lsi_depth_correlation=atac_transform.first_component_depth_correlation,
                    drop_first=atac_transform.drop_first,
                ),
            ],
            cohort_flow=dict(
                before=len(obs),
                after=int(passing.sum()),
                by_split=obs.loc[passing, "split"].value_counts().to_dict(),
                labeled_passing=int((passing & obs.label.ne("filtered")).sum()),
                vignette_filtered_passing=int((passing & obs.label.eq("filtered")).sum()),
                vignette_labeled_rejected=int((~passing & obs.label.ne("filtered")).sum()),
            ),
            tss_measure="TSS fragment fraction proxy; not conventional flanking-normalized TSS enrichment",
            nucleosome_signal="not computed; fragment-length QC not used",
            feature_selection_hash=digest(dict(genes=rna_transform.selected, peaks=atac_transform.selected)),
        ),
    )
    return out
