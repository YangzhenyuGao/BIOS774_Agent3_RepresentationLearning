"""Physical-resolution tiling and resumable coordinate/feature stores."""

import hashlib
import inspect
import json
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu

from ..utils import checksum, code_hash, digest, read_json, require_compute, write_json


def seed_for(identifier, seed):
    return int(hashlib.sha256(f"{seed}:{identifier}".encode()).hexdigest()[:8], 16)


def slide_metadata(slide):
    properties = dict(slide.properties)
    mpp = properties.get("openslide.mpp-x")
    fallback = False
    if mpp is None:
        objective = float(properties.get("openslide.objective-power", 20))
        mpp = 10 / objective
        fallback = True
    return dict(
        dimensions=list(slide.dimensions),
        level_dimensions=[list(v) for v in slide.level_dimensions],
        level_downsamples=list(slide.level_downsamples),
        mpp_x=float(mpp),
        mpp_y=float(properties.get("openslide.mpp-y", mpp)),
        mpp_fallback=fallback,
        objective_power=properties.get("openslide.objective-power"),
        fallback_rule="10/objective power; objective defaults to 20x when metadata absent",
    )


def tissue_coordinates(path, seed, tile_size=224, target_mpp=0.5, cap=256, minimum_fraction=0.6):
    import openslide

    with openslide.OpenSlide(str(path)) as slide:
        metadata = slide_metadata(slide)
        thumb = np.asarray(slide.get_thumbnail((1536, 1536)).convert("RGB"))
        saturation = rgb2hsv(thumb)[:, :, 1]
        cutoff = float(threshold_otsu(saturation)) if np.ptp(saturation) > 0.01 else 0.05
        mask = (saturation > max(0.05, cutoff)) & (thumb.mean(2) < 245)
        mask = ndimage.binary_closing(mask, iterations=2)
        mask = ndimage.binary_fill_holes(mask)
        labels, n = ndimage.label(mask)
        if n:
            sizes = np.bincount(labels.ravel())
            mask = (sizes[labels] >= 16) & (labels > 0)
        width, height = slide.dimensions
        l0_size = max(1, int(round(tile_size * target_mpp / metadata["mpp_x"])))
        level = slide.get_best_level_for_downsample(max(1, target_mpp / metadata["mpp_x"]))
        sx, sy = thumb.shape[1] / width, thumb.shape[0] / height
        integral = np.pad(mask.astype(np.int32), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        candidates = []
        for y in range(0, max(0, height - l0_size + 1), l0_size):
            for x in range(0, max(0, width - l0_size + 1), l0_size):
                x1, x2 = int(x * sx), min(mask.shape[1], max(int((x + l0_size) * sx), int(x * sx) + 1))
                y1, y2 = int(y * sy), min(mask.shape[0], max(int((y + l0_size) * sy), int(y * sy) + 1))
                tissue = integral[y2, x2] - integral[y1, x2] - integral[y2, x1] + integral[y1, x1]
                if tissue / max((x2 - x1) * (y2 - y1), 1) >= minimum_fraction:
                    candidates.append((x, y))
        rng = np.random.default_rng(seed)
        if len(candidates) > cap:
            indices = rng.choice(len(candidates), cap, replace=False)
            # Seeded ordering determines nested smaller tile caps.
            candidates = [candidates[i] for i in indices]
        elif candidates:
            candidates = [candidates[i] for i in rng.permutation(len(candidates))]
        metadata.update(
            tile_size=tile_size,
            target_mpp=target_mpp,
            level=level,
            level0_tile_size=l0_size,
            estimated_tissue_fraction=float(mask.mean()),
            estimated_tissue_area_um2=float(
                mask.mean() * width * height * metadata["mpp_x"] * metadata["mpp_y"]
            ),
            tissue_threshold=cutoff,
            retained_coordinates=len(candidates),
            physical_tile_width_um=l0_size * metadata["mpp_x"],
            native_encoder_resize="fixed physical field resized to each native model input; effective pixel size changes",
        )
        return np.asarray(candidates, dtype=np.int64).reshape(-1, 2), metadata, thumb, mask


def read_tile(slide, coordinate, metadata):
    level = metadata["level"]
    size = max(1, int(round(metadata["level0_tile_size"] / metadata["level_downsamples"][level])))
    return (
        slide.read_region(tuple(map(int, coordinate)), level, (size, size))
        .convert("RGB")
        .resize((metadata["tile_size"], metadata["tile_size"]), Image.Resampling.BILINEAR)
    )


def tile_quality(image, blur_threshold=30):
    a = np.asarray(image)
    hsv = rgb2hsv(a)
    gray = a.mean(2)
    blur = float(ndimage.laplace(gray).var())
    # Pen proxy catches strongly blue/green saturated regions; intentionally conservative.
    pen = ((hsv[:, :, 1] > 0.8) & (hsv[:, :, 0] > 0.32) & (hsv[:, :, 0] < 0.62)).mean()
    tissue = ((hsv[:, :, 1] > 0.05) & (gray < 245)).mean()
    return bool(blur >= blur_threshold and pen < 0.15 and tissue >= 0.4), dict(
        blur_variance=blur, pen_fraction=float(pen), tissue_fraction=float(tissue)
    )


def tile_stage(config):
    require_compute()
    slides = pd.read_parquet(config.out / "preprocess/slide_inventory.parquet")
    slides = slides[slides.primary_eligible]
    out = config.out / "features/coordinates"
    out.mkdir(parents=True, exist_ok=True)
    failures, receipts = [], []
    params = config.preprocessing
    cap = max(params.get("tile_cap_options", [256]))
    coordinate_tool_hash = digest({fn.__name__: inspect.getsource(fn)
                                  for fn in [seed_for, slide_metadata, tissue_coordinates]})
    for row in slides.itertuples():
        path = out / f"{row.slide_id}.npz"
        meta = path.with_suffix(".json")
        signature = digest(
            dict(
                path=row.path,
                size=Path(row.path).stat().st_size,
                mtime=Path(row.path).stat().st_mtime_ns,
                params=params,
                seed=config.seed,
                coordinate_tool_hash=coordinate_tool_hash,
            )
        )
        if path.exists() and meta.exists():
            previous = read_json(meta)
            if previous.get("signature") == signature and previous.get("coordinate_sha256") == checksum(path):
                receipts.append(str(meta))
                continue
        try:
            coords, metadata, thumb, mask = tissue_coordinates(
                row.path,
                seed_for(row.slide_id, config.seed),
                params.get("tile_size", 224),
                params.get("target_mpp", 0.5),
                cap,
                params.get("tissue_fraction_min", 0.6),
            )
            if len(coords) == 0:
                raise ValueError("No tissue coordinates pass thumbnail segmentation")
            np.savez_compressed(path, coordinates=coords)
            write_json(
                meta,
                dict(
                    signature=signature,
                    coordinate_tool_hash=coordinate_tool_hash,
                    slide_id=row.slide_id,
                    patient_id=row.patient_id,
                    path=row.path,
                    coordinate_sha256=checksum(path),
                    **metadata,
                ),
            )
            receipts.append(str(meta))
        except Exception as exc:
            failures.append(
                dict(slide_id=row.slide_id, path=row.path, error_type=type(exc).__name__, error=str(exc))
            )
    write_json(
        config.out / "features/tiling_manifest.json",
        dict(
            slides_total=len(slides),
            slides_success=len(receipts),
            failures=failures,
            coordinate_manifests=receipts,
            concurrent_slide_readers=1,
            production_tile_images_stored=False,
            cap_provisional=cap,
        ),
    )
    if failures:
        raise RuntimeError(f"{len(failures)} slides failed tiling; failures retained for explicit repair")
    return [config.out / "features/tiling_manifest.json"] + [Path(p) for p in receipts] + [Path(p).with_suffix('.npz') for p in receipts]


def load_encoders(config, names):
    import torch
    import torchvision

    from ..models.pathology_fm import PathologyEncoder

    encoders = {}
    for name in names:
        if name == "imagenet_resnet18":
            weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
            model = torchvision.models.resnet18(weights=None)
            weight_path = (
                Path(__import__("os").environ["TORCH_HOME"]) / "hub/checkpoints" / Path(weights.url).name
            )
            if not weight_path.exists():
                raise FileNotFoundError("Fetch ImageNet baseline weights on the login node first")
            model.load_state_dict(torch.load(weight_path, map_location="cpu", weights_only=True), strict=True)
            model.fc = torch.nn.Identity()
            transform = weights.transforms()
            record = dict(
                model_id=name,
                pretrained_source=weights.url,
                license="torchvision BSD-3-Clause; ImageNet research terms",
                weight_sha256=checksum(weight_path),
                embedding_dim=512,
                image_size=224,
                strict_state_dict=True,
            )
            encoders[name] = (model.to("cuda").eval(), transform, record)
        else:
            model = PathologyEncoder(name, config.model_store, verify=True).to("cuda").eval()
            encoders[name] = (model, model.transform, model.record())
    return encoders


def feature_signature(config, slide_id, cap, record):
    path = config.out / 'features/coordinates' / f'{slide_id}.npz'
    return digest(dict(coordinates=checksum(path), cap=cap, encoder=record,
                       blur=config.preprocessing.get('blur_variance_min', 30), analysis_code_hash=code_hash()))


def extract_slide(config, row, encoders, cap, destination):
    import openslide
    import torch

    coords_file = config.out / "features/coordinates" / f"{row.slide_id}.npz"
    metadata = read_json(coords_file.with_suffix(".json"))
    coordinates = np.load(coords_file)["coordinates"][:cap]
    destination.mkdir(parents=True, exist_ok=True)
    pending = {}
    for name, (model, transform, record) in encoders.items():
        path = destination / name / f"{row.slide_id}.h5"
        path.parent.mkdir(parents=True, exist_ok=True)
        signature = feature_signature(config, row.slide_id, cap, record)
        if path.exists():
            with h5py.File(path) as f:
                if (
                    f.attrs.get("signature") == signature
                    and f.attrs.get("complete")
                    and len(f["features"]) > 0
                ):
                    continue
        pending[name] = (path, signature)
    if not pending:
        times = {}
        total_seconds, ntiles = 0., 0
        for name in encoders:
            with h5py.File(destination / name / f'{row.slide_id}.h5') as f:
                times[name] = float(f.attrs['encode_seconds'])
                total_seconds = max(total_seconds, float(f.attrs['total_seconds']))
                ntiles = len(f['features'])
        return dict(slide_id=row.slide_id, status='validated_resume', tiles=ntiles,
                    total_seconds=total_seconds, encode_seconds=times, timing_source='cached validated run')
    values = {name: [] for name in pending}
    kept, qualities = [], []
    started = time.monotonic()
    encode_seconds = {name: 0.0 for name in pending}
    batch_images, batch_coords, batch_qc = [], [], []

    def process_batch():
        if not batch_images:
            return
        for name in pending:
            model, transform, _ = encoders[name]
            tensor = torch.stack([transform(image) for image in batch_images]).to("cuda")
            begin = time.monotonic()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                z = model(tensor)
            torch.cuda.synchronize()
            encode_seconds[name] += time.monotonic() - begin
            z = z.float().cpu().numpy()
            if not np.isfinite(z).all():
                raise FloatingPointError(f"Non-finite {name} tile features")
            values[name].append(z)
        kept.extend(batch_coords)
        qualities.extend(batch_qc)
        batch_images.clear()
        batch_coords.clear()
        batch_qc.clear()

    with openslide.OpenSlide(row.path) as slide:
        for coordinate in coordinates:
            image = read_tile(slide, coordinate, metadata)
            ok, qc = tile_quality(image, config.preprocessing.get("blur_variance_min", 30))
            if not ok:
                continue
            batch_images.append(image)
            batch_coords.append(coordinate)
            batch_qc.append(qc)
            if len(batch_images) >= 16:
                process_batch()
        process_batch()
    if not kept:
        raise ValueError("No tiles passed region QC; slide is not silently dropped")
    for name, (path, signature) in pending.items():
        temporary = path.with_suffix(".partial.h5")
        with h5py.File(temporary, "w") as f:
            f.create_dataset("features", data=np.concatenate(values[name]), compression="lzf")
            f.create_dataset("coordinates", data=np.asarray(kept))
            f.attrs.update(
                signature=signature,
                complete=True,
                slide_id=row.slide_id,
                patient_id=row.patient_id,
                encoder=json.dumps(encoders[name][2]),
                preprocessing=json.dumps(metadata),
                cap=cap,
                encode_seconds=encode_seconds[name],
                total_seconds=time.monotonic() - started,
            )
        temporary.replace(path)
    return dict(
        slide_id=row.slide_id,
        status="success",
        tiles=len(kept),
        attempted=len(coordinates),
        total_seconds=time.monotonic() - started,
        encode_seconds=encode_seconds,
        mean_tile_qc={k: float(np.mean([q[k] for q in qualities])) for k in qualities[0]},
    )


def throughput_pilot(config):
    import torch

    from ..orchestrator import require_smoke

    require_compute()
    require_smoke()
    if not torch.cuda.is_available():
        raise RuntimeError("WSI feature encoding requires the Sycamore H100")
    slides = pd.read_parquet(config.out / "preprocess/slide_inventory.parquet")
    train = read_json(config.out / "splits/split_manifest.json")["groups"]["train"]
    eligible = slides[slides.primary_eligible & slides.patient_id.isin(train)].copy()
    eligible = eligible.sample(n=min(20, len(eligible)), random_state=config.seed)
    # Native FM runtime already covers all four. Throughput compares the declared
    # production shortlist; alternative vision-only/image-text choices stay recorded.
    names = config.encoders
    encoders = load_encoders(config, names)
    results, failures = [], []
    for row in eligible.itertuples():
        try:
            results.append(extract_slide(config, row, encoders, 32, config.out / "features/throughput"))
        except Exception as exc:
            failures.append(dict(slide_id=row.slide_id, error=f"{type(exc).__name__}: {exc}"))
    if failures:
        write_json(config.out / "features/throughput_failures.json", failures)
        raise RuntimeError("Throughput pilot slide failure; repair before fixing a production cap")
    tiles = sum(r.get("tiles", 0) for r in results)
    total = sum(r.get("total_seconds", 0) for r in results)
    rates = {
        name: tiles / max(sum(r.get("encode_seconds", {}).get(name, 0) for r in results), 1e-6)
        for name in names
    }
    all_slides = int(slides.primary_eligible.sum())
    available_hours = 8.0
    selected_cap = min(config.preprocessing.get("tile_cap_options", [64]))
    estimates = {}
    for cap in config.preprocessing.get("tile_cap_options", [64, 128, 256]):
        estimate = 2 * total / max(tiles, 1) * all_slides * cap / 3600
        estimates[str(cap)] = estimate
        if estimate <= available_hours:
            selected_cap = cap
    write_json(
        config.out / "features/encoder_selection.json",
        dict(
            primary_encoder=names[0],
            encoders=names,
            tile_cap=selected_cap,
            throughput_tiles_per_second=rates,
            total_pipeline_tiles_per_second=tiles / max(total, 1e-6),
            measured_slides=len(results),
            training_only=True,
            safety_factor=2,
            extraction_hours_envelope=available_hours,
            projected_hours_by_cap=estimates,
            unpiloted_encoders={
                "Virchow2": "Alternate vision-only model; UNI2-h chosen for native 224px compatibility and smaller output size",
                "MUSK": "Alternate image-text model; CONCH-Z text tower aligned to pathology reports",
            },
            records={name: record for name, (_, _, record) in encoders.items()},
            slides=results,
        ),
    )


def feature_stage(config, shard=0, nshards=1):
    require_compute()
    selection = read_json(config.out / "features/encoder_selection.json")
    slides = pd.read_parquet(config.out / "preprocess/slide_inventory.parquet")
    slides = slides[slides.primary_eligible].sort_values("slide_id").iloc[shard::nshards]
    encoders = None
    rows, failures = [], []
    for row in slides.itertuples():
        try:
            complete = True
            for name in selection['encoders']:
                p = config.out / 'features' / name / f'{row.slide_id}.h5'
                if not p.exists():
                    complete = False
                    break
                with h5py.File(p) as f:
                    if not f.attrs.get('complete') or f.attrs.get('signature') != feature_signature(
                            config, row.slide_id, selection['tile_cap'], selection['records'][name]):
                        complete = False
                        break
            if complete:
                result = dict(slide_id=row.slide_id, status='validated_resume')
            else:
                if encoders is None:
                    encoders = load_encoders(config, selection['encoders'])
                result = extract_slide(config, row, encoders, selection['tile_cap'], config.out / 'features')
            rows.append(result)
            print(result, flush=True)
        except Exception as exc:
            failures.append(dict(slide_id=row.slide_id, error_type=type(exc).__name__, error=str(exc)))
    path = config.out / "features" / f"extraction_shard_{shard}.json"
    write_json(path, dict(shard=shard, nshards=nshards, results=rows, failures=failures))
    if failures:
        raise RuntimeError(f"{len(failures)} feature-extraction failures; resume after explicit repair")
    files = [config.out / 'features' / name / f'{row.slide_id}.h5'
             for row in slides.itertuples() for name in selection['encoders']]
    return [path] + files


def patient_features(config, patient_ids, encoder):
    inventory = pd.read_parquet(config.out / "preprocess/slide_inventory.parquet")
    inventory = inventory[inventory.primary_eligible]
    means, bags, metadata = [], [], []
    for patient in patient_ids:
        slide_means, chunks, coords = [], [], []
        for row in inventory[inventory.patient_id.eq(patient)].itertuples():
            path = config.out / "features" / encoder / f"{row.slide_id}.h5"
            if not path.exists():
                raise FileNotFoundError(f"Unresolved feature-store failure for slide {row.slide_id}")
            with h5py.File(path) as f:
                if not f.attrs.get("complete"):
                    raise ValueError(f"Incomplete slide features {path}")
                z, xy = f["features"][:], f["coordinates"][:]
            chunks.append(z)
            slide_means.append(z.mean(0))
            coords.extend(
                [dict(slide_id=row.slide_id, coordinate=v.tolist(), source_path=row.path) for v in xy]
            )
        if not chunks:
            raise ValueError(f"Patient without eligible slide features: {patient}")
        means.append(np.mean(slide_means, axis=0))  # Equal slide weight in the mean baseline.
        bags.append(np.concatenate(chunks))
        metadata.append(coords)
    return np.asarray(means, dtype=np.float32), bags, metadata


def run_wsi_stage(config, objective, stage):
    if stage == "tile":
        return tile_stage(config)
    if not (config.out / "features/encoder_selection.json").exists():
        throughput_pilot(config)
    return feature_stage(config)


def cache_finetune_tiles(config, data, per_patient=4):
    """Bounded derivative cache; never cache under the shared input directory."""
    import openslide

    if data["split"] == "test":
        from ..data.prepared import require_test_access

        require_test_access(config)
    destination = config.out / "features/finetune_cache"
    destination.mkdir(parents=True, exist_ok=True)
    all_paths = []
    for patient, tile_rows in zip(data["ids"], data["tile_metadata"]):
        paths = []
        for i, row in enumerate(tile_rows[:per_patient]):
            output = destination / f"{patient}_{i}.jpg"
            if not output.exists():
                metadata = read_json(config.out / "features/coordinates" / f"{row['slide_id']}.json")
                with openslide.OpenSlide(row["source_path"]) as slide:
                    read_tile(slide, row["coordinate"], metadata).save(output, quality=95)
            paths.append(str(output))
        all_paths.append(paths)
    data["tile_images"] = all_paths
    write_json(
        destination / f"{data['split']}_manifest.json",
        dict(
            ids=list(data["ids"]),
            per_patient_cap=per_patient,
            images=sum(map(len, all_paths)),
            purpose="bounded actual pretrained tower fine-tuning and inference",
            paths=all_paths,
        ),
    )
    return data
