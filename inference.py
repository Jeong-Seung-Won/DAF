import argparse
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast


from utils import (
    pt_to_dataset, pt_to_tensor,
    GLOBAL_MEAN, GLOBAL_STD,
    BATCH_SIZE, NUM_WORKERS, DEVICE,
    SEQ_LEN, PRED_LEN, STRIDE,
)
from config import MODEL_ZOO, CFG_REGISTRY

DISPLAY_NAME: Dict[str, str] = {
    "ours_tt":          "DAF (ours)"
}

def _num_windows_of_pt(fp: str) -> int:
    ts = pt_to_tensor(fp)  # (T, D)
    T = ts.shape[0]
    n = (T - SEQ_LEN - PRED_LEN) // STRIDE + 1
    return int(n)

def _build_single_loader_from_pt(fp: str) -> Optional[DataLoader]:
    n_windows = _num_windows_of_pt(fp)
    if n_windows <= 0:
        return None

    ds = pt_to_dataset(fp, GLOBAL_MEAN, GLOBAL_STD)
    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
        num_workers=NUM_WORKERS,
    )


def _load_model_for_eval(model_key: str, ckpt_path: str) -> nn.Module:
    cfg = CFG_REGISTRY[model_key]
    model_cls = MODEL_ZOO[model_key]
    model = model_cls(cfg).to(DEVICE)

    state_dict = torch.load(ckpt_path, map_location=DEVICE)
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()
    return model


def _load_models_for_seeds(model_key: str,
                           seeds: List[int],
                           ckpt_tmpl: str) -> List[Tuple[int, nn.Module]]:
    loaded: List[Tuple[int, nn.Module]] = []
    for s in seeds:
        ckpt_path = ckpt_tmpl.format(model=model_key, seed=s)
        if not Path(ckpt_path).exists():
            print(f"[{model_key}] checkpoint not found: {ckpt_path} (skip seed={s})")
            continue
        try:
            m = _load_model_for_eval(model_key, ckpt_path)
            loaded.append((s, m))
            print(f"[{model_key}] loaded: {ckpt_path}")
        except Exception as e:
            print(f"[{model_key}] failed to load {ckpt_path}: {e}")
    return loaded


def _eval_epoch_norm_flexible(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    criterion_mae: nn.Module,
) -> Tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    mae_sum  = 0.0
    n = len(loader)
    if n == 0:
        return float("nan"), float("nan")

    amp_device_type = "cuda" if torch.cuda.is_available() and str(DEVICE).startswith("cuda") else "cpu"
    amp_dtype = torch.float16 if amp_device_type == "cuda" else torch.bfloat16

    for batch in loader:
        if isinstance(batch, (list, tuple)):
            if len(batch) < 2:
                raise ValueError("Batch must contain at least (x, y).")
            x, y = batch[0], batch[1]
        else:
            raise ValueError(f"Unexpected batch type: {type(batch)}. Expected a (x, y, ...) tuple/list.")

        x, y = x.to(DEVICE), y.to(DEVICE)
        with autocast(device_type=amp_device_type, dtype=amp_dtype):
            y_hat = model(x)
            loss = criterion(y_hat, y)
            mae  = criterion_mae(y_hat, y)

        loss_sum += float(loss.item())
        mae_sum  += float(mae.item())

    return loss_sum / n, mae_sum / n


@torch.no_grad()
def _eval_one_satellite_norm(
    loader: Optional[DataLoader],
    model: nn.Module,
    criterion: nn.Module,
    criterion_mae: nn.Module,
) -> Optional[Tuple[float, float]]:
    if loader is None:
        return None
    mse, mae = _eval_epoch_norm_flexible(model, loader, criterion, criterion_mae)
    return float(mse), float(mae)

def evaluate_constellation_one_model(
    meta_csv: str,
    group_name: str,
    model_key: str,
    seeds: List[int],
    ckpt_tmpl: str,
    out_dir: str,
    criterion: nn.Module,
    criterion_mae: nn.Module,
) -> Optional[str]:

    df_meta = pd.read_csv(meta_csv)
    assert "file_path" in df_meta.columns, f"[{group_name}] meta CSV need 'file_path' column: {meta_csv}"

    file_ids = df_meta["file_id"].tolist() if "file_id" in df_meta.columns else [None] * len(df_meta)
    file_paths = df_meta["file_path"].tolist()

    models_seed: List[Tuple[int, nn.Module]] = _load_models_for_seeds(model_key, seeds, ckpt_tmpl)
    if len(models_seed) == 0:
        print(f"[{group_name} | {model_key}] no checkpoints loaded. skip group.")
        return None

    loader_cache: Dict[str, Optional[DataLoader]] = {}

    rows = []
    skipped = 0
    for fid, fp in zip(file_ids, file_paths):
        if fp not in loader_cache:
            loader_cache[fp] = _build_single_loader_from_pt(fp)
        loader = loader_cache[fp]

        results_mse: List[float] = []
        results_mae: List[float] = []
        for s, model in models_seed:
            try:
                r = _eval_one_satellite_norm(loader, model, criterion, criterion_mae)
            except Exception as e:
                print(f"[{group_name} | {model_key}] skip seed={s}: {fp}\n  → {e}")
                r = None
            if r is None:
                continue
            mse, mae = r
            results_mse.append(mse)
            results_mae.append(mae)

        if len(results_mse) == 0:
            skipped += 1
            print(f"[{group_name} | {model_key}] Skip (windows=0 or all seeds failed): {fp}")
            continue

        mse_mean = float(pd.Series(results_mse).mean())
        mae_mean = float(pd.Series(results_mae).mean())
        mse_std  = float(pd.Series(results_mse).std(ddof=1)) if len(results_mse) > 1 else float("nan")
        mae_std  = float(pd.Series(results_mae).std(ddof=1)) if len(results_mae) > 1 else float("nan")

        rows.append({
            "file_id": fid,
            "file_path": fp,
            "mse_mean": mse_mean,
            "mse_std":  mse_std,
            "mae_mean": mae_mean,
            "mae_std":  mae_std,
        })

    out_df = pd.DataFrame(rows)
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    out_path = str(out_dir_path / f"{group_name}_metrics_{model_key}_norm.csv")
    out_df.to_csv(out_path, index=False)

    print(f"Saved [{group_name} | {model_key}]: {out_path} | satellites={len(rows)}, skipped={skipped}")
    return out_path

def evaluate_constellation_merge_models_numeric(
    meta_csv: str,
    group_name: str,
    model_keys: List[str],
    seeds: List[int],
    ckpt_tmpl: str,
    out_dir: str,
    float_fmt: Optional[str] = None,  
) -> Optional[str]:
    df_meta = pd.read_csv(meta_csv)
    assert "file_path" in df_meta.columns, f"[{group_name}] meta CSV need 'file_path' column: {meta_csv}"
    file_ids = df_meta["file_id"].tolist() if "file_id" in df_meta.columns else [None] * len(df_meta)
    file_paths = df_meta["file_path"].tolist()

    criterion = nn.MSELoss()
    criterion_mae = nn.L1Loss()

    models_by_key: Dict[str, List[Tuple[int, nn.Module]]] = {}
    for mk in model_keys:
        if mk not in CFG_REGISTRY:
            print(f"[{group_name}] Unknown model_key: {mk} (skip)")
            continue
        models_by_key[mk] = _load_models_for_seeds(mk, seeds, ckpt_tmpl)

    loader_cache: Dict[str, Optional[DataLoader]] = {}

    fid_keys: List[str] = []
    for fid, fp in zip(file_ids, file_paths):
        fid_keys.append(str(fid) if fid is not None else Path(fp).stem)

    metrics = ["MSE", "MAE"]
    stats   = ["mean", "std"]
    columns = [f"{fid}_{metric}_{stat}" for fid in fid_keys for metric in metrics for stat in stats]

    row_labels = [DISPLAY_NAME.get(mk, mk) for mk in model_keys]
    table = pd.DataFrame(index=row_labels, columns=columns, dtype=float)
    table.index.name = "model"

    for fid, fp in zip(file_ids, file_paths):
        if fp not in loader_cache:
            loader_cache[fp] = _build_single_loader_from_pt(fp)
        loader = loader_cache[fp]
        fid_key = str(fid) if fid is not None else Path(fp).stem

        for mk in model_keys:
            model_pack = models_by_key.get(mk, [])
            label = DISPLAY_NAME.get(mk, mk)

            if len(model_pack) == 0:
                continue

            mses: List[float] = []
            maes: List[float] = []
            for s, model in model_pack:
                try:
                    r = _eval_one_satellite_norm(loader, model, criterion, criterion_mae)
                except Exception as e:
                    print(f"[{group_name} | {mk}] skip seed={s}: {fp}\n  → {e}")
                    r = None
                if r is None:
                    continue
                mse, mae = r
                mses.append(mse); maes.append(mae)

            if len(mses) == 0:   
                continue

            mse_mean = float(pd.Series(mses).mean())
            mae_mean = float(pd.Series(maes).mean())
            mse_std  = float(pd.Series(mses).std(ddof=1)) if len(mses) > 1 else float("nan")
            mae_std  = float(pd.Series(maes).std(ddof=1)) if len(maes) > 1 else float("nan")

            table.loc[label, f"{fid_key}_MSE_mean"] = mse_mean
            table.loc[label, f"{fid_key}_MSE_std"]  = mse_std
            table.loc[label, f"{fid_key}_MAE_mean"] = mae_mean
            table.loc[label, f"{fid_key}_MAE_std"]  = mae_std

    out_dir_path = Path(out_dir); out_dir_path.mkdir(parents=True, exist_ok=True)
    out_path = str(out_dir_path / f"{group_name}_merged_norm_numeric.csv")
    if float_fmt:
        table.to_csv(out_path, float_format=float_fmt)
    else:
        table.to_csv(out_path)
    print(f"Saved numeric merged [{group_name}]: {out_path} | models={len(model_keys)} | files={len(fid_keys)}")
    print(f"[DEBUG] table shape: {table.shape}")
    print(f"[DEBUG] first 8 columns: {list(table.columns)[:8]}")
    print(f"[DEBUG] index (models): {list(table.index)}")
    return out_path

def _format_pm(mean: float, std: float, sig: int = 6) -> str:
    m = "nan" if pd.isna(mean) else f"{mean:.{sig}g}"
    s = "nan" if pd.isna(std)  else f"{std:.{sig}g}"
    return f"{m} ± {s}"

def evaluate_constellation_merge_models_pretty(
    meta_csv: str,
    group_name: str,
    model_keys: List[str],
    seeds: List[int],
    ckpt_tmpl: str,
    out_dir: str,
    sig: int = 6,
) -> Optional[str]:
    df_meta = pd.read_csv(meta_csv)
    assert "file_path" in df_meta.columns, f"[{group_name}] meta CSV needs 'file_path' column: {meta_csv}"
    file_ids = df_meta["file_id"].tolist() if "file_id" in df_meta.columns else [None] * len(df_meta)
    file_paths = df_meta["file_path"].tolist()

    criterion = nn.MSELoss()
    criterion_mae = nn.L1Loss()

    models_by_key: Dict[str, List[Tuple[int, nn.Module]]] = {}
    for mk in model_keys:
        if mk not in CFG_REGISTRY:
            print(f"[{group_name}] Unknown model_key: {mk} (skip)")
            continue
        models_by_key[mk] = _load_models_for_seeds(mk, seeds, ckpt_tmpl)

    loader_cache: Dict[str, Optional[DataLoader]] = {}
    fid_keys: List[str] = []
    for fid, fp in zip(file_ids, file_paths):
        fid_keys.append(str(fid) if fid is not None else Path(fp).stem)

    metrics = ["MSE", "MAE"]
    columns = pd.MultiIndex.from_tuples([(fid, m) for fid in fid_keys for m in metrics],
                                        names=["file_id", "metric"])

    row_labels = [DISPLAY_NAME.get(mk, mk) for mk in model_keys]
    table = pd.DataFrame(index=row_labels, columns=columns, dtype=str)
    table.index.name = "model"

    for fid, fp in zip(file_ids, file_paths):
        if fp not in loader_cache:
            loader_cache[fp] = _build_single_loader_from_pt(fp)
        loader = loader_cache[fp]
        fid_key = str(fid) if fid is not None else Path(fp).stem

        for mk in model_keys:
            model_pack = models_by_key.get(mk, [])
            label = DISPLAY_NAME.get(mk, mk)

            if len(model_pack) == 0:
                table.loc[label, (fid_key, "MSE")] = "nan ± nan"
                table.loc[label, (fid_key, "MAE")] = "nan ± nan"
                continue

            mses: List[float] = []
            maes: List[float] = []
            for s, model in model_pack:
                try:
                    r = _eval_one_satellite_norm(loader, model, criterion, criterion_mae)
                except Exception as e:
                    print(f"[{group_name} | {mk}] skip seed={s}: {fp}\n  → {e}")
                    r = None
                if r is None:
                    continue
                mse, mae = r
                mses.append(mse); maes.append(mae)

            if len(mses) == 0:
                table.loc[label, (fid_key, "MSE")] = "nan ± nan"
                table.loc[label, (fid_key, "MAE")] = "nan ± nan"
                continue

            mse_mean = float(pd.Series(mses).mean())
            mae_mean = float(pd.Series(maes).mean())
            mse_std  = float(pd.Series(mses).std(ddof=1)) if len(mses) > 1 else float("nan")
            mae_std  = float(pd.Series(maes).std(ddof=1)) if len(maes) > 1 else float("nan")

            table.loc[label, (fid_key, "MSE")] = _format_pm(mse_mean, mse_std, sig)
            table.loc[label, (fid_key, "MAE")] = _format_pm(mae_mean, mae_std, sig)

    out_dir_path = Path(out_dir); out_dir_path.mkdir(parents=True, exist_ok=True)
    out_path = str(out_dir_path / f"{group_name}_merged_norm_pretty.csv")
    table.to_csv(out_path, encoding="utf-8-sig")
    print(f"Saved pretty merged [{group_name}]: {out_path} | models={len(model_keys)} | files={len(fid_keys)}")
    return out_path


def evaluate_all_constellations_and_models(
    model_keys: List[str],
    seeds: List[int],
    ckpt_tmpl: str,
    out_dir: str,
    base_dir: str,
    do_per_model: bool = True,
    do_merge: bool = True,
    merge_only_models: Optional[List[str]] = None,  # None means use model_keys
    pretty: bool = False,  # False means numeric merge
    float_fmt: Optional[str] = None,
) -> None:
    """
    Generate per-model CSVs and merged CSVs (optional) for 7 constellations
    """
    criterion = nn.MSELoss()
    criterion_mae = nn.L1Loss()

    CONSTELLATION_METAS = {
        "LEMUR":     f"{base_dir}/LEMUR_pt.csv",
        "ASTROCAST": f"{base_dir}/ASTROCAST_pt.csv",
        "CAPELLA":   f"{base_dir}/CAPELLA_pt.csv",
        "ICEYE":     f"{base_dir}/ICEYE_pt.csv",
        "KINEIS":    f"{base_dir}/KINEIS_pt.csv",
        "KUIPER":    f"{base_dir}/KUIPER_pt.csv",
        "SKYSAT":    f"{base_dir}/SKYSAT_pt.csv",
    }

    exported = []

    for group, meta_csv in CONSTELLATION_METAS.items():
        if not Path(meta_csv).exists():
            print(f"[{group}] meta CSV not found: {meta_csv} (skip)")
            continue

        # 1) (Optional) individual CSV per model
        if do_per_model:
            for mk in model_keys:
                if mk not in CFG_REGISTRY:
                    print(f"[{mk}] Unknown model_key. skip. (available values: {list(CFG_REGISTRY.keys())})")
                    continue
                try:
                    out_csv = evaluate_constellation_one_model(
                        meta_csv=meta_csv,
                        group_name=group,
                        model_key=mk,
                        seeds=seeds,
                        ckpt_tmpl=ckpt_tmpl,
                        out_dir=out_dir,
                        criterion=criterion,
                        criterion_mae=criterion_mae,
                    )
                    if out_csv is not None:
                        exported.append(out_csv)
                except Exception as e:
                    print(f"[{group} | {mk}] error, skipping: {e}")

        # 2) merged CSV
        if do_merge:
            merge_models = merge_only_models if merge_only_models is not None else model_keys
            # keep only valid model keys
            merge_models = [m for m in merge_models if m in CFG_REGISTRY]
            if len(merge_models) == 0:
                print(f"[{group}] no valid models to merge. skip merge.")
                continue
            try:
                if pretty:
                    merged_csv = evaluate_constellation_merge_models_pretty(
                        meta_csv=meta_csv,
                        group_name=group,
                        model_keys=merge_models,
                        seeds=seeds,
                        ckpt_tmpl=ckpt_tmpl,
                        out_dir=out_dir,
                    )
                else:
                    merged_csv = evaluate_constellation_merge_models_numeric(
                        meta_csv=meta_csv,
                        group_name=group,
                        model_keys=merge_models,
                        seeds=seeds,
                        ckpt_tmpl=ckpt_tmpl,
                        out_dir=out_dir,
                        float_fmt=float_fmt,
                    )
                if merged_csv is not None:
                    exported.append(merged_csv)
            except Exception as e:
                print(f"[{group} | MERGE] error, skipping: {e}")

    print("\n=== Exported CSV files ===")
    for p in exported:
        print(" -", p)
    print("==========================")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Per-constellation inference (normalized, 3-run mean±std): merged numeric/pretty CSV"
    )
    parser.add_argument(
        "--model_keys", nargs="+", type=str,
        default=['ours_tt'],
        help=f"List of model keys (available values: {list(CFG_REGISTRY.keys())})"
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[1, 2, 3],
        help="List of seeds for mean/std calculation (default: 1 2 3)"
    )
    parser.add_argument(
        "--ckpt_tmpl", type=str, default="best_{model}({seed}).pt",
        help="Checkpoint path pattern (e.g., 'best_{model}({seed}).pt' or 'checkpoints/{model}_seed{seed}.pt')"
    )
    parser.add_argument(
        "--base_dir", type=str, default="please input your path",
        help="Base directory where meta CSVs are located"
    )
    parser.add_argument(
        "--out_dir", type=str, default="./eval_results_norm",
        help="Directory to save result CSVs"
    )
    parser.add_argument(
        "--no_per_model", action="store_true",
        help="Turn off individual CSV generation per model."
    )
    parser.add_argument(
        "--merge_only_models", nargs="+", type=str, default=None,
        help="Specify models to include only in merged CSV"
    )
    parser.add_argument(
        "--pretty", action="store_true",
        help="Save merged CSV in (file_id × [MSE|MAE]) structure with 'mean ± std' strings (default is numeric columns)"
    )
    parser.add_argument(
        "--float_fmt", type=str, default=None,
        help="float_format for numeric merged CSV saving (e.g., '%.8g')"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"DEVICE: {DEVICE}")
    print(f"Models: {args.model_keys}")
    print(f"Seeds : {args.seeds}")
    print(f"CKPTs : {args.ckpt_tmpl}")
    print(f"Base  : {args.base_dir}")
    print(f"Out   : {args.out_dir}")
    print(f"Mode  : {'pretty (mean ± std string)' if args.pretty else 'numeric columns'}")

    evaluate_all_constellations_and_models(
        model_keys=args.model_keys,
        seeds=args.seeds,
        ckpt_tmpl=args.ckpt_tmpl,
        out_dir=args.out_dir,
        base_dir=args.base_dir,
        do_per_model=not args.no_per_model,
        do_merge=True,
        merge_only_models=args.merge_only_models,   
        pretty=args.pretty,
        float_fmt=args.float_fmt,
    )


if __name__ == "__main__":
    main()