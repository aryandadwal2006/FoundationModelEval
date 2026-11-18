import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.linear_model import Ridge
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ETT_URLS = {
    "ETTh1": "https://raw.githubusercontent.com/thuml/Time-Series-Library/main/dataset/ETT-small/ETTh1.csv",
    "ETTh2": "https://raw.githubusercontent.com/thuml/Time-Series-Library/main/dataset/ETT-small/ETTh2.csv",
    "ETTm1": "https://raw.githubusercontent.com/thuml/Time-Series-Library/main/dataset/ETT-small/ETTm1.csv",
    "ETTm2": "https://raw.githubusercontent.com/thuml/Time-Series-Library/main/dataset/ETT-small/ETTm2.csv",
}
TIMESFM_ID = "google/timesfm-2.5-200m-transformers"


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_series(name, target, seed):
    if name == "synthetic":
        rng = np.random.default_rng(seed)
        t = np.arange(6000, dtype=np.float32)
        values = (
            0.0004 * t
            + 1.5 * np.sin(2 * np.pi * t / 24)
            + 0.55 * np.sin(2 * np.pi * t / 168)
            + rng.normal(0, 0.2, size=len(t))
        )
        return values.astype(np.float32), "synthetic"
    if name not in ETT_URLS:
        raise ValueError(f"unknown dataset: {name}")
    data = load_dataset("csv", data_files=ETT_URLS[name], split="train")
    frame = data.to_pandas()
    if target not in frame.columns:
        numeric = frame.select_dtypes(include=np.number).columns.tolist()
        if not numeric:
            raise ValueError(f"no numeric columns found in {name}")
        raise ValueError(f"target '{target}' not found in {name}; numeric columns: {numeric}")
    values = pd.to_numeric(frame[target], errors="coerce").dropna().to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError(f"target '{target}' contains non-finite values")
    return values, name


def make_windows(values, context, horizon, start, stop, stride):
    starts = range(max(context, start), stop - horizon + 1, stride)
    pairs = [(values[i - context:i], values[i:i + horizon]) for i in starts]
    if not pairs:
        raise ValueError("not enough observations to build forecasting windows")
    x, y = zip(*pairs)
    return np.stack(x).astype(np.float32), np.stack(y).astype(np.float32)


def forecast_metrics(y_true, y_pred):
    error = np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64)
    absolute = np.abs(error)
    denominator = np.maximum(np.abs(y_true), 1e-6)
    smape_denominator = np.maximum(np.abs(y_true) + np.abs(y_pred), 1e-6)
    return {
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "mape": float((absolute / denominator).mean() * 100.0),
        "smape": float((2.0 * absolute / smape_denominator).mean() * 100.0),
    }


def _window_normalize(x, y=None):
    means = x.mean(axis=1, keepdims=True)
    scales = np.maximum(x.std(axis=1, keepdims=True), 1e-5)
    xn = (x - means) / scales
    if y is None:
        return xn, means, scales
    return xn, (y - means) / scales, means, scales


def _fit_ridge(train_x, train_y):
    x_norm, y_norm, means, scales = _window_normalize(train_x, train_y)
    model = Ridge(alpha=1.0)
    model.fit(x_norm, y_norm)

    def predict(x):
        x_norm, current_means, current_scales = _window_normalize(x)
        return model.predict(x_norm) * current_scales + current_means

    return predict


def _fit_mlp(train_x, train_y, epochs, seed):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_norm, y_norm, _, _ = _window_normalize(train_x, train_y)
    dataset = TensorDataset(
        torch.tensor(x_norm, dtype=torch.float32),
        torch.tensor(y_norm, dtype=torch.float32),
    )
    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    model = nn.Sequential(
        nn.Linear(train_x.shape[1], 256),
        nn.GELU(),
        nn.Linear(256, train_y.shape[1]),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

    def predict(x):
        model.eval()
        x_norm, means, scales = _window_normalize(x)
        with torch.no_grad():
            pred = model(torch.tensor(x_norm, dtype=torch.float32, device=device))
        return pred.cpu().numpy() * scales + means

    return predict


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _load_timesfm():
    try:
        from transformers import TimesFm2_5ModelForPrediction
    except ImportError as exc:
        raise RuntimeError(
            "TimesFM requires a recent Transformers release; install requirements.txt"
        ) from exc
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TimesFm2_5ModelForPrediction.from_pretrained(TIMESFM_ID)
    model.to(device)
    model.eval()
    return model, device


def _predict_timesfm(model, device, x, context, horizon):
    predictions = []
    with torch.no_grad():
        for offset in range(0, len(x), 8):
            batch = x[offset:offset + 8]
            past = [torch.tensor(row, dtype=torch.float32, device=device) for row in batch]
            output = model(
                past_values=past,
                forecast_context_len=context,
                force_flip_invariance=False,
                return_dict=True,
            )
            predictions.append(output.mean_predictions[:, :horizon].detach().cpu().numpy())
    return np.concatenate(predictions, axis=0)


def _finetune_timesfm(model, device, train_x, train_y, context, steps, seed):
    horizon_length = int(model.config.horizon_length)
    if train_y.shape[1] != horizon_length:
        raise ValueError(
            f"TimesFM fine-tuning requires --horizon {horizon_length}; "
            f"the selected horizon is {train_y.shape[1]}"
        )
    set_seed(seed)
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    model.train()
    batch_size = min(4, len(train_x))
    losses = []
    for _ in range(steps):
        indices = rng.integers(0, len(train_x), size=batch_size)
        past = [
            torch.tensor(row, dtype=torch.float32, device=device)
            for row in train_x[indices]
        ]
        future = torch.tensor(train_y[indices], dtype=torch.float32, device=device)
        optimizer.zero_grad(set_to_none=True)
        output = model(
            past_values=past,
            future_values=future,
            forecast_context_len=context,
            force_flip_invariance=False,
            return_dict=True,
        )
        loss = output.loss.mean()
        if not torch.isfinite(loss):
            raise RuntimeError("TimesFM fine-tuning produced a non-finite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    model.eval()
    return float(np.mean(losses))


def _evaluate_predictor(name, regime, predict, test_x, test_y, noisy_x, fit_seconds):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _sync(device)
    start = time.perf_counter()
    clean_pred = predict(test_x)
    _sync(device)
    inference_seconds = time.perf_counter() - start
    noisy_pred = predict(noisy_x)
    clean_scores = forecast_metrics(test_y, clean_pred)
    noisy_scores = forecast_metrics(test_y, noisy_pred)
    return {
        "model": name,
        "regime": regime,
        **clean_scores,
        "noise_mae": noisy_scores["mae"],
        "noise_mae_delta": noisy_scores["mae"] - clean_scores["mae"],
        "inference_ms_per_window": inference_seconds * 1000.0 / len(test_x),
        "fit_seconds": fit_seconds,
        "test_windows": int(len(test_x)),
    }


def _run_dataset(name, args):
    values, dataset_name = load_series(name, args.target, args.seed)
    n = len(values)
    train_end = int(n * 0.70)
    validation_end = int(n * 0.80)
    if min(train_end, n - validation_end) <= args.context + args.horizon:
        raise ValueError(f"{name} is too short for the selected context and horizon")
    train_x, train_y = make_windows(
        values, args.context, args.horizon, args.context, train_end, 32
    )
    if len(train_x) > 2048:
        selected = np.linspace(0, len(train_x) - 1, 2048).astype(int)
        train_x, train_y = train_x[selected], train_y[selected]
    test_x, test_y = make_windows(
        values, args.context, args.horizon, validation_end, n, args.horizon
    )
    test_x = test_x[:args.max_test_windows]
    test_y = test_y[:args.max_test_windows]
    rng = np.random.default_rng(args.seed + 1000)
    noise_std = float(np.std(values[:train_end]) * args.noise_scale)
    noisy_x = test_x + rng.normal(0.0, noise_std, size=test_x.shape).astype(np.float32)
    rows = []
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    allowed = {"naive", "seasonal_naive", "ridge", "mlp", "timesfm"}
    unknown = set(models) - allowed
    if unknown:
        raise ValueError(f"unknown forecasting models: {sorted(unknown)}")
    if args.finetune and "timesfm" not in models:
        raise ValueError("--finetune requires timesfm in --models")

    for model_name in models:
        start = time.perf_counter()
        if model_name == "naive":
            predict = lambda x: np.repeat(x[:, -1:], args.horizon, axis=1)
            fit_seconds = 0.0
            rows.append(_evaluate_predictor(
                model_name, "baseline", predict, test_x, test_y, noisy_x, fit_seconds
            ))
        elif model_name == "seasonal_naive":
            period = min(24, args.context)
            predict = lambda x: np.tile(x[:, -period:], (1, int(np.ceil(args.horizon / period))))[:, :args.horizon]
            fit_seconds = 0.0
            rows.append(_evaluate_predictor(
                model_name, "baseline", predict, test_x, test_y, noisy_x, fit_seconds
            ))
        elif model_name == "ridge":
            predict = _fit_ridge(train_x, train_y)
            fit_seconds = time.perf_counter() - start
            rows.append(_evaluate_predictor(
                model_name, "trained", predict, test_x, test_y, noisy_x, fit_seconds
            ))
        elif model_name == "mlp":
            predict = _fit_mlp(train_x, train_y, args.mlp_epochs, args.seed)
            fit_seconds = time.perf_counter() - start
            rows.append(_evaluate_predictor(
                model_name, "trained", predict, test_x, test_y, noisy_x, fit_seconds
            ))
        else:
            model, device = _load_timesfm()
            load_seconds = time.perf_counter() - start
            predict = lambda x: _predict_timesfm(
                model, device, x, args.context, args.horizon
            )
            rows.append(_evaluate_predictor(
                model_name, "zero_shot", predict, test_x, test_y, noisy_x, load_seconds
            ))
            if args.finetune:
                tune_start = time.perf_counter()
                train_loss = _finetune_timesfm(
                    model, device, train_x, train_y, args.context,
                    args.finetune_steps, args.seed,
                )
                tune_seconds = time.perf_counter() - tune_start
                result = _evaluate_predictor(
                    model_name, "fine_tuned", predict, test_x, test_y, noisy_x, tune_seconds
                )
                result["finetune_loss"] = train_loss
                rows.append(result)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    metadata = {
        "dataset": dataset_name,
        "target": args.target,
        "observations": n,
        "seed": args.seed,
        "context": args.context,
        "horizon": args.horizon,
        "train_fraction": 0.70,
        "validation_fraction": 0.10,
        "test_fraction": 0.20,
        "train_windows": int(len(train_x)),
        "test_windows": int(len(test_x)),
        "noise_std": noise_std,
        "models": models,
        "finetune": args.finetune,
    }
    for row in rows:
        row["dataset"] = dataset_name
        row["target"] = args.target
    return rows, metadata


def run_forecast(args):
    if args.context < 8 or args.horizon < 1:
        raise ValueError("context must be at least 8 and horizon must be positive")
    if args.max_test_windows < 1 or args.finetune_steps < 1:
        raise ValueError("window and fine-tuning counts must be positive")
    set_seed(args.seed)
    names = list(ETT_URLS) if args.dataset == "all" else [args.dataset]
    rows = []
    metadata = []
    for name in names:
        print(f"evaluating {name}")
        dataset_rows, dataset_meta = _run_dataset(name, args)
        rows.extend(dataset_rows)
        metadata.append(dataset_meta)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "forecast_metrics.csv", index=False)
    with (output_dir / "forecast_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    print(frame.to_string(index=False))
    print(f"saved results to {output_dir / 'forecast_metrics.csv'}")
