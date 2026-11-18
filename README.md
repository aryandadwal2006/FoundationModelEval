# FoundationModelEval

A small, reproducible evaluation harness for time-series foundation models, multimodal retrieval, and repository question answering. It runs locally with public checkpoints and datasets. No API keys or hosted inference services are used.

## Scope

- Forecasting: ETT public datasets, naive and seasonal-naive statistical baselines, a scikit-learn Ridge baseline, a PyTorch MLP baseline, and pretrained TimesFM 2.5 through Hugging Face Transformers.
- Forecast evaluation: zero-shot and optional fine-tuned TimesFM, MAE, RMSE, MAPE, sMAPE, noisy-context robustness, and per-window inference latency.
- Multimodal retrieval: local text documents, CSV tables, and rendered document-page images. Compare MiniLM and CLIP embeddings with cosine, dot-product, and TF-IDF retrieval.
- Repository QA: retrieve relevant code chunks for fixed natural-language questions and report Recall@k and MRR. The output is evidence/context retrieval, not generated answers.

## Setup

Python 3.10 or newer is recommended.

```bash
git clone https://github.com/aryandadwal2006/FoundationModelEval.git
cd FoundationModelEval
python -m venv .venv
```

Activate the environment:

```bash
# windows
.venv\Scripts\activate

# macos or linux
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

All checkpoints and datasets used here are public. The first run downloads the selected Hugging Face model and caches it locally. No token, API key, or account configuration is required.

## 1. Time-series forecasting

Run a quick baseline experiment without downloading a pretrained checkpoint:

```bash
python -m fmeval forecast --dataset synthetic --models naive,seasonal_naive,ridge,mlp
```

Run on one public ETT dataset:

```bash
python -m fmeval forecast --dataset ETTh1 --models naive,seasonal_naive,ridge,mlp,timesfm
```

Run all four ETT variants:

```bash
python -m fmeval forecast --dataset all --models naive,seasonal_naive,ridge,mlp,timesfm
```

Include fine-tuning of the pretrained model on the chronological training split:

```bash
python -m fmeval forecast --dataset ETTh1 --models naive,seasonal_naive,ridge,mlp,timesfm --finetune
```

The default forecasting configuration uses the OT target, 512 context points, 128 forecast points, seed 42, chronological 70/10/20 train/validation/test boundaries, and at most 32 test windows. The TimesFM checkpoint has a 128-point prediction horizon. Keep the horizon at 128 when fine-tuning it.

Useful options:

```text
--dataset ETTh1|ETTh2|ETTm1|ETTm2|all|synthetic
--target OT
--context 512
--horizon 128
--models naive,seasonal_naive,ridge,mlp,timesfm
--finetune
--finetune-steps 20
--max-test-windows 32
--seed 42
--output-dir outputs
```

ETT data is loaded as public CSV files through Hugging Face Datasets. The target column defaults to OT. Set `--target` to another numeric column when needed.

### Forecast protocol

The split is chronological: the first 70% is used for model fitting, the next 10% is reserved, and the final 20% is evaluated. Training windows use a stride of 32. Test windows do not overlap and never use future observations as context. The same test windows are used for every model.

The robustness measurement adds Gaussian noise to each test context with standard deviation equal to 5% of the training-series standard deviation. The report records the noisy MAE and its change from clean MAE. Inference latency is wall-clock milliseconds per forecast window and is measured separately from model fitting and checkpoint loading. The output includes both `fit_seconds` and `model_load_seconds` where applicable. Every model uses the same held-out windows.

TimesFM is loaded from `google/timesfm-2.5-200m-transformers`. Its first run downloads the public checkpoint; CPU inference is supported, while CUDA is used automatically when available. Fine-tuning updates the loaded checkpoint on training windows only.

## 2. Multimodal retrieval

Create the small local evaluation corpus and render its document-page images:

```bash
python -m fmeval demo-data
```

Run all retrieval experiments:

```bash
python -m fmeval retrieval --models minilm,clip --strategies cosine,dot,tfidf
```

The benchmark evaluates text, table, and document-image queries. MiniLM is evaluated on text and tables. CLIP is evaluated on text, tables, and images. Images are embedded from their pixels and compared with text-query embeddings.

Models:

- `minilm`: `sentence-transformers/all-MiniLM-L6-v2`, loaded with AutoTokenizer and AutoModel.
- `clip`: `openai/clip-vit-base-patch32`, loaded with CLIPProcessor and CLIPModel. It embeds images and text into a shared space.

Retrieval strategies:

- `cosine`: cosine similarity between dense embeddings.
- `dot`: raw dot-product similarity between dense embeddings.
- `tfidf`: lexical TF-IDF cosine similarity.

The included corpus is deliberately small so the pipeline is easy to inspect. Demo files are created only when missing, so rerunning a command does not overwrite your edits. For a custom corpus, keep these fields:

- `documents.jsonl`: one JSON object per line with `id` and `text`.
- `tables.csv`: one row per retrievable record; include a unique `record_id` column. All other columns are converted into searchable text.
- `image_manifest.jsonl`: one JSON object per line with `id`, `title`, `description`, and `image_path`. Image paths are resolved from the repository working directory.
- `queries.jsonl`: one JSON object per line with `id`, `modality` (`text`, `table`, or `image`), `query`, and `relevant_id`.


## 3. Repository question-answering benchmark

Run the included code benchmark:

```bash
python -m fmeval repo-qa
```

Evaluate another local repository:

```bash
python -m fmeval repo-qa --repo-path path/to/repository --queries path/to/queries.jsonl
```

The indexer reads source and documentation files, splits them into overlapping line chunks, embeds each chunk with MiniLM, and ranks candidate context for each question. It reports Recall@1, Recall@5, and MRR for dense cosine, dense dot product, and TF-IDF retrieval. The query file is JSONL with `id`, `question`, and `expected_paths` fields. Paths are relative to `--repo-path`.

This isolates retrieval/context-selection quality for AI-assisted software engineering. It intentionally does not call a generative LLM or claim to score generated answers.

## Outputs

Commands write machine-readable CSV files under `outputs/`:

- `forecast_metrics.csv`
- `forecast_metadata.json`
- `retrieval_metrics.csv`
- `repo_qa_metrics.csv`
- `repo_qa_rankings.csv`

The output directory is created automatically. Runs are deterministic where the underlying libraries permit it; seed, split boundaries, model names, and experiment settings are recorded.

## Tests

Run the unit tests:

```bash
python -m unittest discover -s tests -v
```

The unit tests cover window construction, forecasting metrics, and ranking metrics. Running the full experiments additionally downloads the selected public Hugging Face checkpoints and datasets. TimesFM fine-tuning is compute-intensive; CUDA is recommended for practical runtimes.

## Project layout

```text
fmeval/
  __init__.py
  __main__.py
  forecast.py
  retrieval.py
tests/
  test_core.py
requirements.txt
```

## References

- [TimesFM 2.5 Transformers checkpoint](https://huggingface.co/google/timesfm-2.5-200m-transformers)
- [TimesFM 2.5 Transformers documentation](https://huggingface.co/docs/transformers/model_doc/timesfm2_5)
- [Hugging Face Datasets: loading local and remote files](https://huggingface.co/docs/datasets/loading)
- [ETT dataset source](https://github.com/thuml/Time-Series-Library/tree/main/dataset/ETT-small)
