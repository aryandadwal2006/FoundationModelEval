import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from PIL import Image, ImageDraw, ImageFont
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DATA_ROOT = Path("data")
RETRIEVAL_ROOT = DATA_ROOT / "retrieval"
CODE_ROOT = DATA_ROOT / "codebase"


def ensure_demo_data():
    RETRIEVAL_ROOT.mkdir(parents=True, exist_ok=True)
    (RETRIEVAL_ROOT / "images").mkdir(parents=True, exist_ok=True)
    CODE_ROOT.mkdir(parents=True, exist_ok=True)

    documents = [
        {"id": "text_grid_storage", "text": "Grid-scale battery storage supports renewable energy integration. Lithium iron phosphate systems provide frequency response, peak shaving, and multi-hour storage for utility operators."},
        {"id": "text_clinical_trial", "text": "The clinical trial protocol defines participant eligibility, randomization, primary endpoints, adverse event reporting, and follow-up visits for a controlled study."},
        {"id": "text_supply_chain", "text": "Supply chain analytics tracks supplier lead time, inventory turnover, demand uncertainty, and fulfillment delays across distribution centers."},
        {"id": "text_forecasting", "text": "Time series forecasting evaluates chronological data splits, rolling windows, prediction horizons, and mean absolute error on held-out observations."},
    ]
    tables = [
        {"record_id": "T001", "category": "solar", "region": "north", "quarter": "2025-Q1", "units": 12, "revenue_m": 18.4},
        {"record_id": "T002", "category": "wind", "region": "west", "quarter": "2025-Q1", "units": 8, "revenue_m": 15.1},
        {"record_id": "T003", "category": "battery", "region": "north", "quarter": "2025-Q2", "units": 21, "revenue_m": 32.7},
        {"record_id": "T004", "category": "solar", "region": "south", "quarter": "2025-Q2", "units": 17, "revenue_m": 24.2},
    ]
    image_specs = [
        {"id": "image_energy", "title": "Renewable energy report", "description": "A report page showing renewable energy generation, solar capacity, wind output, and battery storage by region.", "lines": ["Solar capacity: 420 MW", "Wind output: 315 MWh", "Battery storage: 180 MWh"]},
        {"id": "image_health", "title": "Clinical study summary", "description": "A clinical study document page with participant enrollment, treatment groups, primary endpoints, and adverse event counts.", "lines": ["Participants enrolled: 240", "Treatment groups: 2", "Primary endpoint: 12 weeks"]},
        {"id": "image_logistics", "title": "Logistics dashboard", "description": "A logistics document page summarizing warehouse inventory, supplier delays, delivery lead time, and fulfillment performance.", "lines": ["Inventory turnover: 6.2", "Supplier delay: 4.1 days", "On-time delivery: 94 percent"]},
    ]
    queries = [
        {"id": "q_text_storage", "modality": "text", "query": "battery storage for renewable energy and grid frequency response", "relevant_id": "text_grid_storage"},
        {"id": "q_text_protocol", "modality": "text", "query": "participant eligibility randomization and clinical trial endpoints", "relevant_id": "text_clinical_trial"},
        {"id": "q_table_battery", "modality": "table", "query": "north region battery revenue during the second quarter", "relevant_id": "T003"},
        {"id": "q_table_wind", "modality": "table", "query": "west region wind units and revenue in first quarter", "relevant_id": "T002"},
        {"id": "q_image_energy", "modality": "image", "query": "document with solar capacity, wind generation, and battery storage metrics", "relevant_id": "image_energy"},
        {"id": "q_image_health", "modality": "image", "query": "clinical study page showing enrolled participants and treatment groups", "relevant_id": "image_health"},
    ]

    documents_path = RETRIEVAL_ROOT / "documents.jsonl"
    tables_path = RETRIEVAL_ROOT / "tables.csv"
    manifest_path = RETRIEVAL_ROOT / "image_manifest.jsonl"
    queries_path = RETRIEVAL_ROOT / "queries.jsonl"
    if not documents_path.exists():
        _write_jsonl(documents_path, documents)
    if not tables_path.exists():
        pd.DataFrame(tables).to_csv(tables_path, index=False)
    if not manifest_path.exists():
        _write_jsonl(manifest_path, [
            {
                "id": item["id"],
                "title": item["title"],
                "description": item["description"],
                "image_path": f"data/retrieval/images/{item['id']}.png",
                "lines": item["lines"],
            }
            for item in image_specs
        ])
    if not queries_path.exists():
        _write_jsonl(queries_path, queries)

    for item in image_specs:
        path = RETRIEVAL_ROOT / "images" / f"{item['id']}.png"
        if not path.exists():
            _render_page(path, item["title"], item["lines"])

    code_files = {
        "forecasting.py": '''import numpy as np

def make_windows(values, context, horizon):
    x, y = [], []
    for end in range(context, len(values) - horizon + 1):
        x.append(values[end - context:end])
        y.append(values[end:end + horizon])
    return np.asarray(x), np.asarray(y)

def mean_absolute_error(target, prediction):
    return float(np.abs(target - prediction).mean())
''',
        "retrieval.py": '''import numpy as np

def cosine_scores(query, documents):
    query = query / max(np.linalg.norm(query), 1e-8)
    documents = documents / np.maximum(np.linalg.norm(documents, axis=1, keepdims=True), 1e-8)
    return documents @ query

def top_k(scores, k):
    return np.argsort(scores)[::-1][:k]
''',
        "training.py": '''import torch

def train_step(model, optimizer, inputs, targets):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    predictions = model(inputs)
    loss = torch.nn.functional.mse_loss(predictions, targets)
    loss.backward()
    optimizer.step()
    return float(loss.detach())
''',
    }
    for filename, source in code_files.items():
        path = CODE_ROOT / filename
        if not path.exists():
            path.write_text(source, encoding="utf-8")

    code_queries = [
        {"id": "code_windowing", "question": "Where are historical context and future target windows constructed?", "expected_paths": ["forecasting.py"]},
        {"id": "code_similarity", "question": "Which function normalizes query and document vectors before similarity ranking?", "expected_paths": ["retrieval.py"]},
        {"id": "code_optimizer", "question": "Where is the model optimizer zeroed, loss backpropagated, and parameters updated?", "expected_paths": ["training.py"]},
        {"id": "code_metric", "question": "Where is mean absolute forecasting error computed?", "expected_paths": ["forecasting.py"]},
    ]
    if not (DATA_ROOT / "code_qa.jsonl").exists():
        _write_jsonl(DATA_ROOT / "code_qa.jsonl", code_queries)


def _render_page(path, title, lines):
    image = Image.new("RGB", (900, 1160), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.rectangle((48, 48, 852, 1112), outline="black", width=3)
    draw.text((90, 105), title, fill="black", font=font)
    draw.line((90, 155, 810, 155), fill="black", width=2)
    y = 235
    for line in lines:
        draw.text((100, y), line, fill="black", font=font)
        draw.line((100, y + 44, 790, y + 44), fill="gray", width=1)
        y += 125
    draw.text((100, 970), "Evaluation corpus sample page", fill="black", font=font)
    image.save(path)


def _write_jsonl(path, records):
    with Path(path).open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def _load_jsonl(path):
    dataset = load_dataset("json", data_files=str(path), split="train")
    return list(dataset)


def _load_tables(path):
    dataset = load_dataset("csv", data_files=str(path), split="train")
    return list(dataset)


class MiniLMEncoder:
    def __init__(self):
        from transformers import AutoModel, AutoTokenizer

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_id = "sentence-transformers/all-MiniLM-L6-v2"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()

    def encode_texts(self, texts):
        output = []
        with torch.no_grad():
            for start in range(0, len(texts), 32):
                batch = texts[start:start + 32]
                tokens = self.tokenizer(
                    batch, padding=True, truncation=True, max_length=256, return_tensors="pt"
                ).to(self.device)
                hidden = self.model(**tokens).last_hidden_state
                mask = tokens["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                output.append(pooled.cpu().numpy())
        return np.concatenate(output, axis=0).astype(np.float32)


class CLIPEncoder:
    def __init__(self):
        from transformers import CLIPModel, CLIPProcessor

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_id = "openai/clip-vit-base-patch32"
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id).to(self.device).eval()

    def encode_texts(self, texts):
        output = []
        with torch.no_grad():
            for start in range(0, len(texts), 32):
                batch = self.processor(
                    text=texts[start:start + 32],
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                features = self.model.get_text_features(
                    input_ids=input_ids, attention_mask=attention_mask
                )
                output.append(features.cpu().numpy())
        return np.concatenate(output, axis=0).astype(np.float32)

    def encode_images(self, paths):
        output = []
        with torch.no_grad():
            for start in range(0, len(paths), 16):
                images = [
                    Image.open(path).convert("RGB")
                    for path in paths[start:start + 16]
                ]
                batch = self.processor(images=images, return_tensors="pt")
                features = self.model.get_image_features(
                    pixel_values=batch["pixel_values"].to(self.device)
                )
                output.append(features.cpu().numpy())
        return np.concatenate(output, axis=0).astype(np.float32)


def _score(query, corpus, query_vector, corpus_vectors, strategy):
    if strategy == "cosine":
        q = query_vector / max(float(np.linalg.norm(query_vector)), 1e-8)
        docs = corpus_vectors / np.maximum(
            np.linalg.norm(corpus_vectors, axis=1, keepdims=True), 1e-8
        )
        return docs @ q
    if strategy == "dot":
        return corpus_vectors @ query_vector
    if strategy == "tfidf":
        matrix = TfidfVectorizer().fit_transform([query] + corpus)
        return cosine_similarity(matrix[0:1], matrix[1:]).ravel()
    raise ValueError(f"unknown retrieval strategy: {strategy}")


def ranking_metrics(ranked_ids, relevant_ids, k=5):
    relevant = set(relevant_ids)
    reciprocal_rank = 0.0
    for rank, item_id in enumerate(ranked_ids, start=1):
        if item_id in relevant:
            reciprocal_rank = 1.0 / rank
            break
    return {
        "recall_at_1": float(any(item in relevant for item in ranked_ids[:1])),
        "recall_at_5": float(any(item in relevant for item in ranked_ids[:k])),
        "mrr": reciprocal_rank,
    }


def _load_retrieval_items(modality):
    if modality == "text":
        return [
            {"id": row["id"], "text": row["text"], "image_path": None}
            for row in _load_jsonl(RETRIEVAL_ROOT / "documents.jsonl")
        ]
    if modality == "table":
        rows = _load_tables(RETRIEVAL_ROOT / "tables.csv")
        output = []
        for row in rows:
            text = " ".join(f"{key} {value}" for key, value in row.items())
            output.append({"id": str(row["record_id"]), "text": text, "image_path": None})
        return output
    if modality == "image":
        rows = _load_jsonl(RETRIEVAL_ROOT / "image_manifest.jsonl")
        return [
            {
                "id": row["id"],
                "text": f"{row['title']}. {row['description']}",
                "image_path": row["image_path"],
            }
            for row in rows
        ]
    raise ValueError(f"unknown modality: {modality}")


def _evaluate_retrieval_model(model_name, encoder, queries, strategies):
    rows = []
    for modality in ("text", "table", "image"):
        modality_queries = [q for q in queries if q["modality"] == modality]
        if not modality_queries or (modality == "image" and model_name != "clip"):
            continue
        items = _load_retrieval_items(modality)
        candidate_texts = [item["text"] for item in items]
        item_ids = [item["id"] for item in items]
        if modality == "image":
            candidate_vectors = encoder.encode_images([item["image_path"] for item in items])
        else:
            candidate_vectors = encoder.encode_texts(candidate_texts)
        query_vectors = encoder.encode_texts([q["query"] for q in modality_queries])
        for query_row, query_vector in zip(modality_queries, query_vectors):
            for strategy in strategies:
                scores = _score(
                    query_row["query"], candidate_texts,
                    query_vector, candidate_vectors, strategy,
                )
                order = np.argsort(scores)[::-1]
                ranked_ids = [item_ids[index] for index in order]
                metrics = ranking_metrics(ranked_ids, [query_row["relevant_id"]])
                rows.append({
                    "model": model_name,
                    "modality": modality,
                    "strategy": strategy,
                    "query_id": query_row["id"],
                    **metrics,
                })
    return rows


def run_retrieval(args):
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    strategies = [item.strip() for item in args.strategies.split(",") if item.strip()]
    if set(models) - {"minilm", "clip"}:
        raise ValueError("--models accepts minilm and clip")
    if set(strategies) - {"cosine", "dot", "tfidf"}:
        raise ValueError("--strategies accepts cosine, dot, and tfidf")
    queries = _load_jsonl(RETRIEVAL_ROOT / "queries.jsonl")
    rows = []
    for model_name in models:
        print(f"loading {model_name}")
        encoder = MiniLMEncoder() if model_name == "minilm" else CLIPEncoder()
        rows.extend(_evaluate_retrieval_model(model_name, encoder, queries, strategies))
        del encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    frame = pd.DataFrame(rows)
    summary = (
        frame.groupby(["model", "modality", "strategy"], as_index=False)
        [["recall_at_1", "recall_at_5", "mrr"]]
        .mean()
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "retrieval_metrics.csv", index=False)
    print(summary.to_string(index=False))
    print(f"saved results to {output_dir / 'retrieval_metrics.csv'}")


def _chunk_repository(repo_path, chunk_lines=80, overlap=10):
    root = Path(repo_path).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"repository directory does not exist: {root}")
    suffixes = {".py", ".md", ".rst", ".txt", ".json", ".yaml", ".yml", ".toml", ".js", ".ts", ".tsx", ".go", ".rs", ".java"}
    excluded = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", "node_modules", "outputs", ".mypy_cache"}
    chunks = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        relative = path.relative_to(root)
        if any(part in excluded for part in relative.parts) or path.stat().st_size > 1_000_000:
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (UnicodeDecodeError, OSError):
            continue
        step = max(1, chunk_lines - overlap)
        for start in range(0, max(1, len(lines)), step):
            section = lines[start:start + chunk_lines]
            if not section:
                continue
            text = f"file: {relative}\n" + "\n".join(section)
            chunks.append({
                "path": relative.as_posix(),
                "start_line": start + 1,
                "text": text,
            })
            if start + chunk_lines >= len(lines):
                break
    if not chunks:
        raise ValueError(f"no supported source files found under {root}")
    return chunks


def run_repo_qa(args):
    queries = _load_jsonl(args.queries)
    chunks = _chunk_repository(args.repo_path)
    texts = [item["text"] for item in chunks]
    encoder = MiniLMEncoder() if args.model == "minilm" else None
    if encoder is None:
        raise ValueError("--model currently accepts minilm")
    print(f"embedding {len(chunks)} code chunks")
    document_vectors = encoder.encode_texts(texts)
    query_vectors = encoder.encode_texts([item["question"] for item in queries])
    strategies = [item.strip() for item in args.strategies.split(",") if item.strip()]
    if set(strategies) - {"cosine", "dot", "tfidf"}:
        raise ValueError("--strategies accepts cosine, dot, and tfidf")
    metric_rows = []
    ranking_rows = []
    paths = [item["path"] for item in chunks]
    for query_row, query_vector in zip(queries, query_vectors):
        for strategy in strategies:
            scores = _score(
                query_row["question"], texts, query_vector, document_vectors, strategy
            )
            order = np.argsort(scores)[::-1]
            ranked_paths = [paths[index] for index in order]
            relevant = set(query_row["expected_paths"])
            rank = next(
                (index for index, path in enumerate(ranked_paths, start=1) if path in relevant),
                None,
            )
            top_index = int(order[0])
            metric_rows.append({
                "strategy": strategy,
                "query_id": query_row["id"],
                "recall_at_1": float(ranked_paths[0] in relevant),
                "recall_at_5": float(any(path in relevant for path in ranked_paths[:5])),
                "mrr": 0.0 if rank is None else 1.0 / rank,
            })
            ranking_rows.append({
                "query_id": query_row["id"],
                "question": query_row["question"],
                "strategy": strategy,
                "expected_paths": ",".join(query_row["expected_paths"]),
                "top1_path": chunks[top_index]["path"],
                "top1_start_line": chunks[top_index]["start_line"],
                "top1_context": chunks[top_index]["text"][:500].replace("\n", " "),
                "hit_at_1": float(ranked_paths[0] in relevant),
                "rank_of_first_relevant": rank,
            })
    metrics = pd.DataFrame(metric_rows)
    summary = metrics.groupby("strategy", as_index=False)[
        ["recall_at_1", "recall_at_5", "mrr"]
    ].mean()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "repo_qa_metrics.csv", index=False)
    pd.DataFrame(ranking_rows).to_csv(output_dir / "repo_qa_rankings.csv", index=False)
    print(summary.to_string(index=False))
    print(f"saved results to {output_dir / 'repo_qa_metrics.csv'}")
