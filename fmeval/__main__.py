import argparse
from pathlib import Path

from fmeval.forecast import run_forecast
from fmeval.retrieval import ensure_demo_data, run_retrieval, run_repo_qa


def build_parser():
    parser = argparse.ArgumentParser(prog="fmeval")
    commands = parser.add_subparsers(dest="command", required=True)

    forecast = commands.add_parser("forecast")
    forecast.add_argument("--dataset", default="ETTh1", choices=["ETTh1", "ETTh2", "ETTm1", "ETTm2", "all", "synthetic"])
    forecast.add_argument("--target", default="OT")
    forecast.add_argument("--context", type=int, default=512)
    forecast.add_argument("--horizon", type=int, default=128)
    forecast.add_argument("--models", default="naive,seasonal_naive,ridge,mlp")
    forecast.add_argument("--finetune", action="store_true")
    forecast.add_argument("--finetune-steps", type=int, default=20)
    forecast.add_argument("--max-test-windows", type=int, default=32)
    forecast.add_argument("--mlp-epochs", type=int, default=5)
    forecast.add_argument("--noise-scale", type=float, default=0.05)
    forecast.add_argument("--seed", type=int, default=42)
    forecast.add_argument("--output-dir", default="outputs")

    retrieval = commands.add_parser("retrieval")
    retrieval.add_argument("--models", default="minilm,clip")
    retrieval.add_argument("--strategies", default="cosine,dot,tfidf")
    retrieval.add_argument("--seed", type=int, default=42)
    retrieval.add_argument("--output-dir", default="outputs")

    repo = commands.add_parser("repo-qa")
    repo.add_argument("--repo-path", default="data/codebase")
    repo.add_argument("--queries", default="data/code_qa.jsonl")
    repo.add_argument("--model", default="minilm")
    repo.add_argument("--strategies", default="cosine,dot,tfidf")
    repo.add_argument("--seed", type=int, default=42)
    repo.add_argument("--output-dir", default="outputs")

    commands.add_parser("demo-data")
    return parser


def main():
    args = build_parser().parse_args()
    if args.command == "forecast":
        run_forecast(args)
    elif args.command == "retrieval":
        ensure_demo_data()
        run_retrieval(args)
    elif args.command == "repo-qa":
        ensure_demo_data()
        run_repo_qa(args)
    elif args.command == "demo-data":
        ensure_demo_data()
        print("demo data ready")


if __name__ == "__main__":
    main()
