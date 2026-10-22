"""Resumeable orchestration of the existing full-catalog evaluation CLI.

This script never trains a model. It uses the unchanged main.py evaluation
entry point and writes every run to an explicitly selected new directory.
"""
import argparse
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
DATASETS = ("fashion", "yelp", "beauty")
MODELS = ("llmesr_sasrec", "llmesr_bert4rec", "llmesr_gru4rec")
SEEDS = (41, 42, 43, 44, 45)
THRESHOLDS = {"fashion": (3, 4), "yelp": (12, 13), "beauty": (9, 4)}
VARIANTS = {
    "base": ["--disable_semantic_rerank"],
    "mcrr_c1r1z1_full": [],
    "mcrr_c0r0z0": ["--disable_semantic_centering", "--semantic_pooling", "mean", "--disable_candidate_standardization"],
    "mcrr_c1r0z0": ["--semantic_pooling", "mean", "--disable_candidate_standardization"],
    "mcrr_c0r1z0": ["--disable_semantic_centering", "--disable_candidate_standardization"],
    "mcrr_c0r0z1": ["--disable_semantic_centering", "--semantic_pooling", "mean"],
    "mcrr_c1r1z0": ["--disable_candidate_standardization"],
    "mcrr_c1r0z1": ["--semantic_pooling", "mean"],
    "mcrr_c0r1z1": ["--disable_semantic_centering"],
    "mcrr_c1r1z1_fixed_w1": ["--disable_auto_semantic_weight", "--semantic_rerank_weight", "1.0"],
}


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def save_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def summarize(root, results):
    rows = []
    for result in results:
        rows.append({
            "dataset": result["dataset"], "backbone": result["backbone"],
            "seed": result["seed"], "variant": result["variant"],
            "evaluation": result["protocol"]["evaluation"],
            "selected_weight": result["mcrr"]["selected_weight"],
            **result["metrics"],
            "checkpoint_sha256": result["source_checkpoint_sha256"],
            "result_file": str(root / "results" / result["dataset"] / result["backbone"] / ("seed" + str(result["seed"])) / (result["variant"] + ".json")),
        })
    save_csv(root / "01_逐次全量结果.csv", rows)
    groups = {}
    for result in results:
        key = (result["dataset"], result["backbone"], result["variant"])
        groups.setdefault(key, []).append(result)
    summaries = []
    for (dataset, backbone, variant), group in sorted(groups.items()):
        row = {"dataset": dataset, "backbone": backbone, "variant": variant,
               "n_runs": len(group), "complete_five_seeds": sorted(r["seed"] for r in group) == list(SEEDS)}
        for metric in group[0]["metrics"]:
            values = [r["metrics"][metric] for r in group]
            row[metric + " mean"] = statistics.mean(values)
            row[metric + " sample_std"] = statistics.stdev(values) if len(values) > 1 else ""
        summaries.append(row)
    save_csv(root / "02_五次均值与样本标准差.csv", summaries)


def verify_result(path, dataset, model, seed, variant, checkpoint_hash):
    result = json.loads(path.read_text(encoding="utf-8"))
    expected = {"dataset": dataset, "backbone": model, "seed": seed,
                "eval_seed": seed, "variant": variant, "source_checkpoint_sha256": checkpoint_hash}
    for key, value in expected.items():
        if result.get(key) != value:
            raise ValueError(f"Result mismatch: {path}: {key}")
    protocol = result["protocol"]
    if protocol["evaluation"] != "full" or protocol["test_negatives"] is not None or protocol["catalog_items"] <= 0:
        raise ValueError(f"Not full-catalog evaluation: {path}")
    if not path.with_suffix(".ranks.npz").is_file():
        raise ValueError(f"Missing per-user ranks: {path}")
    if len(result["metrics"]) != 10 or any(not math.isfinite(v) or not 0 <= v <= 1 for v in result["metrics"].values()):
        raise ValueError(f"Invalid metrics: {path}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, help="Stop after this many new runs (for initial real-data validation).")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = args.run_dir.resolve()
    if args.eval_batch_size < 1 or (args.limit is not None and args.limit < 1):
        parser.error("Batch size and limit must be positive")

    checkpoints = {}
    for dataset, model, seed in itertools.product(DATASETS, MODELS, SEEDS):
        path = REPO / "saved_mcrr_base" / dataset / model / f"base_seed{seed}" / "pytorch_model.bin"
        if not path.is_file():
            raise FileNotFoundError(f"Missing checkpoint; no training is permitted: {path}")
        checkpoints[(dataset, model, seed)] = path
    jobs = list(itertools.product(DATASETS, MODELS, SEEDS, VARIANTS))
    print(f"Full-catalog evaluation: {len(checkpoints)} checkpoints; {len(jobs)} results; batch={args.eval_batch_size}", flush=True)
    print(f"Output: {root}", flush=True)
    if args.dry_run:
        return

    root.mkdir(parents=True, exist_ok=True)
    hashes = {key: sha256(path) for key, path in checkpoints.items()}
    source_files = ["main.py", "trainers/trainer.py", "trainers/sequence_trainer.py", "generators/generator.py",
                    "generators/data.py", "generators/bert_generator.py", "utils/utils.py"]
    source_files += [str(p.relative_to(REPO)) for p in sorted((REPO / "models").glob("*.py"))]
    signature = {
        "evaluation": "full", "datasets": list(DATASETS), "models": list(MODELS), "seeds": list(SEEDS),
        "variants": VARIANTS, "eval_batch_size": args.eval_batch_size,
        "source_hashes": {p: sha256(REPO / p) for p in source_files},
        "checkpoints": {str(checkpoints[k]): v for k, v in hashes.items()},
        "data_hashes": {d: sha256(REPO / "data" / d / "handled" / "inter.txt") for d in DATASETS},
    }
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["signature"] != signature:
            raise ValueError("Run configuration or source hashes changed; choose a new run directory")
    else:
        manifest = {"created_at": now(), "python": args.python, "repo": str(REPO), "signature": signature}
        save_json(manifest_path, manifest)

    # Fail immediately rather than evaluate concurrently into one result directory.
    lock_path = root / "runner.lock"
    lock_fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    os.write(lock_fd, str(os.getpid()).encode("ascii"))
    os.close(lock_fd)
    results = []
    new_runs = 0
    active = None
    started = time.perf_counter()
    try:
        for dataset, model, seed, variant in jobs:
            active = f"{dataset}/{model}/seed{seed}/{variant}"
            result_dir = root / "results" / dataset / model / f"seed{seed}"
            result_path = result_dir / f"{variant}.json"
            checkpoint = checkpoints[(dataset, model, seed)]
            checkpoint_hash = hashes[(dataset, model, seed)]
            if result_path.exists():
                results.append(verify_result(result_path, dataset, model, seed, variant, checkpoint_hash))
                continue
            if args.limit is not None and new_runs >= args.limit:
                break
            result_dir.mkdir(parents=True, exist_ok=True)
            ts_user, ts_item = THRESHOLDS[dataset]
            command = [args.python, "-u", "-B", str(REPO / "main.py"), "--do_test",
                       "--dataset", dataset, "--model_name", model,
                       "--source_checkpoint", str(checkpoint),
                       "--output_dir", str(root / "scratch"),
                       "--check_path", f"{variant}/seed{seed}",
                       "--experiment_variant", variant,
                       "--eval_mode", "full", "--eval_batch_size", str(args.eval_batch_size),
                       "--seed", str(seed), "--eval_seed", str(seed), "--gpu_id", "0", "--num_workers", "0",
                       "--hidden_size", "64", "--max_len", "200",
                       "--ts_user", str(ts_user), "--ts_item", str(ts_item), "--freeze", "--use_cross_att",
                       "--log", "--log_root", str(root / "logs"),
                       "--result_json", str(result_path), "--rank_output", str(result_path.with_suffix(".ranks.npz"))]
            command += VARIANTS[variant]
            with (root / "commands.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"started_at": now(), "job": active, "cwd": str(REPO), "argv": command}, ensure_ascii=False) + "\n")
            save_json(root / "progress.json", {"status": "running", "updated_at": now(), "completed": len(results), "expected": len(jobs), "active": active, "runner_pid": os.getpid()})
            print(f"[{len(results) + 1}/{len(jobs)}] {active}", flush=True)
            with (result_dir / f"{variant}.console.log").open("w", encoding="utf-8") as log:
                completed = subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                                           env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"})
            if completed.returncode:
                raise RuntimeError(f"Evaluation failed ({completed.returncode}): {active}; see console log")
            results.append(verify_result(result_path, dataset, model, seed, variant, checkpoint_hash))
            new_runs += 1
            summarize(root, results)
        status = "complete" if len(results) == len(jobs) else "partial"
        summarize(root, results)
        save_json(root / "progress.json", {"status": status, "updated_at": now(), "completed": len(results), "expected": len(jobs), "new_runs": new_runs, "elapsed_seconds": time.perf_counter() - started})
        print(f"{status}: {len(results)}/{len(jobs)} results, {time.perf_counter() - started:.1f}s this invocation", flush=True)
    except BaseException as exc:
        save_json(root / "progress.json", {"status": "failed", "updated_at": now(), "completed": len(results), "expected": len(jobs), "active": active, "error": str(exc)})
        raise
    finally:
        lock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
