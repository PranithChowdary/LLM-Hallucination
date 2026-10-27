"""
Download every model B3-B6 needs into models/, so the SLURM jobs can run
with HF_HUB_OFFLINE=1.

Run ONCE on the LOGIN node (compute nodes have no internet):
    python scripts/download_signal_models.py

Already-downloaded models are skipped. Only PyTorch weights are fetched
(safetensors when the repo has them, otherwise pytorch_model.bin).
"""

from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"

MODELS = {
    # local folder name           Hugging Face repo                  used by
    "Qwen2.5-0.5B-Instruct": "Qwen/Qwen2.5-0.5B-Instruct",       # all (already there from A2/A3)
    "deberta-v3-large-mnli": "potsawee/deberta-v3-large-mnli",   # B3 SelfCheck-NLI
    "deberta-v2-xlarge-mnli": "microsoft/deberta-v2-xlarge-mnli",  # B4 clustering, labels
    "roberta-large": "FacebookAI/roberta-large",                 # B3 SelfCheck-BERTScore
}

NON_TORCH = ["*.h5", "*.msgpack", "*.ot", "*.onnx", "onnx/*", "tf_model*", "flax_model*", "rust_model*", "coreml/*"]


def ignore_patterns(repo_id: str) -> list[str]:
    files = HfApi().list_repo_files(repo_id)
    patterns = list(NON_TORCH)
    if any(f.endswith(".safetensors") for f in files):
        patterns.append("*.bin")  # safetensors present: skip the duplicate .bin
    return patterns


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for name, repo_id in MODELS.items():
        target = MODELS_DIR / name
        if (target / "config.json").exists():
            print(f"[SKIP] {name} already in {target}")
            continue
        print(f"[DOWNLOAD] {repo_id} -> {target}")
        snapshot_download(
            repo_id=repo_id,
            local_dir=target,
            ignore_patterns=ignore_patterns(repo_id),
        )
    print("\n[SUCCESS] Models ready:")
    for name in MODELS:
        print("  ", MODELS_DIR / name)


if __name__ == "__main__":
    main()
