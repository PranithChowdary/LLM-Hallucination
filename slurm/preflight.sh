#!/bin/bash
# Check that everything B3-B6 needs is on the server BEFORE queueing jobs.
# Run on the LOGIN node from the repository root:
#     bash slurm/preflight.sh
# Exit code 0 = ready. submit_all.sh runs this automatically.

cd "$(dirname "$0")/.." || exit 1
source ~/miniconda3/bin/activate ${CONDA_ENV:-}

fail=0
ok()   { echo "  [ OK ] $1"; }
bad()  { echo "  [FAIL] $1"; fail=1; }
warn() { echo "  [WARN] $1"; }

need_file() { if [ -e "$1" ]; then ok "$1"; else bad "$1 missing -- $2"; fi; }

echo "Data"
need_file data/calibration.parquet "frozen A1 split"
need_file data/generation_qwen_calibration.parquet "A2 main answers"
need_file data/generation_qwen_calibration_extra15.parquet "run src/generation/generate_extra.py (15 samples/prompt) first"
need_file data/processed/b1_answer_features.parquet "run scripts/b1_features.py (needed by B6)"
if ls data/a3/token_features/part_*.parquet >/dev/null 2>&1; then
    ok "data/a3/token_features/ ($(ls data/a3/token_features/part_*.parquet | wc -l) shards)"
else
    warn "data/a3/token_features/ empty -- B3 grey-box baselines will be skipped (unzip data/a3/token_features.zip)"
fi

echo "Models (download on the login node: python scripts/download_signal_models.py)"
for m in Qwen2.5-0.5B-Instruct deberta-v3-large-mnli deberta-v2-xlarge-mnli roberta-large; do
    need_file "models/$m/config.json" "python scripts/download_signal_models.py"
done

echo "Python packages ($(which python))"
python - <<'EOF' || fail=1
import importlib, sys
missing = []
for name in ["pandas", "pyarrow", "numpy", "sklearn", "torch", "transformers", "tqdm",
             "sentencepiece", "matplotlib", "bert_score", "nbconvert", "ipykernel", "huggingface_hub"]:
    try:
        mod = importlib.import_module(name)
        print(f"  [ OK ] {name} {getattr(mod, '__version__', '')}")
    except Exception as exc:
        print(f"  [FAIL] {name}: {exc}")
        missing.append(name)
if missing:
    print("  -> pip install -r requirements.txt")
    sys.exit(1)
EOF

if jupyter kernelspec list 2>/dev/null | grep -q "python3"; then
    ok "jupyter kernel 'python3'"
else
    bad "no 'python3' jupyter kernel -- python -m ipykernel install --user --name python3"
fi

echo "Sample completeness"
python - <<'EOF' || fail=1
import sys
from pathlib import Path
import pandas as pd
path = Path("data/generation_qwen_calibration_extra15.parquet")
if not path.exists():
    sys.exit(0)  # already reported above
main = pd.read_parquet("data/generation_qwen_calibration.parquet", columns=["uid"])
extra = pd.read_parquet(path, columns=["uid", "response_index"])
per_uid = extra.groupby("uid").size()
complete = int((per_uid >= 15).sum())
print(f"  prompts: {len(main):,}   with samples: {len(per_uid):,}   with all 15: {complete:,}")
if complete < len(main):
    print("  [WARN] some prompts have <15 samples; they are kept with what they have, "
          "prompts with none are skipped")
EOF

echo
if [ $fail -eq 0 ]; then
    echo "Preflight PASSED"
else
    echo "Preflight FAILED -- fix the [FAIL] lines above"
fi
exit $fail
