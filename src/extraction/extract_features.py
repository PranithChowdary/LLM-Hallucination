from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CALIBRATION_FILE = PROJECT_ROOT / "data" / "calibration.parquet"
GENERATION_FILE = PROJECT_ROOT / "data" / "generation_qwen_calibration.parquet"


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

calibration = pd.read_parquet(CALIBRATION_FILE)
generation = pd.read_parquet(GENERATION_FILE)


# ---------------------------------------------------------
# A3 alignment check
# ---------------------------------------------------------

assert calibration["uid"].is_unique, \
    "Calibration dataset contains duplicate UIDs."

assert generation["uid"].is_unique, \
    "Generation dataset contains duplicate UIDs."

assert set(calibration["uid"]) == set(generation["uid"]), \
    "Generation UIDs do not exactly match calibration UIDs."


# Verify that the prompt attached to each UID is unchanged.
calibration_prompts = calibration.set_index("uid")["prompt"]
generation_prompts = generation.set_index("uid")["prompt"]

assert (
    calibration_prompts.loc[generation["uid"]].values
    == generation["prompt"].values
).all(), \
    "Prompt mismatch detected between calibration and generation data."


print(f"A3 alignment verified: {len(generation)} examples")