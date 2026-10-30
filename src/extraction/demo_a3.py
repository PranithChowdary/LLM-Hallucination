'''from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = "models/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    local_files_only=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    local_files_only=True,
    torch_dtype=torch.float32
)

model.eval()

print("Model loaded successfully!")
print("Device: CPU")
print("Hidden size:", model.config.hidden_size)'''


'''import pandas as pd

# Load the saved A2 generation results
file_path = "data/generation_qwen_calibration.parquet"
df = pd.read_parquet(file_path)

# Select one existing example
example = df.iloc[0]

print("A2 example loaded successfully!")
print("UID:", example["uid"])
print("Dataset:", example["dataset"])
print("Prompt:", str(example["prompt"])[:300])
print("Saved answer:", str(example["generated_answer"])[:500])'''


'''import pandas as pd
from transformers import AutoTokenizer

# Load the same saved A2 example
df = pd.read_parquet("data/generation_qwen_calibration.parquet")
example = df.iloc[0]

# Load the tokenizer saved with our model
tokenizer = AutoTokenizer.from_pretrained(
    "models/Qwen2.5-0.5B-Instruct",
    local_files_only=True
)

# Convert the saved answer text into token IDs
answer = str(example["generated_answer"])
encoded = tokenizer(answer, add_special_tokens=False)
token_ids = encoded["input_ids"]
tokens = tokenizer.convert_ids_to_tokens(token_ids)

# Display the results
print("UID:", example["uid"])
print("Answer:", answer[:300])
print("Number of answer tokens:", len(token_ids))
print("\nFirst 20 tokens:")
for token_id, token in zip(token_ids[:20], tokens[:20]):
    print(f"Token ID: {token_id:<8} Token: {token!r}")'''


import json
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# --------------------------------------------------
# 1. Load the same saved A2 example
# --------------------------------------------------
generation_file = "data/generation_qwen_calibration.parquet"
model_path = "models/Qwen2.5-0.5B-Instruct"
output_dir = Path("data/a3_test")
output_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(generation_file)
example = df.iloc[0]

uid = str(example["uid"])
prompt = str(example["prompt"])
answer = str(example["generated_answer"])

if not answer.strip():
    raise ValueError("The selected A2 answer is empty.")

# --------------------------------------------------
# 2. Load the locally saved tokenizer and model
# --------------------------------------------------
print("Loading tokenizer and model on CPU...")

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    local_files_only=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    local_files_only=True,
    torch_dtype=torch.float32
)
model.eval()

# --------------------------------------------------
# 3. Reconstruct prompt and answer token IDs
# --------------------------------------------------
messages = [{"role": "user", "content": prompt}]

prompt_encoding = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
)

prompt_ids = prompt_encoding["input_ids"]
answer_ids_list = tokenizer(
    answer, add_special_tokens=False
)["input_ids"]

if not answer_ids_list:
    raise ValueError("The saved answer produced no tokens.")

answer_ids = torch.tensor([answer_ids_list], dtype=torch.long)
full_ids = torch.cat([prompt_ids, answer_ids], dim=1)

prompt_length = prompt_ids.shape[1]
answer_token_count = answer_ids.shape[1]
hidden_size = model.config.hidden_size

max_length = getattr(
    model.config, "max_position_embeddings", None
)
if max_length and full_ids.shape[1] > max_length:
    raise ValueError(
        "Prompt plus answer exceeds the model context limit."
    )

print("UID:", uid)
print("Answer token count:", answer_token_count)

# --------------------------------------------------
# 4. Teacher-forced forward pass: no generation
# --------------------------------------------------
print("Calculating A3 features...")

with torch.inference_mode():
    result = model(
        input_ids=full_ids,
        output_hidden_states=True,
        use_cache=False
    )

    # The previous sequence position predicts each answer token.
    logits = result.logits[
        0,
        prompt_length - 1:
        prompt_length + answer_token_count - 1,
        :
    ].float()

    # Raw-model distribution over the complete vocabulary.
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)

    # Score of each actual saved-answer token.
    token_ids = answer_ids[0]
    actual_logits = logits.gather(
        1, token_ids.unsqueeze(1)
    ).squeeze(1)

    actual_log_probs = log_probs.gather(
        1, token_ids.unsqueeze(1)
    ).squeeze(1)

    actual_probs = actual_log_probs.exp()

    # Entropy of the complete next-token distribution.
    entropy = -(probs * log_probs).sum(dim=-1)

    # Final-layer hidden state at each answer token's own position.
    hidden_states = result.hidden_states[-1][
        0,
        prompt_length:
        prompt_length + answer_token_count,
        :
    ].detach().to(dtype=torch.float16, device="cpu")

# --------------------------------------------------
# 5. Save the token-level feature table
# --------------------------------------------------
token_strings = tokenizer.convert_ids_to_tokens(
    token_ids.tolist()
)

features = pd.DataFrame({
    "uid": [uid] * answer_token_count,
    "token_position": list(range(answer_token_count)),
    "token_id": token_ids.tolist(),
    "token_text": token_strings,
    "logit": actual_logits.tolist(),
    "probability": actual_probs.tolist(),
    "log_probability": actual_log_probs.tolist(),
    "entropy": entropy.tolist(),
})

# Basic validation of the newly computed features.
assert len(features) == answer_token_count
assert torch.isfinite(actual_logits).all().item()
assert torch.isfinite(actual_log_probs).all().item()
assert torch.isfinite(actual_probs).all().item()
assert torch.isfinite(entropy).all().item()
assert torch.isfinite(hidden_states).all().item()
assert hidden_states.shape == (answer_token_count, hidden_size)

features.to_parquet(
    output_dir / "token_features.parquet",
    index=False
)

torch.save(
    {
        "uid": uid,
        "token_positions": torch.arange(answer_token_count),
        "hidden_states": hidden_states,
        "hidden_size": hidden_size,
        "storage_dtype": "float16",
    },
    output_dir / "hidden_states.pt"
)

# --------------------------------------------------
# 6. Save a small summary for inspection
# --------------------------------------------------
summary = {
    "uid": uid,
    "answer_token_count": answer_token_count,
    "hidden_state_shape": list(hidden_states.shape),
    "hidden_size": hidden_size,
    "mean_probability": actual_probs.mean().item(),
    "mean_log_probability": actual_log_probs.mean().item(),
    "mean_entropy": entropy.mean().item(),
    "method": "teacher-forced forward pass",
    "generation_performed": False,
    "token_ids_note": (
        "Reconstructed from saved decoded answer text."
    ),
    "scores_note": (
        "Recomputed model scores, not original A2 sampling scores."
    ),
}

(output_dir / "example_summary.json").write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8"
)

print("\nA3 one-example extraction completed.")
print("Token features:", output_dir / "token_features.parquet")
print("Hidden states:", output_dir / "hidden_states.pt")
print("Summary:", output_dir / "example_summary.json")
print("\nFirst five token rows:")
print(features.head().to_string(index=False))
print("\nHidden-state shape:", tuple(hidden_states.shape))
print("All numerical checks passed.")