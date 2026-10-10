from huggingface_hub import snapshot_download
import os

model_id = "Glint-Research/Blink-1" #smallest model
models_dir = "/users/staff/pranith/LLM_Hallucinations/models"
target_folder = os.path.join(models_dir, model_id.split("/")[-1])

os.makedirs(target_folder, exist_ok=True)

print(f"Downloading model to: {target_folder}")

snapshot_download(
    repo_id=model_id,
    local_dir=target_folder
)

print(f"Done! Model downloaded to: {target_folder}")
