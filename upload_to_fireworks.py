import os
import requests

# Path to your local model folder
MODEL_DIR = "fireworks_model_test"
API_KEY = os.getenv("FIREWORKS_API_KEY")  # pulls from environment variable
UPLOAD_URL = "https://api.fireworks.ai/v1/model"

# Prepare headers
headers = {
    "Authorization": f"Bearer {API_KEY}"
}

# List of files to upload
files = {
    "adapter_config.json": open(os.path.join(MODEL_DIR, "adapter_config.json"), "rb"),
    "adapter_model.safetensors": open(os.path.join(MODEL_DIR, "adapter_model.safetensors"), "rb"),
    "fireworks.json": open(os.path.join(MODEL_DIR, "fireworks.json"), "rb"),
}

# Metadata
metadata = {
    "model_id": "resume-lora-test",  # Change this if you want a different name
    "base_model": "accounts/fireworks/models/mixtral-8x7b-instruct",
    "visibility": "private",  # can also be "public"
}

# Upload the model
response = requests.post(
    UPLOAD_URL,
    headers=headers,
    files={
        key: (key, file, "application/octet-stream")
        for key, file in files.items()
    },
    data=metadata
)

# Show result
print("✅ Upload status:", response.status_code)
print(response.json())

