import torch
from safetensors.torch import save_file

# Simulated LoRA weights for testing
weights = {
    "base_model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.randn(16, 4096),
    "base_model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.randn(4096, 16),
    "base_model.model.layers.0.self_attn.v_proj.lora_A.weight": torch.randn(16, 4096),
    "base_model.model.layers.0.self_attn.v_proj.lora_B.weight": torch.randn(4096, 16),
}

# Save to dummy safetensors file
save_file(weights, "adapter_model.safetensors")
print("✅ Dummy LoRA adapter_model.safetensors created!")
