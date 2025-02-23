import torch  # ✅ Make sure torch is imported

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Torch CUDA Available: {torch.cuda.is_available()}")
print(f"Using CUDA Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"Current Device: {torch.cuda.current_device() if torch.cuda.is_available() else 'CPU'}")
