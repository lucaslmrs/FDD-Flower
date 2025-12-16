import torch

print("="*60)
print("GPU Detection Status")
print("="*60)
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
print(f"PyTorch version: {torch.__version__}")
print(f"Device count: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    print(f"Device capability: {torch.cuda.get_device_capability(0)}")
    
    # Test GPU
    x = torch.randn(100, 100).cuda()
    print(f"Test tensor on GPU: {x.device}")
else:
    print("GPU not available - using CPU")

print("="*60)
