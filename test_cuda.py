import torch
import os

print("="*60)
print("CUDA DIAGNOSTICS")
print("="*60)

print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA compiled version: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    print(f"Device capability: {torch.cuda.get_device_capability(0)}")
else:
    print("\n❌ CUDA not available!")
    print("\nTrying to get more details...")
    try:
        torch.cuda.init()
        print("CUDA initialized")
    except Exception as e:
        print(f"Error initializing CUDA: {e}")
        
print(f"\nLD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'Not set')}")
print(f"CUDA_HOME: {os.environ.get('CUDA_HOME', 'Not set')}")
print("="*60)

