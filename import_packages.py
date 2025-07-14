import importlib
import subprocess
import sys
import warnings
import os

# List of libraries and optional custom import names
libraries = {
    "pandas": "pd",
    "matplotlib.pyplot": "plt",
    "os": None,
    "glob": "gb",
    "tqdm.auto": "tqdm",
    "torch": None,
    "torch.nn": "nn",
    "torch.optim": "optim",
    "torch.utils.data": ["Dataset", "DataLoader", "RandomSampler"],
    "torchmetrics.classification": ["MulticlassAccuracy"],
    "torchinfo":['summary'],
    "torchvision.datasets": "datasets",
    "torchvision.transforms": "transforms",
    "PIL.Image": "Image",
}

# Helper to install a package
def install_package(package_name):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
    except Exception as e:
        print(f"❌ Failed to install {package_name}: {e}")

# Load libraries
successfully_loaded = []

for module_path, alias in libraries.items():
    base_package = module_path.split('.')[0]
    try:
        module = importlib.import_module(module_path)
        if isinstance(alias, str):
            globals()[alias] = module
        elif isinstance(alias, list):
            for item in alias:
                globals()[item] = getattr(module, item)
        elif alias is None:
            globals()[base_package] = module
        successfully_loaded.append(module_path)
    except ImportError:
        print(f"📦 {module_path} not found. Installing...")
        install_package(base_package)
        try:
            module = importlib.import_module(module_path)
            if isinstance(alias, str):
                globals()[alias] = module
            elif isinstance(alias, list):
                for item in alias:
                    globals()[item] = getattr(module, item)
            elif alias is None:
                globals()[base_package] = module
            successfully_loaded.append(module_path)
        except Exception as e:
            print(f"❌ Failed to load {module_path} after install: {e}")

# Suppress warnings
warnings.filterwarnings("ignore")


# Final message
print("\n✅ Successfully loaded libraries:")
for lib in successfully_loaded:
    print(f"  - {lib}")
