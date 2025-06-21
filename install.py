import subprocess

# List of non-default modules
modules_to_install = [
    'pandas',
    'numpy',
    'torch',
    'torchvision',
    'torchaudio',
    'scikit-learn',
    'matplotlib',
    'tqdm',
]

# Specify the index URL for PyTorch components
pytorch_index_url = 'https://download.pytorch.org/whl/cu121'

# Install each module
for module in modules_to_install:
    if 'torch' in module:
        # Install PyTorch components with the specified index URL
        subprocess.call(['pip3', 'install', module, f'--find-links={pytorch_index_url}'])
    else:
        # Install other modules
        subprocess.call(['pip3', 'install', module])

print("Installation completed.")


# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121