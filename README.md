# SAE Environment Setup Guide

This guide is for **Linux server environments (e.g., GPU machines)** where you are using a **conda virtual environment**.  
Goal: install **SAELens + SAEBench**, and ensure **GPU + CUDA** support works properly.

---

## 1. Environment Preparation

### 1.1 Create a conda environment (Python 3.10)

```bash
conda create -n saelens python=3.10 -y
```

Activate the environment:

```bash
conda activate saelens
```

Upgrade pip:

```bash
pip install --upgrade pip setuptools wheel
```

---

## 2. Check GPU Driver & CUDA Support

### 2.1 GPU status

```bash
nvidia-smi
```

Example output may include:

```
Driver Version: 575.57.08
CUDA Version: 12.9
```

### 2.2 Check PyTorch GPU availability

```python
import torch
print("torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version in torch:", torch.version.cuda)
print("GPU name:", torch.cuda.get_device_name(0))
```

---

## 3. Install PyTorch (CUDA 12.x)

Install the CUDA 12.8 compatible wheel:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

---

## 4. Install SAELens and SAEBench

### SAELens

```bash
pip install sae-lens
```

### SAEBench

```bash
pip install sae-bench
```

Optional version pin:

```bash
pip install sae-bench==0.3.0
```

### Test installation

```python
import torch
import sae_lens
import sae_bench

print("torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version in torch:", torch.version.cuda)
print("GPU name:", torch.cuda.get_device_name(0))

print("SAELens version:", sae_lens.__version__)
# For SAEBench version:
# pip show sae-bench
```

---

## 5. Recommended Project Structure

```
project_root/
├── README.md
├── test.py
├── requirements.txt
└── your_sae_work_directory/
```

---

## 6. Optional: `environment.yml` for reproducibility

```yaml
name: saelens
channels:
  - pytorch
  - nvidia
  - defaults
dependencies:
  - python=3.10
  - pip
  - pytorch
  - pytorch-cuda=12.1
  - torchvision
  - torchaudio
  - pip:
      - sae-lens
      - sae-bench
      - numpy
      - pandas
      - matplotlib
```

Create the environment:

```bash
conda env create -f environment.yml
conda activate saelens
```
