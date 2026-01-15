# IPEO-2025

This project provides tools for running inference using the provided notebooks. 

# IMPORTANT

Place the validation\_set.npz file in the `data/` directory

## Setup

There are two ways to set up the environment: using conda or using uv.

### Option 1: Using Conda

#### 1. Create the Conda Environment

Create the environment from the `environment.yml` file:

```bash
conda env create -f environment.yml
```

This will create a conda environment named `newenv` with Python 3.13 and all required dependencies.

#### 2. Activate the Environment

```bash
conda activate newenv
```

#### 3. Run the Inference Notebook

Start Jupyter and open the notebook:

```bash
jupyter notebook notebooks/inference.ipynb
```

Alternatively, you can use JupyterLab:

```bash
jupyter lab notebooks/inference.ipynb
```

### Option 2: Using UV

#### 1. Install UV

If you don't have uv installed, install it first:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Or on Windows:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### 2. Create the Environment

UV will automatically create and manage the virtual environment. Install all dependencies:

```bash
uv sync
```

This will create a virtual environment and install all dependencies from `pyproject.toml`, including PyTorch with CUDA 12.6 support.

#### 3. Install Jupyter Kernel (Development Dependencies)

Install the development dependencies which include Jupyter:

```bash
uv sync --group dev
```

#### 4. Run the Inference Notebook

Start a Jupyter server using uv:

```bash
uv run jupyter notebook notebooks/inference.ipynb
```

Or with JupyterLab:

```bash
uv run jupyter lab notebooks/inference.ipynb
```

## Project Structure

- `notebooks/` - Jupyter notebooks including `inference.ipynb`
- `src/` - Source code
- `data/` - Data directory
- `checkpoints/` - Model checkpoints
- `environment.yml` - Conda environment specification
- `pyproject.toml` - UV/pip project configuration

## Notes

- This project requires Python 3.13
- PyTorch is configured with CUDA 12.6 support when using uv
- All dependencies are listed in `requirements.txt` and `pyproject.toml`
