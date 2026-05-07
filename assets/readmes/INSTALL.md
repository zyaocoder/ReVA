# Installation Instructions


## Installation

`requirements.txt` is the only official dependency specification for this repo.

```bash
conda create -n reva python=3.10 pip
conda activate reva
pip install -r requirements.txt
```

Optional:

```bash
pip install mamba-ssm causal-conv1d
```

The current official stack follows the pinned CUDA build in `requirements.txt`.

## Environment

```bash
export PYTHONPATH="$PWD/ReMoScene:$PWD/ReMoScene/src:$PYTHONPATH"
```