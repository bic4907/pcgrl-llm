# PCGRL-LLM

## Install

```
pip install -r requirements.txt
```
Then, install CUDA
### CUDA 11
```
pip install "jax[cuda11]"
```
### CUDA 12
```
pip install "jax[cuda12]"
```

## Training

To train a model, run:
```
python experiment.py
```
Arguments (pass these by running, e.g., `python experiment.py overwrite=True`):
- `overwrite`, bool, default=False`
    Whether to overwrite the model if it already exists.

## Debug

```bash
python experiment.py overwrite=True total_iterations=2 total_timesteps="int(1e2)" bypass_reward_path=bypass_reward
```
