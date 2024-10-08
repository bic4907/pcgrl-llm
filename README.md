# PCGRL-jax

## Install

```
pip install -r requirements.txt
```

Then [install jax](https://jax.readthedocs.io/en/latest/installation.html):

## Training

To train a model, run:
```
python train.py
```
Arguments (pass these by running, e.g., `python train.py overwrite=True`):
- `overwrite`, bool, default=False`
    Whether to overwrite the model if it already exists.
- `render_freq`, int, default=100
    How often to render the environment.

During training, we render a few episodes to see how the model is doing (every `render_freq` updates). We use the same 
random seeds when resetting the environment, so that initial level layouts are the same between rounds of rendering.

## Debug

```bash
python experiment.py overwrite=True total_iterations=2 total_timesteps="int(1e2)" bypass_reward_path=bypass_reward
```
