# Hindsight policy gradients

This software supplements the paper ["Hindsight policy gradients"](https://arxiv.org/abs/1711.06006).

The implementation focuses on clarity and flexibility rather than computational efficiency.

## Examples

Training an agent in a bit flipping environment (k = 8) using a weighted per-decision hindsight policy gradient estimator (HPG):

```bash
python3 hpg/scripts/run.py hpg/examples/flipbit8/flipbit8_bs2_hpg
```

Training an agent in a bit flipping environment (k = 8) using a goal-conditional policy gradient estimator (GCPG):

```bash
python3 hpg/scripts/run.py hpg/examples/flipbit8/flipbit8_bs2_gcpg
```

Combining the corresponding results into a single plot (see folder "results/flipbit8_bs2"):

```bash
mkdir -p results/flipbit8_bs2
cp -r hpg/examples/flipbit8/flipbit8_bs2_hpg hpg/examples/flipbit8/flipbit8_bs2_gcpg results/flipbit8_bs2
python3 hpg/scripts/analysis.py results/flipbit8_bs2
```

## Dependencies

- matplotlib (2.1.1)
- numpy (1.17.2)
- pandas (0.23.4)
- scipy (1.3.0)
- seaborn (0.9.0)
- tensorflow (1.12.0)
- gym (0.13.1)
- atari-py (?, https://github.com/openai/atari-py)
- mujoco-py (2.0.2.6, https://github.com/openai/mujoco-py)
- ray (0.7.2)
