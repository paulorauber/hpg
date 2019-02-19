# Hindsight policy gradients

This software supplements the paper ["Hindsight policy gradients"](https://arxiv.org/abs/1711.06006).

The implementation focuses on clarity and flexibility rather than computational efficiency.

## Examples

Training an agent in a bit flipping environment (k = 8) using a weighted per-decision hindsight policy gradient estimator (HPG):

```bash
python3 hpg/experiments/pg.py with 'seed=1' 'env_name=flipbit_8' 'max_steps=9' 'batch_size=2' 'subgoals_per_episode=0' 'per_decision=True' 'eval_size=256' 'weighted=True' 'n_train_batches=5000' 'n_restarts=5' 'eval_freq=50' 'policy_hidden_layers=[256, 256]' 'policy_learning_rate=0.0005' 'use_hindsight=True' 'use_baseline=False' 
```

Training an agent in a bit flipping environment (k = 8) using a goal-conditional policy gradient estimator (GCPG):

```bash
python3 hpg/experiments/pg.py with 'seed=1' 'env_name=flipbit_8' 'max_steps=9' 'batch_size=2' 'eval_size=256' 'n_train_batches=5000' 'n_restarts=5' 'eval_freq=50' 'policy_hidden_layers=[256, 256]' 'policy_learning_rate=0.0005' 'use_hindsight=False' 'use_baseline=False' 
```

Combining the corresponding results into a single plot (see folder "results"):

```bash
python3 hpg/utils/analysis.py results
```

## Dependencies

- matplotlib
- numpy
- pandas
- sacred
- scipy
- seaborn
- tensorflow
