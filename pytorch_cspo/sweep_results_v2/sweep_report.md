# MiniGrid DDQN Ablation Sweep Report
Generated: 2026-09-11 22:10:34

## Overview
每个 trial 除了标注的改动外, 其他参数与基线 (`00_baseline`) 相同. 目的是隔离出
导致 Q 值坍缩 / 学不到策略的变量.

## Quick Ranking (by eval success rate)
| rank | trial | eval succ | train succ | final Q spread | eval Q spread | loss first → last |
|---|---|---|---|---|---|---|
| 1 | `dp_60k_seed0` | 20.0% (6/30) | 99.9% (1076/1077) | 0.1264 | 0.0881 | 0.0570 → 0.0026 |
| 2 | `dp_seed3` | 16.7% (5/30) | 98.8% (429/434) | 0.0389 | 0.0672 | 0.8034 → 0.0050 |
| 3 | `dp_norm1_affine_seed1` | 13.3% (4/30) | 97.3% (284/292) | 0.0578 | 0.0310 | 0.4243 → 0.0047 |
| 4 | `dp_norm1_affine_seed0` | 3.3% (1/30) | 99.2% (476/480) | 0.0356 | 0.0623 | 0.0570 → 0.0030 |
| 5 | `dp_seed0` | 0.0% (0/30) | 99.4% (522/525) | 0.0631 | 0.1156 | 0.0570 → 0.0033 |
| 6 | `dp_seed1` | 0.0% (0/30) | 97.3% (394/405) | 0.0302 | 0.1086 | 0.4243 → 0.0092 |
| 7 | `dp_seed2` | 0.0% (0/30) | 93.7% (178/190) | 0.0783 | 0.1606 | 0.0907 → 0.0107 |
| 8 | `dp_seed4` | 0.0% (0/30) | 94.8% (257/271) | 0.1099 | 0.1027 | 0.3556 → 0.0057 |
| 9 | `dp_60k_seed1` | 0.0% (0/30) | 99.3% (1016/1023) | 0.5465 | 0.0828 | 0.4243 → 0.0043 |
| 10 | `dp_no_elu_seed0` | 0.0% (0/30) | 99.6% (528/530) | 0.1519 | 0.0805 | 0.0695 → 0.0024 |
| 11 | `dp_no_elu_seed1` | 0.0% (0/30) | 99.6% (495/497) | 0.0287 | 0.0881 | 0.4693 → 0.0085 |
| 12 | `dp_eps_anneal_seed0` | 0.0% (0/30) | 98.9% (813/822) | 0.4093 | 0.1712 | 0.1508 → 0.0024 |
| 13 | `dp_eps_anneal_seed1` | 0.0% (0/30) | 98.4% (667/678) | 0.0314 | 0.0583 | 0.7632 → 0.0119 |
| 14 | `dp_no_replay_dup_seed0` | 0.0% (0/30) | 99.6% (493/495) | 0.0392 | 0.0691 | 0.0570 → 0.0070 |
| 15 | `dp_no_replay_dup_seed1` | 0.0% (0/30) | 89.8% (167/186) | 0.0122 | 0.0649 | 0.4243 → 0.0045 |

## Trial-by-trial detail

### `dp_seed0`
**Config**:
```json
{
  "env_id": "MiniGrid-DoorKey-5x5-v0",
  "fully_obs": false,
  "obs_size": 7,
  "attn_type": "dot_product",
  "use_elu_on_q": true,
  "pool": "max",
  "norm1_affine": false,
  "epochs": 30000,
  "replay_size": 9000,
  "batch_size": 64,
  "lr": 0.0005,
  "gamma": 0.99,
  "update_freq": 100,
  "maxsteps": 400,
  "eps_schedule": "fixed",
  "eps_start": 0.5,
  "eps_end": 0.05,
  "eps_decay_steps": 10000,
  "positive_multiplier": 50,
  "normalize_by": "none",
  "seed": 0
}
```

**Results**:
- Eval success rate: **0.0%** (0/30)
- Eval avg steps: 400.0
- Train success rate: 99.4% (522/525)
- Train min episode length: 7
- Positive-reward transitions in replay: 4700
- Loss trajectory: 0.0570 → 0.0033 (last-100 mean 0.0033)
- Gradient steps: 29936
- Final train Q spread: 0.0631
- Eval Q spread (mean over eval steps): 0.1156
- Eval Q spread (max over eval steps): 0.5444
- Q spread on 5 held-out replay states: ['0.0377', '0.3674', '0.0822', '0.3674', '0.3674']
- Wall time: 1.3 min

### `dp_seed1`
**Config**:
```json
{
  "env_id": "MiniGrid-DoorKey-5x5-v0",
  "fully_obs": false,
  "obs_size": 7,
  "attn_type": "dot_product",
  "use_elu_on_q": true,
  "pool": "max",
  "norm1_affine": false,
  "epochs": 30000,
  "replay_size": 9000,
  "batch_size": 64,
  "lr": 0.0005,
  "gamma": 0.99,
  "update_freq": 100,
  "maxsteps": 400,
  "eps_schedule": "fixed",
  "eps_start": 0.5,
  "eps_end": 0.05,
  "eps_decay_steps": 10000,
  "positive_multiplier": 50,
  "normalize_by": "none",
  "seed": 1
}
```

**Results**:
- Eval success rate: **0.0%** (0/30)
- Eval avg steps: 400.0
- Train success rate: 97.3% (394/405)
- Train min episode length: 8
- Positive-reward transitions in replay: 4382
- Loss trajectory: 0.4243 → 0.0092 (last-100 mean 0.0073)
- Gradient steps: 29936
- Final train Q spread: 0.0302
- Eval Q spread (mean over eval steps): 0.1086
- Eval Q spread (max over eval steps): 0.2103
- Q spread on 5 held-out replay states: ['0.3798', '0.1243', '0.3798', '0.0352', '0.3798']
- Wall time: 1.3 min

### `dp_seed2`
**Config**:
```json
{
  "env_id": "MiniGrid-DoorKey-5x5-v0",
  "fully_obs": false,
  "obs_size": 7,
  "attn_type": "dot_product",
  "use_elu_on_q": true,
  "pool": "max",
  "norm1_affine": false,
  "epochs": 30000,
  "replay_size": 9000,
  "batch_size": 64,
  "lr": 0.0005,
  "gamma": 0.99,
  "update_freq": 100,
  "maxsteps": 400,
  "eps_schedule": "fixed",
  "eps_start": 0.5,
  "eps_end": 0.05,
  "eps_decay_steps": 10000,
  "positive_multiplier": 50,
  "normalize_by": "none",
  "seed": 2
}
```

**Results**:
- Eval success rate: **0.0%** (0/30)
- Eval avg steps: 400.0
- Train success rate: 93.7% (178/190)
- Train min episode length: 17
- Positive-reward transitions in replay: 3200
- Loss trajectory: 0.0907 → 0.0107 (last-100 mean 0.0117)
- Gradient steps: 29936
- Final train Q spread: 0.0783
- Eval Q spread (mean over eval steps): 0.1606
- Eval Q spread (max over eval steps): 0.3411
- Q spread on 5 held-out replay states: ['0.0300', '0.1678', '0.1220', '0.1220', '0.1220']
- Wall time: 1.2 min

### `dp_seed3`
**Config**:
```json
{
  "env_id": "MiniGrid-DoorKey-5x5-v0",
  "fully_obs": false,
  "obs_size": 7,
  "attn_type": "dot_product",
  "use_elu_on_q": true,
  "pool": "max",
  "norm1_affine": false,
  "epochs": 30000,
  "replay_size": 9000,
  "batch_size": 64,
  "lr": 0.0005,
  "gamma": 0.99,
  "update_freq": 100,
  "maxsteps": 400,
  "eps_schedule": "fixed",
  "eps_start": 0.5,
  "eps_end": 0.05,
  "eps_decay_steps": 10000,
  "positive_multiplier": 50,
  "normalize_by": "none",
  "seed": 3
}
```

**Results**:
- Eval success rate: **16.7%** (5/30)
- Eval avg steps: 334.9
- Train success rate: 98.8% (429/434)
- Train min episode length: 8
- Positive-reward transitions in replay: 4550
- Loss trajectory: 0.8034 → 0.0050 (last-100 mean 0.0036)
- Gradient steps: 29936
- Final train Q spread: 0.0389
- Eval Q spread (mean over eval steps): 0.0672
- Eval Q spread (max over eval steps): 0.4374
- Q spread on 5 held-out replay states: ['0.3049', '0.0606', '0.4257', '0.0263', '0.0171']
- Wall time: 1.2 min

### `dp_seed4`
**Config**:
```json
{
  "env_id": "MiniGrid-DoorKey-5x5-v0",
  "fully_obs": false,
  "obs_size": 7,
  "attn_type": "dot_product",
  "use_elu_on_q": true,
  "pool": "max",
  "norm1_affine": false,
  "epochs": 30000,
  "replay_size": 9000,
  "batch_size": 64,
  "lr": 0.0005,
  "gamma": 0.99,
  "update_freq": 100,
  "maxsteps": 400,
  "eps_schedule": "fixed",
  "eps_start": 0.5,
  "eps_end": 0.05,
  "eps_decay_steps": 10000,
  "positive_multiplier": 50,
  "normalize_by": "none",
  "seed": 4
}
```

**Results**:
- Eval success rate: **0.0%** (0/30)
- Eval avg steps: 400.0
- Train success rate: 94.8% (257/271)
- Train min episode length: 12
- Positive-reward transitions in replay: 4412
- Loss trajectory: 0.3556 → 0.0057 (last-100 mean 0.0051)
- Gradient steps: 29936
- Final train Q spread: 0.1099
- Eval Q spread (mean over eval steps): 0.1027
- Eval Q spread (max over eval steps): 0.5213
- Q spread on 5 held-out replay states: ['0.0865', '0.4160', '0.4125', '0.0194', '0.4160']
- Wall time: 1.3 min

### `dp_60k_seed0`
**Config**:
```json
{
  "env_id": "MiniGrid-DoorKey-5x5-v0",
  "fully_obs": false,
  "obs_size": 7,
  "attn_type": "dot_product",
  "use_elu_on_q": true,
  "pool": "max",
  "norm1_affine": false,
  "epochs": 60000,
  "replay_size": 9000,
  "batch_size": 64,
  "lr": 0.0005,
  "gamma": 0.99,
  "update_freq": 100,
  "maxsteps": 400,
  "eps_schedule": "fixed",
  "eps_start": 0.5,
  "eps_end": 0.05,
  "eps_decay_steps": 10000,
  "positive_multiplier": 50,
  "normalize_by": "none",
  "seed": 0
}
```

**Results**:
- Eval success rate: **20.0%** (6/30)
- Eval avg steps: 321.7
- Train success rate: 99.9% (1076/1077)
- Train min episode length: 9
- Positive-reward transitions in replay: 4686
- Loss trajectory: 0.0570 → 0.0026 (last-100 mean 0.0038)
- Gradient steps: 59936
- Final train Q spread: 0.1264
- Eval Q spread (mean over eval steps): 0.0881
- Eval Q spread (max over eval steps): 0.6280
- Q spread on 5 held-out replay states: ['0.4572', '0.4572', '0.2440', '0.4572', '0.6280']
- Wall time: 2.5 min

### `dp_60k_seed1`
**Config**:
```json
{
  "env_id": "MiniGrid-DoorKey-5x5-v0",
  "fully_obs": false,
  "obs_size": 7,
  "attn_type": "dot_product",
  "use_elu_on_q": true,
  "pool": "max",
  "norm1_affine": false,
  "epochs": 60000,
  "replay_size": 9000,
  "batch_size": 64,
  "lr": 0.0005,
  "gamma": 0.99,
  "update_freq": 100,
  "maxsteps": 400,
  "eps_schedule": "fixed",
  "eps_start": 0.5,
  "eps_end": 0.05,
  "eps_decay_steps": 10000,
  "positive_multiplier": 50,
  "normalize_by": "none",
  "seed": 1
}
```

**Results**:
- Eval success rate: **0.0%** (0/30)
- Eval avg steps: 400.0
- Train success rate: 99.3% (1016/1023)
- Train min episode length: 8
- Positive-reward transitions in replay: 4453
- Loss trajectory: 0.4243 → 0.0043 (last-100 mean 0.0055)
- Gradient steps: 59936
- Final train Q spread: 0.5465
- Eval Q spread (mean over eval steps): 0.0828
- Eval Q spread (max over eval steps): 0.1662
- Q spread on 5 held-out replay states: ['0.0348', '0.5167', '0.0275', '0.3741', '0.0275']
- Wall time: 2.5 min

### `dp_no_elu_seed0`
**Config**:
```json
{
  "env_id": "MiniGrid-DoorKey-5x5-v0",
  "fully_obs": false,
  "obs_size": 7,
  "attn_type": "dot_product",
  "use_elu_on_q": false,
  "pool": "max",
  "norm1_affine": false,
  "epochs": 30000,
  "replay_size": 9000,
  "batch_size": 64,
  "lr": 0.0005,
  "gamma": 0.99,
  "update_freq": 100,
  "maxsteps": 400,
  "eps_schedule": "fixed",
  "eps_start": 0.5,
  "eps_end": 0.05,
  "eps_decay_steps": 10000,
  "positive_multiplier": 50,
  "normalize_by": "none",
  "seed": 0
}
```

**Results**:
- Eval success rate: **0.0%** (0/30)
- Eval avg steps: 400.0
- Train success rate: 99.6% (528/530)
- Train min episode length: 9
- Positive-reward transitions in replay: 4496
- Loss trajectory: 0.0695 → 0.0024 (last-100 mean 0.0041)
- Gradient steps: 29936
- Final train Q spread: 0.1519
- Eval Q spread (mean over eval steps): 0.0805
- Eval Q spread (max over eval steps): 0.1125
- Q spread on 5 held-out replay states: ['0.5361', '0.0492', '0.0351', '0.5361', '0.0440']
- Wall time: 1.2 min

### `dp_no_elu_seed1`
**Config**:
```json
{
  "env_id": "MiniGrid-DoorKey-5x5-v0",
  "fully_obs": false,
  "obs_size": 7,
  "attn_type": "dot_product",
  "use_elu_on_q": false,
  "pool": "max",
  "norm1_affine": false,
  "epochs": 30000,
  "replay_size": 9000,
  "batch_size": 64,
  "lr": 0.0005,
  "gamma": 0.99,
  "update_freq": 100,
  "maxsteps": 400,
  "eps_schedule": "fixed",
  "eps_start": 0.5,
  "eps_end": 0.05,
  "eps_decay_steps": 10000,
  "positive_multiplier": 50,
  "normalize_by": "none",
  "seed": 1
}
```

**Results**:
- Eval success rate: **0.0%** (0/30)
- Eval avg steps: 400.0
- Train success rate: 99.6% (495/497)
- Train min episode length: 9
- Positive-reward transitions in replay: 4534
- Loss trajectory: 0.4693 → 0.0085 (last-100 mean 0.0058)
- Gradient steps: 29936
- Final train Q spread: 0.0287
- Eval Q spread (mean over eval steps): 0.0881
- Eval Q spread (max over eval steps): 0.1696
- Q spread on 5 held-out replay states: ['0.5929', '0.1495', '0.0545', '0.4064', '0.4064']
- Wall time: 1.2 min

### `dp_norm1_affine_seed0`
**Config**:
```json
{
  "env_id": "MiniGrid-DoorKey-5x5-v0",
  "fully_obs": false,
  "obs_size": 7,
  "attn_type": "dot_product",
  "use_elu_on_q": true,
  "pool": "max",
  "norm1_affine": true,
  "epochs": 30000,
  "replay_size": 9000,
  "batch_size": 64,
  "lr": 0.0005,
  "gamma": 0.99,
  "update_freq": 100,
  "maxsteps": 400,
  "eps_schedule": "fixed",
  "eps_start": 0.5,
  "eps_end": 0.05,
  "eps_decay_steps": 10000,
  "positive_multiplier": 50,
  "normalize_by": "none",
  "seed": 0
}
```

**Results**:
- Eval success rate: **3.3%** (1/30)
- Eval avg steps: 386.9
- Train success rate: 99.2% (476/480)
- Train min episode length: 7
- Positive-reward transitions in replay: 4050
- Loss trajectory: 0.0570 → 0.0030 (last-100 mean 0.0045)
- Gradient steps: 29936
- Final train Q spread: 0.0356
- Eval Q spread (mean over eval steps): 0.0623
- Eval Q spread (max over eval steps): 0.5330
- Q spread on 5 held-out replay states: ['0.5330', '0.5330', '0.5330', '0.5330', '0.0945']
- Wall time: 1.2 min

### `dp_norm1_affine_seed1`
**Config**:
```json
{
  "env_id": "MiniGrid-DoorKey-5x5-v0",
  "fully_obs": false,
  "obs_size": 7,
  "attn_type": "dot_product",
  "use_elu_on_q": true,
  "pool": "max",
  "norm1_affine": true,
  "epochs": 30000,
  "replay_size": 9000,
  "batch_size": 64,
  "lr": 0.0005,
  "gamma": 0.99,
  "update_freq": 100,
  "maxsteps": 400,
  "eps_schedule": "fixed",
  "eps_start": 0.5,
  "eps_end": 0.05,
  "eps_decay_steps": 10000,
  "positive_multiplier": 50,
  "normalize_by": "none",
  "seed": 1
}
```

**Results**:
- Eval success rate: **13.3%** (4/30)
- Eval avg steps: 347.8
- Train success rate: 97.3% (284/292)
- Train min episode length: 12
- Positive-reward transitions in replay: 4000
- Loss trajectory: 0.4243 → 0.0047 (last-100 mean 0.0065)
- Gradient steps: 29936
- Final train Q spread: 0.0578
- Eval Q spread (mean over eval steps): 0.0310
- Eval Q spread (max over eval steps): 0.7031
- Q spread on 5 held-out replay states: ['0.3889', '0.2999', '0.3434', '0.0314', '0.6294']
- Wall time: 1.2 min

### `dp_eps_anneal_seed0`
**Config**:
```json
{
  "env_id": "MiniGrid-DoorKey-5x5-v0",
  "fully_obs": false,
  "obs_size": 7,
  "attn_type": "dot_product",
  "use_elu_on_q": true,
  "pool": "max",
  "norm1_affine": false,
  "epochs": 30000,
  "replay_size": 9000,
  "batch_size": 64,
  "lr": 0.0005,
  "gamma": 0.99,
  "update_freq": 100,
  "maxsteps": 400,
  "eps_schedule": "anneal",
  "eps_start": 1.0,
  "eps_end": 0.05,
  "eps_decay_steps": 10000,
  "positive_multiplier": 50,
  "normalize_by": "none",
  "seed": 0
}
```

**Results**:
- Eval success rate: **0.0%** (0/30)
- Eval avg steps: 400.0
- Train success rate: 98.9% (813/822)
- Train min episode length: 7
- Positive-reward transitions in replay: 6400
- Loss trajectory: 0.1508 → 0.0024 (last-100 mean 0.0017)
- Gradient steps: 29936
- Final train Q spread: 0.4093
- Eval Q spread (mean over eval steps): 0.1712
- Eval Q spread (max over eval steps): 0.4878
- Q spread on 5 held-out replay states: ['0.1490', '0.3972', '0.3879', '0.3972', '0.2994']
- Wall time: 1.3 min

### `dp_eps_anneal_seed1`
**Config**:
```json
{
  "env_id": "MiniGrid-DoorKey-5x5-v0",
  "fully_obs": false,
  "obs_size": 7,
  "attn_type": "dot_product",
  "use_elu_on_q": true,
  "pool": "max",
  "norm1_affine": false,
  "epochs": 30000,
  "replay_size": 9000,
  "batch_size": 64,
  "lr": 0.0005,
  "gamma": 0.99,
  "update_freq": 100,
  "maxsteps": 400,
  "eps_schedule": "anneal",
  "eps_start": 1.0,
  "eps_end": 0.05,
  "eps_decay_steps": 10000,
  "positive_multiplier": 50,
  "normalize_by": "none",
  "seed": 1
}
```

**Results**:
- Eval success rate: **0.0%** (0/30)
- Eval avg steps: 400.0
- Train success rate: 98.4% (667/678)
- Train min episode length: 7
- Positive-reward transitions in replay: 4850
- Loss trajectory: 0.7632 → 0.0119 (last-100 mean 0.0036)
- Gradient steps: 29936
- Final train Q spread: 0.0314
- Eval Q spread (mean over eval steps): 0.0583
- Eval Q spread (max over eval steps): 0.1199
- Q spread on 5 held-out replay states: ['0.4725', '0.4698', '0.0504', '0.0588', '0.4245']
- Wall time: 1.2 min

### `dp_no_replay_dup_seed0`
**Config**:
```json
{
  "env_id": "MiniGrid-DoorKey-5x5-v0",
  "fully_obs": false,
  "obs_size": 7,
  "attn_type": "dot_product",
  "use_elu_on_q": true,
  "pool": "max",
  "norm1_affine": false,
  "epochs": 30000,
  "replay_size": 9000,
  "batch_size": 64,
  "lr": 0.0005,
  "gamma": 0.99,
  "update_freq": 100,
  "maxsteps": 400,
  "eps_schedule": "fixed",
  "eps_start": 0.5,
  "eps_end": 0.05,
  "eps_decay_steps": 10000,
  "positive_multiplier": 1,
  "normalize_by": "none",
  "seed": 0
}
```

**Results**:
- Eval success rate: **0.0%** (0/30)
- Eval avg steps: 400.0
- Train success rate: 99.6% (493/495)
- Train min episode length: 9
- Positive-reward transitions in replay: 190
- Loss trajectory: 0.0570 → 0.0070 (last-100 mean 0.0067)
- Gradient steps: 29936
- Final train Q spread: 0.0392
- Eval Q spread (mean over eval steps): 0.0691
- Eval Q spread (max over eval steps): 0.4401
- Q spread on 5 held-out replay states: ['0.0812', '0.0137', '0.0323', '0.1089', '0.0212']
- Wall time: 1.3 min

### `dp_no_replay_dup_seed1`
**Config**:
```json
{
  "env_id": "MiniGrid-DoorKey-5x5-v0",
  "fully_obs": false,
  "obs_size": 7,
  "attn_type": "dot_product",
  "use_elu_on_q": true,
  "pool": "max",
  "norm1_affine": false,
  "epochs": 30000,
  "replay_size": 9000,
  "batch_size": 64,
  "lr": 0.0005,
  "gamma": 0.99,
  "update_freq": 100,
  "maxsteps": 400,
  "eps_schedule": "fixed",
  "eps_start": 0.5,
  "eps_end": 0.05,
  "eps_decay_steps": 10000,
  "positive_multiplier": 1,
  "normalize_by": "none",
  "seed": 1
}
```

**Results**:
- Eval success rate: **0.0%** (0/30)
- Eval avg steps: 400.0
- Train success rate: 89.8% (167/186)
- Train min episode length: 14
- Positive-reward transitions in replay: 85
- Loss trajectory: 0.4243 → 0.0045 (last-100 mean 0.0053)
- Gradient steps: 29936
- Final train Q spread: 0.0122
- Eval Q spread (mean over eval steps): 0.0649
- Eval Q spread (max over eval steps): 0.1498
- Q spread on 5 held-out replay states: ['0.0367', '0.0412', '0.0361', '0.0222', '0.4538']
- Wall time: 1.3 min

## Interpretation guide

- **eval Q spread < 0.01** → 网络输出常量, 完全坍缩.
- **eval Q spread ≈ 0.05, eval success ≈ 0%** → 略有差异但 argmax 无意义.
- **eval success ≥ 30%** → 该组合缓解了坍缩问题, 值得深入.
- **train min ep len < 100** → 训练过程中至少有过一次成功轨迹.
- **train success rate = 0** → 从未成功, 检查探索或环境.
- **对比同种子不同变量**: 找出 eval succ 明显上升的变量, 就是关键因子.
- **对比不同种子相同变量** (如 11, 15, 16): 若差异很大, 说明训练不稳定, 结论敏感.
