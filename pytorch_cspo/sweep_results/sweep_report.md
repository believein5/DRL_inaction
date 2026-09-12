# MiniGrid DDQN Ablation Sweep Report
Generated: 2026-09-11 21:40:10

## Overview
每个 trial 除了标注的改动外, 其他参数与基线 (`00_baseline`) 相同. 目的是隔离出
导致 Q 值坍缩 / 学不到策略的变量.

## Quick Ranking (by eval success rate)
| rank | trial | eval succ | train succ | final Q spread | eval Q spread | loss first → last |
|---|---|---|---|---|---|---|
| 1 | `15_kitchen_sink_seed1` | 40.0% (12/30) | 93.0% (361/388) | 0.0211 | 0.0247 | 0.2589 → 0.0005 |
| 2 | `01_dotproduct_attn` | 26.7% (8/30) | 98.1% (424/432) | 0.3053 | 0.2165 | 0.0570 → 0.0032 |
| 3 | `11_kitchen_sink` | 6.7% (2/30) | 92.6% (300/324) | 0.4370 | 0.0480 | 0.1600 → 0.0006 |
| 4 | `00_baseline` | 3.3% (1/30) | 97.5% (308/316) | 0.0125 | 0.0638 | 0.2110 → 0.0065 |
| 5 | `02_no_elu_on_q` | 3.3% (1/30) | 93.5% (230/246) | 0.0152 | 0.1145 | 0.2283 → 0.0072 |
| 6 | `09_normalize_maxv` | 3.3% (1/30) | 93.9% (185/197) | 0.0918 | 0.0830 | 0.2523 → 0.0138 |
| 7 | `03_mean_pool` | 0.0% (0/30) | 48.4% (46/95) | 0.0161 | 0.0132 | 0.2038 → 0.0005 |
| 8 | `04_norm1_affine` | 0.0% (0/30) | 94.0% (249/265) | 0.3660 | 0.0713 | 0.2110 → 0.0046 |
| 9 | `05_no_replay_dup` | 0.0% (0/30) | 95.4% (272/285) | 0.0248 | 0.1778 | 0.2110 → 0.0034 |
| 10 | `06_eps_anneal` | 0.0% (0/30) | 69.7% (92/132) | 0.0540 | 0.0231 | 0.3842 → 0.0099 |
| 11 | `07_lower_lr` | 0.0% (0/30) | 88.3% (151/171) | 0.4364 | 0.2044 | 0.2110 → 0.0119 |
| 12 | `08_slower_target_update` | 0.0% (0/30) | 67.8% (78/115) | 0.0245 | 0.0139 | 0.2110 → 0.0141 |
| 13 | `10_dotproduct_no_elu_mean` | 0.0% (0/30) | 97.7% (294/301) | 0.0163 | 0.0493 | 0.0611 → 0.0131 |
| 14 | `12_fullyobs_baseline` | 0.0% (0/30) | 91.2% (228/250) | 0.0958 | 0.1008 | 0.0571 → 0.0027 |
| 15 | `13_fullyobs_kitchen_sink` | 0.0% (0/30) | 60.5% (69/114) | 0.0086 | 0.0420 | 0.1782 → 0.0055 |
| 16 | `14_empty5x5_kitchen_sink` | 0.0% (0/30) | 100.0% (3518/3518) | 0.0726 | 0.0889 | 0.1385 → 0.0040 |
| 17 | `16_kitchen_sink_seed2` | 0.0% (0/30) | 69.9% (93/133) | 0.0083 | 0.0983 | 0.1026 → 0.0002 |

## Trial-by-trial detail

### `00_baseline`
**Config**:
```json
{
  "env_id": "MiniGrid-DoorKey-5x5-v0",
  "fully_obs": false,
  "obs_size": 7,
  "attn_type": "additive",
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
- Eval success rate: **3.3%** (1/30)
- Eval avg steps: 386.9
- Train success rate: 97.5% (308/316)
- Train min episode length: 10
- Positive-reward transitions in replay: 4330
- Loss trajectory: 0.2110 → 0.0065 (last-100 mean 0.0053)
- Gradient steps: 29936
- Final train Q spread: 0.0125
- Eval Q spread (mean over eval steps): 0.0638
- Eval Q spread (max over eval steps): 0.5440
- Q spread on 5 held-out replay states: ['0.0380', '0.5132', '0.0398', '0.0325', '0.2428']
- Wall time: 1.3 min

### `01_dotproduct_attn`
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
- Eval success rate: **26.7%** (8/30)
- Eval avg steps: 295.6
- Train success rate: 98.1% (424/432)
- Train min episode length: 8
- Positive-reward transitions in replay: 4900
- Loss trajectory: 0.0570 → 0.0032 (last-100 mean 0.0029)
- Gradient steps: 29936
- Final train Q spread: 0.3053
- Eval Q spread (mean over eval steps): 0.2165
- Eval Q spread (max over eval steps): 0.5858
- Q spread on 5 held-out replay states: ['0.0732', '0.0185', '0.0617', '0.2841', '0.1064']
- Wall time: 1.3 min

### `02_no_elu_on_q`
**Config**:
```json
{
  "env_id": "MiniGrid-DoorKey-5x5-v0",
  "fully_obs": false,
  "obs_size": 7,
  "attn_type": "additive",
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
- Eval success rate: **3.3%** (1/30)
- Eval avg steps: 386.9
- Train success rate: 93.5% (230/246)
- Train min episode length: 11
- Positive-reward transitions in replay: 4007
- Loss trajectory: 0.2283 → 0.0072 (last-100 mean 0.0070)
- Gradient steps: 29936
- Final train Q spread: 0.0152
- Eval Q spread (mean over eval steps): 0.1145
- Eval Q spread (max over eval steps): 0.8175
- Q spread on 5 held-out replay states: ['0.8175', '0.0647', '0.7925', '0.7925', '0.0267']
- Wall time: 1.3 min

### `03_mean_pool`
**Config**:
```json
{
  "env_id": "MiniGrid-DoorKey-5x5-v0",
  "fully_obs": false,
  "obs_size": 7,
  "attn_type": "additive",
  "use_elu_on_q": true,
  "pool": "mean",
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
- Train success rate: 48.4% (46/95)
- Train min episode length: 29
- Positive-reward transitions in replay: 950
- Loss trajectory: 0.2038 → 0.0005 (last-100 mean 0.0058)
- Gradient steps: 29936
- Final train Q spread: 0.0161
- Eval Q spread (mean over eval steps): 0.0132
- Eval Q spread (max over eval steps): 0.0472
- Q spread on 5 held-out replay states: ['0.0133', '0.0148', '0.0133', '0.0110', '0.0148']
- Wall time: 1.3 min

### `04_norm1_affine`
**Config**:
```json
{
  "env_id": "MiniGrid-DoorKey-5x5-v0",
  "fully_obs": false,
  "obs_size": 7,
  "attn_type": "additive",
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
- Eval success rate: **0.0%** (0/30)
- Eval avg steps: 400.0
- Train success rate: 94.0% (249/265)
- Train min episode length: 12
- Positive-reward transitions in replay: 3800
- Loss trajectory: 0.2110 → 0.0046 (last-100 mean 0.0047)
- Gradient steps: 29936
- Final train Q spread: 0.3660
- Eval Q spread (mean over eval steps): 0.0713
- Eval Q spread (max over eval steps): 0.3981
- Q spread on 5 held-out replay states: ['0.2361', '0.0916', '0.1300', '0.2361', '0.2641']
- Wall time: 1.3 min

### `05_no_replay_dup`
**Config**:
```json
{
  "env_id": "MiniGrid-DoorKey-5x5-v0",
  "fully_obs": false,
  "obs_size": 7,
  "attn_type": "additive",
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
- Train success rate: 95.4% (272/285)
- Train min episode length: 12
- Positive-reward transitions in replay: 149
- Loss trajectory: 0.2110 → 0.0034 (last-100 mean 0.0052)
- Gradient steps: 29936
- Final train Q spread: 0.0248
- Eval Q spread (mean over eval steps): 0.1778
- Eval Q spread (max over eval steps): 0.5065
- Q spread on 5 held-out replay states: ['0.3967', '0.0588', '0.0463', '0.0524', '0.0621']
- Wall time: 1.3 min

### `06_eps_anneal`
**Config**:
```json
{
  "env_id": "MiniGrid-DoorKey-5x5-v0",
  "fully_obs": false,
  "obs_size": 7,
  "attn_type": "additive",
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
- Train success rate: 69.7% (92/132)
- Train min episode length: 7
- Positive-reward transitions in replay: 1550
- Loss trajectory: 0.3842 → 0.0099 (last-100 mean 0.0053)
- Gradient steps: 29936
- Final train Q spread: 0.0540
- Eval Q spread (mean over eval steps): 0.0231
- Eval Q spread (max over eval steps): 0.1001
- Q spread on 5 held-out replay states: ['0.6450', '0.0260', '0.0929', '0.3571', '0.3562']
- Wall time: 1.3 min

### `07_lower_lr`
**Config**:
```json
{
  "env_id": "MiniGrid-DoorKey-5x5-v0",
  "fully_obs": false,
  "obs_size": 7,
  "attn_type": "additive",
  "use_elu_on_q": true,
  "pool": "max",
  "norm1_affine": false,
  "epochs": 30000,
  "replay_size": 9000,
  "batch_size": 64,
  "lr": 0.0001,
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
- Train success rate: 88.3% (151/171)
- Train min episode length: 20
- Positive-reward transitions in replay: 2050
- Loss trajectory: 0.2110 → 0.0119 (last-100 mean 0.0136)
- Gradient steps: 29936
- Final train Q spread: 0.4364
- Eval Q spread (mean over eval steps): 0.2044
- Eval Q spread (max over eval steps): 0.4102
- Q spread on 5 held-out replay states: ['0.1917', '0.2699', '0.0336', '0.0353', '0.4102']
- Wall time: 1.3 min

### `08_slower_target_update`
**Config**:
```json
{
  "env_id": "MiniGrid-DoorKey-5x5-v0",
  "fully_obs": false,
  "obs_size": 7,
  "attn_type": "additive",
  "use_elu_on_q": true,
  "pool": "max",
  "norm1_affine": false,
  "epochs": 30000,
  "replay_size": 9000,
  "batch_size": 64,
  "lr": 0.0005,
  "gamma": 0.99,
  "update_freq": 500,
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
- Train success rate: 67.8% (78/115)
- Train min episode length: 35
- Positive-reward transitions in replay: 1800
- Loss trajectory: 0.2110 → 0.0141 (last-100 mean 0.0097)
- Gradient steps: 29936
- Final train Q spread: 0.0245
- Eval Q spread (mean over eval steps): 0.0139
- Eval Q spread (max over eval steps): 0.0315
- Q spread on 5 held-out replay states: ['0.8009', '0.3343', '0.0606', '0.0134', '0.0102']
- Wall time: 1.3 min

### `09_normalize_maxv`
**Config**:
```json
{
  "env_id": "MiniGrid-DoorKey-5x5-v0",
  "fully_obs": false,
  "obs_size": 7,
  "attn_type": "additive",
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
  "normalize_by": "maxv",
  "seed": 0
}
```

**Results**:
- Eval success rate: **3.3%** (1/30)
- Eval avg steps: 386.9
- Train success rate: 93.9% (185/197)
- Train min episode length: 14
- Positive-reward transitions in replay: 2450
- Loss trajectory: 0.2523 → 0.0138 (last-100 mean 0.0092)
- Gradient steps: 29936
- Final train Q spread: 0.0918
- Eval Q spread (mean over eval steps): 0.0830
- Eval Q spread (max over eval steps): 0.6947
- Q spread on 5 held-out replay states: ['0.0266', '0.6947', '0.6947', '0.0260', '0.2549']
- Wall time: 1.3 min

### `10_dotproduct_no_elu_mean`
**Config**:
```json
{
  "env_id": "MiniGrid-DoorKey-5x5-v0",
  "fully_obs": false,
  "obs_size": 7,
  "attn_type": "dot_product",
  "use_elu_on_q": false,
  "pool": "mean",
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
- Train success rate: 97.7% (294/301)
- Train min episode length: 9
- Positive-reward transitions in replay: 3450
- Loss trajectory: 0.0611 → 0.0131 (last-100 mean 0.0070)
- Gradient steps: 29936
- Final train Q spread: 0.0163
- Eval Q spread (mean over eval steps): 0.0493
- Eval Q spread (max over eval steps): 0.7081
- Q spread on 5 held-out replay states: ['0.4397', '0.5237', '0.5237', '0.0390', '0.5237']
- Wall time: 1.2 min

### `11_kitchen_sink`
**Config**:
```json
{
  "env_id": "MiniGrid-DoorKey-5x5-v0",
  "fully_obs": false,
  "obs_size": 7,
  "attn_type": "dot_product",
  "use_elu_on_q": false,
  "pool": "mean",
  "norm1_affine": true,
  "epochs": 30000,
  "replay_size": 9000,
  "batch_size": 64,
  "lr": 0.0001,
  "gamma": 0.99,
  "update_freq": 500,
  "maxsteps": 400,
  "eps_schedule": "anneal",
  "eps_start": 1.0,
  "eps_end": 0.05,
  "eps_decay_steps": 10000,
  "positive_multiplier": 1,
  "normalize_by": "none",
  "seed": 0
}
```

**Results**:
- Eval success rate: **6.7%** (2/30)
- Eval avg steps: 374.0
- Train success rate: 92.6% (300/324)
- Train min episode length: 7
- Positive-reward transitions in replay: 195
- Loss trajectory: 0.1600 → 0.0006 (last-100 mean 0.0028)
- Gradient steps: 29936
- Final train Q spread: 0.4370
- Eval Q spread (mean over eval steps): 0.0480
- Eval Q spread (max over eval steps): 0.7132
- Q spread on 5 held-out replay states: ['0.1084', '0.7132', '0.1367', '0.0336', '0.5014']
- Wall time: 1.3 min

### `12_fullyobs_baseline`
**Config**:
```json
{
  "env_id": "MiniGrid-DoorKey-5x5-v0",
  "fully_obs": true,
  "obs_size": 5,
  "attn_type": "additive",
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
- Train success rate: 91.2% (228/250)
- Train min episode length: 9
- Positive-reward transitions in replay: 4250
- Loss trajectory: 0.0571 → 0.0027 (last-100 mean 0.0047)
- Gradient steps: 29936
- Final train Q spread: 0.0958
- Eval Q spread (mean over eval steps): 0.1008
- Eval Q spread (max over eval steps): 0.2853
- Q spread on 5 held-out replay states: ['0.0447', '0.4784', '0.4784', '0.4471', '0.4471']
- Wall time: 1.3 min

### `13_fullyobs_kitchen_sink`
**Config**:
```json
{
  "env_id": "MiniGrid-DoorKey-5x5-v0",
  "fully_obs": true,
  "obs_size": 5,
  "attn_type": "dot_product",
  "use_elu_on_q": false,
  "pool": "mean",
  "norm1_affine": true,
  "epochs": 30000,
  "replay_size": 9000,
  "batch_size": 64,
  "lr": 0.0001,
  "gamma": 0.99,
  "update_freq": 500,
  "maxsteps": 400,
  "eps_schedule": "anneal",
  "eps_start": 1.0,
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
- Train success rate: 60.5% (69/114)
- Train min episode length: 8
- Positive-reward transitions in replay: 19
- Loss trajectory: 0.1782 → 0.0055 (last-100 mean 0.0011)
- Gradient steps: 29936
- Final train Q spread: 0.0086
- Eval Q spread (mean over eval steps): 0.0420
- Eval Q spread (max over eval steps): 0.6035
- Q spread on 5 held-out replay states: ['0.0544', '0.0674', '0.0123', '0.0407', '0.1628']
- Wall time: 1.3 min

### `14_empty5x5_kitchen_sink`
**Config**:
```json
{
  "env_id": "MiniGrid-Empty-5x5-v0",
  "fully_obs": false,
  "obs_size": 7,
  "attn_type": "dot_product",
  "use_elu_on_q": false,
  "pool": "mean",
  "norm1_affine": true,
  "epochs": 30000,
  "replay_size": 9000,
  "batch_size": 64,
  "lr": 0.0001,
  "gamma": 0.99,
  "update_freq": 500,
  "maxsteps": 400,
  "eps_schedule": "anneal",
  "eps_start": 1.0,
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
- Train success rate: 100.0% (3518/3518)
- Train min episode length: 5
- Positive-reward transitions in replay: 1352
- Loss trajectory: 0.1385 → 0.0040 (last-100 mean 0.0115)
- Gradient steps: 29936
- Final train Q spread: 0.0726
- Eval Q spread (mean over eval steps): 0.0889
- Eval Q spread (max over eval steps): 0.1053
- Q spread on 5 held-out replay states: ['0.1081', '0.0654', '0.0607', '0.1169', '0.0654']
- Wall time: 1.2 min

### `15_kitchen_sink_seed1`
**Config**:
```json
{
  "env_id": "MiniGrid-DoorKey-5x5-v0",
  "fully_obs": false,
  "obs_size": 7,
  "attn_type": "dot_product",
  "use_elu_on_q": false,
  "pool": "mean",
  "norm1_affine": true,
  "epochs": 30000,
  "replay_size": 9000,
  "batch_size": 64,
  "lr": 0.0001,
  "gamma": 0.99,
  "update_freq": 500,
  "maxsteps": 400,
  "eps_schedule": "anneal",
  "eps_start": 1.0,
  "eps_end": 0.05,
  "eps_decay_steps": 10000,
  "positive_multiplier": 1,
  "normalize_by": "none",
  "seed": 1
}
```

**Results**:
- Eval success rate: **40.0%** (12/30)
- Eval avg steps: 243.5
- Train success rate: 93.0% (361/388)
- Train min episode length: 7
- Positive-reward transitions in replay: 281
- Loss trajectory: 0.2589 → 0.0005 (last-100 mean 0.0033)
- Gradient steps: 29936
- Final train Q spread: 0.0211
- Eval Q spread (mean over eval steps): 0.0247
- Eval Q spread (max over eval steps): 0.7600
- Q spread on 5 held-out replay states: ['0.0204', '0.1022', '0.0251', '0.0612', '0.0251']
- Wall time: 1.2 min

### `16_kitchen_sink_seed2`
**Config**:
```json
{
  "env_id": "MiniGrid-DoorKey-5x5-v0",
  "fully_obs": false,
  "obs_size": 7,
  "attn_type": "dot_product",
  "use_elu_on_q": false,
  "pool": "mean",
  "norm1_affine": true,
  "epochs": 30000,
  "replay_size": 9000,
  "batch_size": 64,
  "lr": 0.0001,
  "gamma": 0.99,
  "update_freq": 500,
  "maxsteps": 400,
  "eps_schedule": "anneal",
  "eps_start": 1.0,
  "eps_end": 0.05,
  "eps_decay_steps": 10000,
  "positive_multiplier": 1,
  "normalize_by": "none",
  "seed": 2
}
```

**Results**:
- Eval success rate: **0.0%** (0/30)
- Eval avg steps: 400.0
- Train success rate: 69.9% (93/133)
- Train min episode length: 13
- Positive-reward transitions in replay: 36
- Loss trajectory: 0.1026 → 0.0002 (last-100 mean 0.0017)
- Gradient steps: 29936
- Final train Q spread: 0.0083
- Eval Q spread (mean over eval steps): 0.0983
- Eval Q spread (max over eval steps): 0.7761
- Q spread on 5 held-out replay states: ['0.0702', '0.1055', '0.0504', '0.0504', '0.0786']
- Wall time: 1.2 min

## Interpretation guide

- **eval Q spread < 0.01** → 网络输出常量, 完全坍缩.
- **eval Q spread ≈ 0.05, eval success ≈ 0%** → 略有差异但 argmax 无意义.
- **eval success ≥ 30%** → 该组合缓解了坍缩问题, 值得深入.
- **train min ep len < 100** → 训练过程中至少有过一次成功轨迹.
- **train success rate = 0** → 从未成功, 检查探索或环境.
- **对比同种子不同变量**: 找出 eval succ 明显上升的变量, 就是关键因子.
- **对比不同种子相同变量** (如 11, 15, 16): 若差异很大, 说明训练不稳定, 结论敏感.
