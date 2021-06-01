# Time Discretization-Invariant Safe Action Repetition for Policy Gradient Methods

This repository is the official implementation of

- *Seohong Park, Jaekyeom Kim, Gunhee Kim*. **Time Discretization-Invariant Safe Action Repetition for Policy Gradient Methods**. In *NeurIPS, 2021*.

It contains the implementations for SAR, FiGAR-C and base policy gradient algorithms (PPO, TRPO and A2C).

The code is based on [Stable Baselines3 (SB3)](https://github.com/DLR-RM/stable-baselines3) for PPO and A2C,
and [Stable Baselines (SB)](https://github.com/hill-a/stable-baselines) for TRPO.

## Requirements
- Python 3.7.8
- [MuJoCo](http://mujoco.org/) 1.5

## Run examples

### PPO and A2C (based on Stable Baselines3)

To install requirements:
```
cd sb3
pip install -r requirements.txt
pip install -e .
```

Train SAR-PPO on InvertedPendulum-v2 with δ = 0.01:
```
python repeat/main.py --env=InvertedPendulum-v2 --algo=ppo_rg --frame_skip=1 --dt=0.01 --max_t=0.05 --max_d=0.5
```
Train FiGAR-C-PPO on InvertedPendulum-v2 with δ = 0.01:
```
python repeat/main.py --env=InvertedPendulum-v2 --algo=ppo_rg --frame_skip=1 --dt=0.01 --max_t=0.05
```
Train PPO on InvertedPendulum-v2 with the original δ:
```
python repeat/main.py --env=InvertedPendulum-v2 --algo=ppo_rg
```
Train SAR-A2C on InvertedPendulum-v2 with δ = 0.01:
```
python repeat/main.py --env=InvertedPendulum-v2 --algo=a2c_rg --frame_skip=1 --dt=0.01 --max_t=0.05 --max_d=0.5
```
Train SAR-PPO on InvertedPendulum-v2 with δ = 0.002 and the "Action Noise" setting:
```
python repeat/main.py --env=InvertedPendulum-v2 --algo=ppo_rg --frame_skip=1 --dt=0.002 --max_t=0.05 --max_d=0.5 --anoise_type=action --anoise_prob=0.05 --anoise_std=3
```
Train SAR-PPO on InvertedPendulum-v2 with δ = 0.002 and the "External Force" setting:
```
python repeat/main.py --env=InvertedPendulum-v2 --algo=ppo_rg --frame_skip=1 --dt=0.002 --max_t=0.05 --max_d=0.5 --anoise_type=ext_f --anoise_prob=0.05 --anoise_std=300
```
Train SAR-PPO on InvertedPendulum-v2 with δ = 0.002 and the "Strong External Force (Perceptible)" setting:
```
python repeat/main.py --env=InvertedPendulum-v2 --algo=ppo_rg --frame_skip=1 --dt=0.002 --max_t=0.05 --max_d=0.5 --anoise_type=ext_fpc --anoise_prob=0.05 --anoise_std=1000
```

### TRPO (based on Stable Baselines)

To install requirements:
```
cd sb
pip install -r requirements.txt
pip install -e .
```
Train SAR-TRPO on InvertedPendulum-v2 with δ = 0.01:
```
python repeat/main.py --env=InvertedPendulum-v2 --frame_skip=1 --dt=0.01 --max_t=0.05 --max_d=0.5
```

## License

This codebase is licensed under the MIT License. See also `sb3/LICENSE_SB3` and `sb/LICENSE_SB`.