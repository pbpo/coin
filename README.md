# Coin RL Environment

This repository contains a reinforcement learning environment for cryptocurrency trading using a transformer-based policy. The environment loads multiple coin price series and provides a Sharpe ratio-based reward.

## Requirements
- Python 3.9+
- `stable-baselines3`
- `torch`
- `gym`

Install the dependencies with:
```bash
pip install -r requirements.txt
```

## Usage
Place your CSV price data under `crypto_15m` (each file named `<coin>_something.csv` and containing columns `timestamp, open, high, low, close, volume`). Then run:
```bash
python d.py
```

Training uses `VecNormalize` for observation and reward normalization and saves the trained PPO model and statistics.
