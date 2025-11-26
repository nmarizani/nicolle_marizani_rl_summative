## AI-Veins: Reinforcement Learning for Healthcare Resource Allocation

AI-Veins is a reinforcement-learning system designed to optimize varicose-vein screening workflows in low-resource African healthcare settings. The agent allocates nurses, Doppler machines, and mobile units to maximize early detections while minimizing severe progressions and cost.

## Project Overview

Custom 35-dim healthcare environment

12 discrete actions (resource allocation)

Reward range: +200 to –200

Models implemented: DQN, PPO, A2C, REINFORCE

Includes hyperparameter tuning, Pygame visualization, and a 90-day simulation loop

## Why This Matters

African clinics often face:

- Limited Doppler machines

- Overloaded nurses

- High-risk patients progressing unnoticed

This RL system supports SDG 3 by optimizing scarce resources for early intervention.

## Project Structure

project/
├── healthcare_env.py
├── visualize.py
├── train_dqn.py
├── train_ppo.py
├── train_a2c.py
├── train_reinforce.py
├── compare_models.py
├── create_report_plots.py
├── saved_models/
└── logs/

## Installation
pip install --user -r requirements.txt

python verify_setup.py

## Test environment
python healthcare_env.py

## Run baseline random agent
python test_random_agent.py

## Train a model
python train_ppo.py

## Visualize a trained model
python visualize.py saved_models/ppo/ppo_model.zip

## Best Model Summary

| Rank | Model      | Reward | Severe Cases | Notes                    |
|------|-----------|--------|--------------|--------------------------|
| 1    | A2C       | 10082  | 0.2          | Most stable, best balance|
| 2    | DQN       | 9060   | 0            | Highest detections       |
| 3    | REINFORCE | 8724   | 0.5          | Simple & effective       |
| 4    | PPO       | 7744   | 0.3          | Good baseline            |

## Run the Best Model (A2C)
python visualize.py saved_models/a2c/a2c_model.zip

## Or evaluate all models:

python compare_models.py

## Demo Video: Add your link here.