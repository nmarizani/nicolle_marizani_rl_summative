"""
train_dqn.py - Train DQN (Deep Q-Network) agent
Value-based reinforcement learning algorithm
"""

import gymnasium as gym
import healthcare_env
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import os
import numpy as np
import matplotlib.pyplot as plt

# Create directories
os.makedirs("saved_models/dqn", exist_ok=True)
os.makedirs("logs/dqn", exist_ok=True)

def train_dqn(
    total_timesteps=100000,
    learning_rate=0.001,
    buffer_size=50000,
    learning_starts=1000,
    batch_size=32,
    gamma=0.99,
    exploration_fraction=0.1,
    exploration_final_eps=0.05,
    target_update_interval=1000,
    train_freq=4,
    save_name="dqn_model"
):
    """
    Train DQN agent with specified hyperparameters
    
    Args:
        total_timesteps: Total training steps
        learning_rate: Learning rate for optimizer
        buffer_size: Size of replay buffer
        learning_starts: Start learning after N steps
        batch_size: Minibatch size for training
        gamma: Discount factor
        exploration_fraction: Fraction of training for exploration
        exploration_final_eps: Final exploration epsilon
        target_update_interval: Update target network every N steps
        train_freq: Update model every N steps
        save_name: Model save name
    """
    
    print("="*60)
    print("TRAINING DQN AGENT")
    print("="*60)
    print(f"Total timesteps: {total_timesteps}")
    print(f"Learning rate: {learning_rate}")
    print(f"Buffer size: {buffer_size}")
    print(f"Gamma: {gamma}")
    print(f"Batch size: {batch_size}")
    print("="*60)
    
    # Create environment
    env = gym.make('HealthcareEnv-v0')
    env = Monitor(env, filename=f"logs/dqn/{save_name}")
    
    # Create evaluation environment
    eval_env = gym.make('HealthcareEnv-v0')
    eval_env = Monitor(eval_env)
    
    # Create DQN model
    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        batch_size=batch_size,
        gamma=gamma,
        exploration_fraction=exploration_fraction,
        exploration_final_eps=exploration_final_eps,
        target_update_interval=target_update_interval,
        train_freq=train_freq,
        verbose=1,
        tensorboard_log=f"logs/dqn/{save_name}_tb"
    )
    
    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"saved_models/dqn/{save_name}_best",
        log_path=f"logs/dqn/{save_name}_eval",
        eval_freq=5000,
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=f"saved_models/dqn/{save_name}_checkpoints",
        name_prefix=save_name
    )
    
    # Train
    print("\nStarting training...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )
    
    # Save final model
    model.save(f"saved_models/dqn/{save_name}")
    print(f"\n✓ Model saved to saved_models/dqn/{save_name}.zip")
    
    # Close environments
    env.close()
    eval_env.close()
    
    return model

def evaluate_model(model_path, num_episodes=10):
    """Evaluate trained model"""
    print("\n" + "="*60)
    print("EVALUATING DQN MODEL")
    print("="*60)
    
    # Load model
    model = DQN.load(model_path)
    
    # Create environment
    env = gym.make('HealthcareEnv-v0')
    
    episode_rewards = []
    episode_detections = []
    episode_severe = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        
        episode_rewards.append(total_reward)
        episode_detections.append(info['total_detections'])
        episode_severe.append(info['severe_progressions'])
        
        print(f"Episode {episode+1}: Reward={total_reward:.2f}, "
              f"Detections={info['total_detections']}, "
              f"Severe={info['severe_progressions']}")
    
    env.close()
    
    # Summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average Detections: {np.mean(episode_detections):.2f}")
    print(f"Average Severe Cases: {np.mean(episode_severe):.2f}")
    print("="*60)
    
    return episode_rewards, episode_detections, episode_severe

def hyperparameter_tuning():
    """
    Run multiple training runs with different hyperparameters
    Required: At least 10 different combinations
    """
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING - DQN")
    print("Running 10 different configurations...")
    print("="*60)
    
    configs = [
        # Config 1: Default
        {"learning_rate": 0.001, "gamma": 0.99, "batch_size": 32, "buffer_size": 50000},
        # Config 2: Lower learning rate
        {"learning_rate": 0.0005, "gamma": 0.99, "batch_size": 32, "buffer_size": 50000},
        # Config 3: Higher learning rate
        {"learning_rate": 0.002, "gamma": 0.99, "batch_size": 32, "buffer_size": 50000},
        # Config 4: Different gamma
        {"learning_rate": 0.001, "gamma": 0.95, "batch_size": 32, "buffer_size": 50000},
        # Config 5: Higher gamma
        {"learning_rate": 0.001, "gamma": 0.995, "batch_size": 32, "buffer_size": 50000},
        # Config 6: Larger batch
        {"learning_rate": 0.001, "gamma": 0.99, "batch_size": 64, "buffer_size": 50000},
        # Config 7: Smaller batch
        {"learning_rate": 0.001, "gamma": 0.99, "batch_size": 16, "buffer_size": 50000},
        # Config 8: Larger buffer
        {"learning_rate": 0.001, "gamma": 0.99, "batch_size": 32, "buffer_size": 100000},
        # Config 9: Combined adjustments
        {"learning_rate": 0.0005, "gamma": 0.995, "batch_size": 64, "buffer_size": 100000},
        # Config 10: Aggressive settings
        {"learning_rate": 0.002, "gamma": 0.95, "batch_size": 16, "buffer_size": 30000},
    ]
    
    results = []
    
    for i, config in enumerate(configs):
        print(f"\n{'='*60}")
        print(f"Configuration {i+1}/10")
        print(f"LR={config['learning_rate']}, Gamma={config['gamma']}, "
              f"Batch={config['batch_size']}, Buffer={config['buffer_size']}")
        print(f"{'='*60}")
        
        # Train with reduced timesteps for tuning
        model = train_dqn(
            total_timesteps=50000,  # Reduced for faster tuning
            learning_rate=config['learning_rate'],
            gamma=config['gamma'],
            batch_size=config['batch_size'],
            buffer_size=config['buffer_size'],
            save_name=f"dqn_tune_{i+1}"
        )
        
        # Evaluate
        rewards, detections, severe = evaluate_model(
            f"saved_models/dqn/dqn_tune_{i+1}.zip",
            num_episodes=5
        )
        
        results.append({
            'config': config,
            'avg_reward': np.mean(rewards),
            'avg_detections': np.mean(detections),
            'avg_severe': np.mean(severe)
        })
    
    # Find best configuration
    best_idx = np.argmax([r['avg_reward'] for r in results])
    best_config = results[best_idx]
    
    print("\n" + "="*60)
    print("BEST CONFIGURATION FOUND")
    print("="*60)
    print(f"Configuration {best_idx + 1}:")
    print(f"  Learning Rate: {best_config['config']['learning_rate']}")
    print(f"  Gamma: {best_config['config']['gamma']}")
    print(f"  Batch Size: {best_config['config']['batch_size']}")
    print(f"  Buffer Size: {best_config['config']['buffer_size']}")
    print(f"  Average Reward: {best_config['avg_reward']:.2f}")
    print(f"  Average Detections: {best_config['avg_detections']:.2f}")
    print(f"  Average Severe Cases: {best_config['avg_severe']:.2f}")
    print("="*60)
    
    # Save results
    with open("saved_models/dqn/tuning_results.txt", "w") as f:
        f.write("DQN Hyperparameter Tuning Results\n")
        f.write("="*60 + "\n\n")
        for i, result in enumerate(results):
            f.write(f"Configuration {i+1}:\n")
            f.write(f"  {result['config']}\n")
            f.write(f"  Avg Reward: {result['avg_reward']:.2f}\n")
            f.write(f"  Avg Detections: {result['avg_detections']:.2f}\n")
            f.write(f"  Avg Severe: {result['avg_severe']:.2f}\n\n")
        f.write(f"\nBest: Configuration {best_idx + 1}\n")
    
    print("\n✓ Tuning results saved to saved_models/dqn/tuning_results.txt")
    
    return results, best_config

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "tune":
        # Hyperparameter tuning mode
        hyperparameter_tuning()
    else:
        # Standard training mode
        print("\nTraining DQN with default hyperparameters...")
        print("For hyperparameter tuning, run: python train_dqn.py tune\n")
        
        # Train with default settings
        model = train_dqn(
            total_timesteps=100000,
            save_name="dqn_model"
        )
        
        # Evaluate
        print("\nEvaluating trained model...")
        evaluate_model("saved_models/dqn/dqn_model.zip", num_episodes=10)
        
        print("\n✓ Training complete!")
        print("To visualize: python visualize.py saved_models/dqn/dqn_model.zip")