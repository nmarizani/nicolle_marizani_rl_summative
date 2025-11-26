import gymnasium as gym
import healthcare_env
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import os
import numpy as np

# Create directories
os.makedirs("saved_models/a2c", exist_ok=True)
os.makedirs("logs/a2c", exist_ok=True)

def train_a2c(
    total_timesteps=100000,
    learning_rate=0.0007,
    n_steps=5,
    gamma=0.99,
    gae_lambda=1.0,
    ent_coef=0.0,
    vf_coef=0.5,
    save_name="a2c_model"
):
    """
    Train A2C agent with specified hyperparameters
    
    Args:
        total_timesteps: Total training steps
        learning_rate: Learning rate
        n_steps: Steps before update
        gamma: Discount factor
        gae_lambda: GAE lambda
        ent_coef: Entropy coefficient
        vf_coef: Value function coefficient
        save_name: Model save name
    """
    
    print("="*60)
    print("TRAINING A2C AGENT")
    print("="*60)
    print(f"Total timesteps: {total_timesteps}")
    print(f"Learning rate: {learning_rate}")
    print(f"N steps: {n_steps}")
    print(f"Gamma: {gamma}")
    print(f"GAE Lambda: {gae_lambda}")
    print("="*60)
    
    # Create environment
    env = gym.make('HealthcareEnv-v0')
    env = Monitor(env, filename=f"logs/a2c/{save_name}")
    
    # Create evaluation environment
    eval_env = gym.make('HealthcareEnv-v0')
    eval_env = Monitor(eval_env)
    
    # Create A2C model
    model = A2C(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        gamma=gamma,
        gae_lambda=gae_lambda,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        verbose=1,
        tensorboard_log=f"logs/a2c/{save_name}_tb"
    )
    
    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"saved_models/a2c/{save_name}_best",
        log_path=f"logs/a2c/{save_name}_eval",
        eval_freq=5000,
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=f"saved_models/a2c/{save_name}_checkpoints",
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
    model.save(f"saved_models/a2c/{save_name}")
    print(f"\n✓ Model saved to saved_models/a2c/{save_name}.zip")
    
    # Close environments
    env.close()
    eval_env.close()
    
    return model

def evaluate_model(model_path, num_episodes=10):
    """Evaluate trained model"""
    print("\n" + "="*60)
    print("EVALUATING A2C MODEL")
    print("="*60)
    
    model = A2C.load(model_path)
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
    
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average Detections: {np.mean(episode_detections):.2f}")
    print(f"Average Severe Cases: {np.mean(episode_severe):.2f}")
    print("="*60)
    
    return episode_rewards, episode_detections, episode_severe

def hyperparameter_tuning():
    """Run hyperparameter tuning with 10+ configurations"""
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING - A2C")
    print("Running 10 different configurations...")
    print("="*60)
    
    configs = [
        # Config 1: Default
        {"learning_rate": 0.0007, "n_steps": 5, "gamma": 0.99, "gae_lambda": 1.0, "ent_coef": 0.0},
        # Config 2: Lower LR
        {"learning_rate": 0.0003, "n_steps": 5, "gamma": 0.99, "gae_lambda": 1.0, "ent_coef": 0.0},
        # Config 3: Higher LR
        {"learning_rate": 0.001, "n_steps": 5, "gamma": 0.99, "gae_lambda": 1.0, "ent_coef": 0.0},
        # Config 4: More steps
        {"learning_rate": 0.0007, "n_steps": 10, "gamma": 0.99, "gae_lambda": 1.0, "ent_coef": 0.0},
        # Config 5: Even more steps
        {"learning_rate": 0.0007, "n_steps": 20, "gamma": 0.99, "gae_lambda": 1.0, "ent_coef": 0.0},
        # Config 6: Different gamma
        {"learning_rate": 0.0007, "n_steps": 5, "gamma": 0.95, "gae_lambda": 1.0, "ent_coef": 0.0},
        # Config 7: Higher gamma
        {"learning_rate": 0.0007, "n_steps": 5, "gamma": 0.995, "gae_lambda": 1.0, "ent_coef": 0.0},
        # Config 8: With entropy
        {"learning_rate": 0.0007, "n_steps": 5, "gamma": 0.99, "gae_lambda": 1.0, "ent_coef": 0.01},
        # Config 9: Different GAE
        {"learning_rate": 0.0007, "n_steps": 5, "gamma": 0.99, "gae_lambda": 0.95, "ent_coef": 0.0},
        # Config 10: Combined
        {"learning_rate": 0.001, "n_steps": 10, "gamma": 0.995, "gae_lambda": 0.95, "ent_coef": 0.01},
    ]
    
    results = []
    
    for i, config in enumerate(configs):
        print(f"\n{'='*60}")
        print(f"Configuration {i+1}/10")
        print(f"LR={config['learning_rate']}, Steps={config['n_steps']}, "
              f"Gamma={config['gamma']}, GAE={config['gae_lambda']}, "
              f"Ent={config['ent_coef']}")
        print(f"{'='*60}")
        
        model = train_a2c(
            total_timesteps=50000,
            learning_rate=config['learning_rate'],
            n_steps=config['n_steps'],
            gamma=config['gamma'],
            gae_lambda=config['gae_lambda'],
            ent_coef=config['ent_coef'],
            save_name=f"a2c_tune_{i+1}"
        )
        
        rewards, detections, severe = evaluate_model(
            f"saved_models/a2c/a2c_tune_{i+1}.zip",
            num_episodes=5
        )
        
        results.append({
            'config': config,
            'avg_reward': np.mean(rewards),
            'avg_detections': np.mean(detections),
            'avg_severe': np.mean(severe)
        })
    
    # Find best
    best_idx = np.argmax([r['avg_reward'] for r in results])
    best_config = results[best_idx]
    
    print("\n" + "="*60)
    print("BEST CONFIGURATION FOUND")
    print("="*60)
    print(f"Configuration {best_idx + 1}:")
    for key, value in best_config['config'].items():
        print(f"  {key}: {value}")
    print(f"  Average Reward: {best_config['avg_reward']:.2f}")
    print(f"  Average Detections: {best_config['avg_detections']:.2f}")
    print("="*60)
    
    # Save results
    with open("saved_models/a2c/tuning_results.txt", "w") as f:
        f.write("A2C Hyperparameter Tuning Results\n")
        f.write("="*60 + "\n\n")
        for i, result in enumerate(results):
            f.write(f"Configuration {i+1}:\n")
            f.write(f"  {result['config']}\n")
            f.write(f"  Avg Reward: {result['avg_reward']:.2f}\n\n")
        f.write(f"\nBest: Configuration {best_idx + 1}\n")
    
    print("\n✓ Results saved to saved_models/a2c/tuning_results.txt")
    
    return results, best_config

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "tune":
        hyperparameter_tuning()
    else:
        print("\nTraining A2C with default hyperparameters...")
        print("For hyperparameter tuning, run: python train_a2c.py tune\n")
        
        model = train_a2c(
            total_timesteps=100000,
            save_name="a2c_model"
        )
        
        print("\nEvaluating trained model...")
        evaluate_model("saved_models/a2c/a2c_model.zip", num_episodes=10)
        
        print("\n✓ Training complete!")
        print("To visualize: python visualize.py saved_models/a2c/a2c_model.zip")