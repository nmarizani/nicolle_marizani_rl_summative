"""
train_reinforce.py - Train REINFORCE agent
Monte Carlo policy gradient algorithm (custom implementation)
"""

import gymnasium as gym
import healthcare_env
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from collections import deque

# Create directories
os.makedirs("saved_models/reinforce", exist_ok=True)
os.makedirs("logs/reinforce", exist_ok=True)

class PolicyNetwork(nn.Module):
    """Simple neural network for policy"""
    def __init__(self, obs_dim, action_dim, hidden_size=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.softmax(x, dim=-1)

class REINFORCEAgent:
    """REINFORCE algorithm implementation"""
    def __init__(
        self,
        obs_dim,
        action_dim,
        learning_rate=0.001,
        gamma=0.99,
        hidden_size=128
    ):
        self.gamma = gamma
        self.policy = PolicyNetwork(obs_dim, action_dim, hidden_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        self.saved_log_probs = []
        self.rewards = []
        
    def select_action(self, state, deterministic=False):
        """Select action from policy"""
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.policy(state)
        
        if deterministic:
            action = torch.argmax(probs, dim=1).item()
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            self.saved_log_probs.append(dist.log_prob(action))
            action = action.item()
        
        return action
    
    def update(self):
        """Update policy using REINFORCE"""
        # Calculate returns (discounted rewards)
        returns = []
        R = 0
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        
        # Normalize returns
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Calculate loss
        policy_loss = []
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
        
        # Update policy
        self.optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        
        # Clear memory
        self.saved_log_probs = []
        self.rewards = []
        
        return policy_loss.item()
    
    def save(self, filepath):
        """Save model"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)
    
    def load(self, filepath):
        """Load model"""
        checkpoint = torch.load(filepath)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

def train_reinforce(
    total_episodes=1000,
    learning_rate=0.001,
    gamma=0.99,
    hidden_size=128,
    save_name="reinforce_model"
):
    """
    Train REINFORCE agent
    
    Args:
        total_episodes: Number of episodes to train
        learning_rate: Learning rate
        gamma: Discount factor
        hidden_size: Hidden layer size
        save_name: Model save name
    """
    
    print("="*60)
    print("TRAINING REINFORCE AGENT")
    print("="*60)
    print(f"Total episodes: {total_episodes}")
    print(f"Learning rate: {learning_rate}")
    print(f"Gamma: {gamma}")
    print(f"Hidden size: {hidden_size}")
    print("="*60)
    
    # Create environment
    env = gym.make('HealthcareEnv-v0')
    
    # Create agent
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = REINFORCEAgent(obs_dim, action_dim, learning_rate, gamma, hidden_size)
    
    # Training loop
    episode_rewards = deque(maxlen=100)
    best_reward = -float('inf')
    
    for episode in range(total_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        
        # Collect episode
        while not done:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            agent.rewards.append(reward)
            episode_reward += reward
            
            obs = next_obs
            done = terminated or truncated
        
        # Update policy
        loss = agent.update()
        episode_rewards.append(episode_reward)
        
        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save(f"saved_models/reinforce/{save_name}_best.pth")
        
        # Logging
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards)
            print(f"Episode {episode+1}/{total_episodes} | "
                  f"Reward: {episode_reward:.2f} | "
                  f"Avg (100): {avg_reward:.2f} | "
                  f"Loss: {loss:.4f}")
        
        # Save checkpoint
        if (episode + 1) % 100 == 0:
            agent.save(f"saved_models/reinforce/{save_name}_ep{episode+1}.pth")
    
    # Save final model
    agent.save(f"saved_models/reinforce/{save_name}.pth")
    print(f"\n✓ Model saved to saved_models/reinforce/{save_name}.pth")
    
    env.close()
    return agent

def evaluate_model(model_path, num_episodes=10):
    """Evaluate trained REINFORCE model"""
    print("\n" + "="*60)
    print("EVALUATING REINFORCE MODEL")
    print("="*60)
    
    # Create environment
    env = gym.make('HealthcareEnv-v0')
    
    # Create and load agent
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = REINFORCEAgent(obs_dim, action_dim)
    agent.load(model_path)
    
    episode_rewards = []
    episode_detections = []
    episode_severe = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(obs, deterministic=True)
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
    print("HYPERPARAMETER TUNING - REINFORCE")
    print("Running 10 different configurations...")
    print("="*60)
    
    configs = [
        # Config 1: Default
        {"learning_rate": 0.001, "gamma": 0.99, "hidden_size": 128},
        # Config 2: Lower LR
        {"learning_rate": 0.0005, "gamma": 0.99, "hidden_size": 128},
        # Config 3: Higher LR
        {"learning_rate": 0.002, "gamma": 0.99, "hidden_size": 128},
        # Config 4: Different gamma
        {"learning_rate": 0.001, "gamma": 0.95, "hidden_size": 128},
        # Config 5: Higher gamma
        {"learning_rate": 0.001, "gamma": 0.995, "hidden_size": 128},
        # Config 6: Smaller network
        {"learning_rate": 0.001, "gamma": 0.99, "hidden_size": 64},
        # Config 7: Larger network
        {"learning_rate": 0.001, "gamma": 0.99, "hidden_size": 256},
        # Config 8: Very large network
        {"learning_rate": 0.001, "gamma": 0.99, "hidden_size": 512},
        # Config 9: Combined
        {"learning_rate": 0.0005, "gamma": 0.995, "hidden_size": 256},
        # Config 10: Aggressive
        {"learning_rate": 0.002, "gamma": 0.95, "hidden_size": 64},
    ]
    
    results = []
    
    for i, config in enumerate(configs):
        print(f"\n{'='*60}")
        print(f"Configuration {i+1}/10")
        print(f"LR={config['learning_rate']}, Gamma={config['gamma']}, "
              f"Hidden={config['hidden_size']}")
        print(f"{'='*60}")
        
        agent = train_reinforce(
            total_episodes=500,  # Reduced for faster tuning
            learning_rate=config['learning_rate'],
            gamma=config['gamma'],
            hidden_size=config['hidden_size'],
            save_name=f"reinforce_tune_{i+1}"
        )
        
        rewards, detections, severe = evaluate_model(
            f"saved_models/reinforce/reinforce_tune_{i+1}.pth",
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
    with open("saved_models/reinforce/tuning_results.txt", "w") as f:
        f.write("REINFORCE Hyperparameter Tuning Results\n")
        f.write("="*60 + "\n\n")
        for i, result in enumerate(results):
            f.write(f"Configuration {i+1}:\n")
            f.write(f"  {result['config']}\n")
            f.write(f"  Avg Reward: {result['avg_reward']:.2f}\n\n")
        f.write(f"\nBest: Configuration {best_idx + 1}\n")
    
    print("\n✓ Results saved to saved_models/reinforce/tuning_results.txt")
    
    return results, best_config

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "tune":
        hyperparameter_tuning()
    else:
        print("\nTraining REINFORCE with default hyperparameters...")
        print("For hyperparameter tuning, run: python train_reinforce.py tune\n")
        
        agent = train_reinforce(
            total_episodes=1000,
            save_name="reinforce_model"
        )
        
        print("\nEvaluating trained model...")
        evaluate_model("saved_models/reinforce/reinforce_model.pth", num_episodes=10)
        
        print("\n✓ Training complete!")
        print("Note: REINFORCE uses .pth format, not .zip")
        print("Visualize with: python visualize.py (will need to add REINFORCE support)")