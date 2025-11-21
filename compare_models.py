"""
compare_models.py - Compare performance of all trained RL models
Evaluates DQN, PPO, A2C, and REINFORCE side-by-side
"""

import gymnasium as gym
import healthcare_env
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN, PPO, A2C
import torch
import torch.nn as nn

# Import REINFORCE components
class PolicyNetwork(nn.Module):
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
    def __init__(self, obs_dim, action_dim, hidden_size=128):
        self.policy = PolicyNetwork(obs_dim, action_dim, hidden_size)
        
    def select_action(self, state, deterministic=True):
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.policy(state)
        if deterministic:
            action = torch.argmax(probs, dim=1).item()
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample().item()
        return action
    
    def load(self, filepath):
        checkpoint = torch.load(filepath)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])

def evaluate_model(model, model_name, num_episodes=20):
    """Evaluate a trained model"""
    env = gym.make('HealthcareEnv-v0')
    
    episode_rewards = []
    episode_detections = []
    episode_severe = []
    episode_dropped = []
    episode_lengths = []
    
    print(f"\nEvaluating {model_name}...")
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0
        done = False
        steps = 0
        
        while not done:
            if isinstance(model, REINFORCEAgent):
                action = model.select_action(obs, deterministic=True)
            else:
                action, _ = model.predict(obs, deterministic=True)
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated
        
        episode_rewards.append(total_reward)
        episode_detections.append(info['total_detections'])
        episode_severe.append(info['severe_progressions'])
        episode_dropped.append(info['patients_dropped'])
        episode_lengths.append(steps)
        
        print(f"  Episode {episode+1}/{num_episodes}: "
              f"Reward={total_reward:.1f}, "
              f"Detections={info['total_detections']}, "
              f"Severe={info['severe_progressions']}")
    
    env.close()
    
    return {
        'model_name': model_name,
        'rewards': episode_rewards,
        'detections': episode_detections,
        'severe': episode_severe,
        'dropped': episode_dropped,
        'lengths': episode_lengths,
        'avg_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'avg_detections': np.mean(episode_detections),
        'avg_severe': np.mean(episode_severe),
        'avg_dropped': np.mean(episode_dropped),
        'avg_length': np.mean(episode_lengths)
    }

def load_models():
    """Load all trained models"""
    models = {}
    
    # Load DQN
    try:
        models['DQN'] = DQN.load("saved_models/dqn/dqn_model.zip")
        print("✓ Loaded DQN model")
    except Exception as e:
        print(f"✗ Could not load DQN: {e}")
    
    # Load PPO
    try:
        models['PPO'] = PPO.load("saved_models/ppo/ppo_model.zip")
        print("✓ Loaded PPO model")
    except Exception as e:
        print(f"✗ Could not load PPO: {e}")
    
    # Load A2C
    try:
        models['A2C'] = A2C.load("saved_models/a2c/a2c_model.zip")
        print("✓ Loaded A2C model")
    except Exception as e:
        print(f"✗ Could not load A2C: {e}")
    
    # Load REINFORCE
    try:
        env = gym.make('HealthcareEnv-v0')
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        reinforce = REINFORCEAgent(obs_dim, action_dim)
        reinforce.load("saved_models/reinforce/reinforce_model.pth")
        models['REINFORCE'] = reinforce
        print("✓ Loaded REINFORCE model")
        env.close()
    except Exception as e:
        print(f"✗ Could not load REINFORCE: {e}")
    
    return models

def compare_all_models(num_episodes=20):
    """Compare all trained models"""
    print("="*80)
    print("COMPARING ALL TRAINED MODELS")
    print("="*80)
    
    # Load models
    models = load_models()
    
    if not models:
        print("\n✗ No models found! Please train models first.")
        return
    
    # Evaluate each model
    results = []
    for model_name, model in models.items():
        result = evaluate_model(model, model_name, num_episodes)
        results.append(result)
    
    # Print comparison table
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Model':<15} {'Avg Reward':<15} {'Detections':<15} {'Severe':<15} {'Dropped':<15}")
    print("-"*80)
    
    for result in results:
        print(f"{result['model_name']:<15} "
              f"{result['avg_reward']:<15.2f} "
              f"{result['avg_detections']:<15.2f} "
              f"{result['avg_severe']:<15.2f} "
              f"{result['avg_dropped']:<15.2f}")
    
    print("="*80)
    
    # Find best model
    best_model = max(results, key=lambda x: x['avg_reward'])
    print(f"\n🏆 BEST PERFORMING MODEL: {best_model['model_name']}")
    print(f"   Average Reward: {best_model['avg_reward']:.2f}")
    print(f"   Average Detections: {best_model['avg_detections']:.2f}")
    print(f"   Average Severe Cases: {best_model['avg_severe']:.2f}")
    
    # Plot comparison
    plot_comparison(results)
    
    # Save results
    save_results(results)
    
    return results, best_model

def plot_comparison(results):
    """Create comparison plots"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Model Comparison - Healthcare Resource Allocation', fontsize=16, fontweight='bold')
    
    models = [r['model_name'] for r in results]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # 1. Average Rewards
    ax1 = axes[0, 0]
    avg_rewards = [r['avg_reward'] for r in results]
    std_rewards = [r['std_reward'] for r in results]
    bars = ax1.bar(models, avg_rewards, yerr=std_rewards, capsize=5, color=colors, alpha=0.7)
    ax1.set_ylabel('Average Reward', fontsize=12, fontweight='bold')
    ax1.set_title('Average Episode Reward (Higher is Better)', fontsize=12)
    ax1.grid(axis='y', alpha=0.3)
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=1)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Detections vs Severe Cases
    ax2 = axes[0, 1]
    x = np.arange(len(models))
    width = 0.35
    detections = [r['avg_detections'] for r in results]
    severe = [r['avg_severe'] for r in results]
    
    bars1 = ax2.bar(x - width/2, detections, width, label='Early Detections', color='green', alpha=0.7)
    bars2 = ax2.bar(x + width/2, severe, width, label='Severe Cases', color='red', alpha=0.7)
    
    ax2.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax2.set_title('Early Detections vs Severe Progressions', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    # 3. Reward Distribution (Box Plot)
    ax3 = axes[1, 0]
    reward_data = [r['rewards'] for r in results]
    bp = ax3.boxplot(reward_data, labels=models, patch_artist=True)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax3.set_ylabel('Reward Distribution', fontsize=12, fontweight='bold')
    ax3.set_title('Reward Stability (Box Plot)', fontsize=12)
    ax3.grid(axis='y', alpha=0.3)
    ax3.axhline(y=0, color='red', linestyle='--', linewidth=1)
    
    # 4. Performance Metrics Radar
    ax4 = axes[1, 1]
    
    # Normalize metrics for comparison
    max_reward = max([r['avg_reward'] for r in results]) if max([r['avg_reward'] for r in results]) > 0 else 1
    max_detections = max([r['avg_detections'] for r in results]) if max([r['avg_detections'] for r in results]) > 0 else 1
    
    categories = ['Reward\n(normalized)', 'Detections\n(normalized)', 'Low Severe\nCases', 'Low Dropouts']
    
    for i, result in enumerate(results):
        # Normalize and invert where needed (lower is better for severe/dropped)
        values = [
            max(0, result['avg_reward'] / max_reward),  # Normalized reward
            result['avg_detections'] / max_detections,  # Normalized detections
            1 - (result['avg_severe'] / 10),  # Inverted severe (lower is better)
            1 - (result['avg_dropped'] / 10)  # Inverted dropped (lower is better)
        ]
        
        x_pos = np.arange(len(categories))
        ax4.plot(x_pos, values, 'o-', label=result['model_name'], color=colors[i], linewidth=2, markersize=8)
    
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(categories, fontsize=9)
    ax4.set_ylim(0, 1.1)
    ax4.set_ylabel('Score (0-1)', fontsize=12, fontweight='bold')
    ax4.set_title('Overall Performance Profile', fontsize=12)
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ Comparison plot saved as 'model_comparison.png'")
    plt.show()

def save_results(results):
    """Save comparison results to file"""
    with open('model_comparison_results.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("MODEL COMPARISON RESULTS\n")
        f.write("Healthcare Resource Allocation - AI-Veins Project\n")
        f.write("="*80 + "\n\n")
        
        for result in results:
            f.write(f"\n{result['model_name']} Model:\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Average Reward:      {result['avg_reward']:.2f} ± {result['std_reward']:.2f}\n")
            f.write(f"  Average Detections:  {result['avg_detections']:.2f}\n")
            f.write(f"  Average Severe:      {result['avg_severe']:.2f}\n")
            f.write(f"  Average Dropped:     {result['avg_dropped']:.2f}\n")
            f.write(f"  Average Episode Length: {result['avg_length']:.1f} days\n")
        
        # Best model
        best_model = max(results, key=lambda x: x['avg_reward'])
        f.write("\n" + "="*80 + "\n")
        f.write(f"BEST PERFORMING MODEL: {best_model['model_name']}\n")
        f.write(f"  Average Reward: {best_model['avg_reward']:.2f}\n")
        f.write("="*80 + "\n")
    
    print("✓ Results saved to 'model_comparison_results.txt'")

if __name__ == "__main__":
    print("\nStarting model comparison...")
    print("This will evaluate all trained models on 20 episodes each.\n")
    
    results, best_model = compare_all_models(num_episodes=20)
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  - model_comparison.png (visualization)")
    print("  - model_comparison_results.txt (detailed results)")
    print(f"\nBest model for visualization: saved_models/{best_model['model_name'].lower()}/{best_model['model_name'].lower()}_model.zip")
    print(f"Run: python visualize.py saved_models/{best_model['model_name'].lower()}/{best_model['model_name'].lower()}_model.zip")