import gymnasium as gym
import healthcare_env
import numpy as np
import matplotlib.pyplot as plt

def test_random_agent(num_episodes=5, render=False):
    """Test the environment with random actions"""
    
    env = gym.make('HealthcareEnv-v0', render_mode='human' if render else None)
    
    episode_rewards = []
    episode_lengths = []
    detections_per_episode = []
    severe_cases_per_episode = []
    
    print("="*60)
    print("TESTING RANDOM AGENT")
    print("="*60)
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0
        steps = 0
        
        print(f"\n--- Episode {episode + 1}/{num_episodes} ---")
        
        for step in range(90):  # Max 90 days
            # Random action
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            steps += 1
            
            # Optional: render every 10 steps
            if render and step % 10 == 0:
                env.render()
            
            if terminated or truncated:
                print(f"  Episode ended early at day {info['day']}")
                break
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        detections_per_episode.append(info['total_detections'])
        severe_cases_per_episode.append(info['severe_progressions'])
        
        print(f"  Total Reward: {total_reward:.2f}")
        print(f"  Days: {steps}")
        print(f"  Detections: {info['total_detections']}")
        print(f"  Severe Progressions: {info['severe_progressions']}")
        print(f"  Patients Dropped: {info['patients_dropped']}")
        print(f"  Final Budget: ${info['budget']:.2f}")
    
    env.close()
    
    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS (Random Agent)")
    print("="*60)
    print(f"Average Episode Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average Episode Length: {np.mean(episode_lengths):.2f} days")
    print(f"Average Detections: {np.mean(detections_per_episode):.2f}")
    print(f"Average Severe Cases: {np.mean(severe_cases_per_episode):.2f}")
    print(f"Best Episode Reward: {np.max(episode_rewards):.2f}")
    print(f"Worst Episode Reward: {np.min(episode_rewards):.2f}")
    
    # Plot results
    plot_results(episode_rewards, detections_per_episode, severe_cases_per_episode)
    
    return episode_rewards

def plot_results(rewards, detections, severe_cases):
    """Plot episode statistics"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Rewards
    axes[0].bar(range(1, len(rewards) + 1), rewards, color='blue', alpha=0.7)
    axes[0].axhline(y=0, color='red', linestyle='--', linewidth=1)
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')
    axes[0].set_title('Episode Rewards (Random Agent)')
    axes[0].grid(True, alpha=0.3)
    
    # Detections
    axes[1].bar(range(1, len(detections) + 1), detections, color='green', alpha=0.7)
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Total Detections')
    axes[1].set_title('Early Detections per Episode')
    axes[1].grid(True, alpha=0.3)
    
    # Severe cases
    axes[2].bar(range(1, len(severe_cases) + 1), severe_cases, color='red', alpha=0.7)
    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel('Severe Progressions')
    axes[2].set_title('Severe Cases per Episode')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('random_agent_results.png', dpi=150)
    print("\n✓ Results plot saved as 'random_agent_results.png'")
    plt.show()

def test_action_space():
    """Test all actions to ensure they work"""
    env = gym.make('HealthcareEnv-v0')
    obs, info = env.reset()
    
    print("\n" + "="*60)
    print("TESTING ALL ACTIONS")
    print("="*60)
    
    action_names = [
        "Screen high-risk with nurse",
        "Screen medium-risk with nurse",
        "Screen low-risk with nurse",
        "Deploy mobile unit (high-risk region)",
        "Deploy mobile unit (medium-risk region)",
        "Doppler on high-risk",
        "Doppler on medium-risk",
        "Refer to specialist",
        "Launch awareness campaign",
        "Train community health worker",
        "Request additional budget",
        "Wait/Do nothing"
    ]
    
    for action in range(12):
        obs, info = env.reset()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Action {action}: {action_names[action]:40s} → Reward: {reward:+.2f}")
    
    env.close()
    print("\n✓ All actions executed successfully!")

if __name__ == "__main__":
    # Test all actions first
    test_action_space()
    
    # Run random agent for 5 episodes
    print("\n" + "="*60)
    input("Press Enter to run random agent test (5 episodes)...")
    test_random_agent(num_episodes=5, render=False)
    
    print("\n✓ Testing complete! Environment is ready for training.")