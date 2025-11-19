"""
healthcare_env.py - Custom Gymnasium Environment
AI-Veins: Healthcare Resource Allocation for Varicose Vein Screening

This environment simulates a community health system where an RL agent
must optimally allocate limited resources to screen patients for varicose veins.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from enum import IntEnum

# Register the environment
from gymnasium.envs.registration import register

register(
    id='HealthcareEnv-v0',
    entry_point='healthcare_env:HealthcareEnvironment',
)


class RiskLevel(IntEnum):
    """Patient risk levels for varicose veins"""
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    SEVERE = 3


class HealthcareEnvironment(gym.Env):
    """
    Healthcare Resource Allocation Environment
    
    Observation Space:
        - Day (0-89): Current day of simulation
        - Budget remaining (0-10000): Dollars left
        - Nurses available (0-5): Staff count
        - Mobile units available (0-2): Equipment count
        - Doppler machines available (0-3): Equipment count
        - Patient queue (next 5 patients): Each has [risk_level, symptoms, wait_days, distance]
        - Regional stats (5 regions): Each has [population_pct, avg_risk]
        
    Action Space (Discrete 12):
        0: Screen high-risk patient with nurse
        1: Screen medium-risk patient with nurse
        2: Screen low-risk patient with nurse
        3: Deploy mobile unit to high-risk region
        4: Deploy mobile unit to medium-risk region
        5: Use Doppler ultrasound on high-risk patient
        6: Use Doppler ultrasound on medium-risk patient
        7: Refer patient to specialist
        8: Launch awareness campaign
        9: Train community health worker
        10: Request additional budget
        11: Wait/Do nothing (conserve resources)
    
    Reward Structure:
        +100: Early detection (risk HIGH, no progression)
        +200: Early detection prevented severe case
        +50: Medium-risk detection
        +10: Low-risk screening (disease ruled out)
        -200: Patient progresses to severe stage (missed opportunity)
        -50: Patient drops out due to long wait
        -30: Inefficient resource use (expensive tool on low-risk)
        -10: Daily penalty for untreated high-risk patients
        -5: Budget overspend penalty
    
    Terminal Conditions:
        - Day 90 reached
        - Budget depleted (< 0)
        - More than 10 patients progress to severe stage
    """
    
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(self, render_mode=None):
        super().__init__()
        
        self.render_mode = render_mode
        
        # Environment parameters
        self.max_days = 90
        self.initial_budget = 10000
        self.max_nurses = 5
        self.max_mobile_units = 2
        self.max_doppler = 3
        self.max_queue_size = 20
        self.num_regions = 5
        
        # Costs
        self.cost_nurse_screening = 50
        self.cost_mobile_unit = 100
        self.cost_doppler = 150
        self.cost_specialist_referral = 200
        self.cost_awareness = 300
        self.cost_training = 250
        self.cost_budget_request = 0  # Free but has cooldown
        
        # Define observation space
        # [day, budget, nurses, mobile_units, doppler, 
        #  5 patients x 4 features (risk, symptoms, wait, distance),
        #  5 regions x 2 features (population_pct, avg_risk)]
        obs_size = 5 + (5 * 4) + (5 * 2)  # 35 total
        self.observation_space = spaces.Box(
            low=0, 
            high=1, 
            shape=(obs_size,), 
            dtype=np.float32
        )
        
        # Define action space (12 discrete actions)
        self.action_space = spaces.Discrete(12)
        
        # State variables
        self.current_day = 0
        self.budget = 0
        self.nurses_available = 0
        self.mobile_units_available = 0
        self.doppler_available = 0
        self.patient_queue = []
        self.regional_stats = []
        self.total_detections = 0
        self.severe_progressions = 0
        self.patients_dropped = 0
        self.episode_reward = 0
        
    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        # Reset state
        self.current_day = 0
        self.budget = self.initial_budget
        self.nurses_available = self.max_nurses
        self.mobile_units_available = self.max_mobile_units
        self.doppler_available = self.max_doppler
        self.total_detections = 0
        self.severe_progressions = 0
        self.patients_dropped = 0
        self.episode_reward = 0
        
        # Generate initial patient queue
        self.patient_queue = self._generate_patients(15)
        
        # Initialize regional stats
        self.regional_stats = self._generate_regional_stats()
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action):
        """Execute one time step within the environment"""
        reward = 0
        
        # Process action
        action_reward, action_cost = self._process_action(action)
        reward += action_reward
        
        # Deduct cost
        self.budget -= action_cost
        if self.budget < 0:
            reward -= 5  # Penalty for overspending
        
        # Simulate patient queue dynamics
        queue_reward = self._update_patient_queue()
        reward += queue_reward
        
        # Daily penalties for high-risk patients waiting
        high_risk_waiting = sum(1 for p in self.patient_queue if p['risk'] == RiskLevel.HIGH)
        reward -= high_risk_waiting * 10
        
        # Recover some resources (simulate shift changes)
        self._recover_resources()
        
        # Add new patients randomly
        if len(self.patient_queue) < self.max_queue_size and np.random.random() > 0.7:
            new_patients = self._generate_patients(np.random.randint(1, 4))
            self.patient_queue.extend(new_patients)
        
        # Advance day
        self.current_day += 1
        
        # Check terminal conditions
        terminated = False
        if self.current_day >= self.max_days:
            terminated = True
            reward += 100  # Bonus for completing episode
        if self.budget < -1000:
            terminated = True
            reward -= 500  # Major penalty for bankruptcy
        if self.severe_progressions > 10:
            terminated = True
            reward -= 1000  # Major penalty for too many severe cases
        
        truncated = False
        
        self.episode_reward += reward
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _process_action(self, action):
        """Process the agent's action and return (reward, cost)"""
        reward = 0
        cost = 0
        
        if action == 0:  # Screen high-risk with nurse
            if self.nurses_available > 0 and len(self.patient_queue) > 0:
                patient = self._get_patient_by_risk(RiskLevel.HIGH)
                if patient:
                    self.patient_queue.remove(patient)
                    self.nurses_available -= 1
                    cost = self.cost_nurse_screening
                    # High chance of early detection
                    if np.random.random() > 0.3:
                        reward = 100
                        self.total_detections += 1
                    else:
                        reward = 50
        
        elif action == 1:  # Screen medium-risk with nurse
            if self.nurses_available > 0 and len(self.patient_queue) > 0:
                patient = self._get_patient_by_risk(RiskLevel.MEDIUM)
                if patient:
                    self.patient_queue.remove(patient)
                    self.nurses_available -= 1
                    cost = self.cost_nurse_screening
                    reward = 50 if np.random.random() > 0.5 else 10
        
        elif action == 2:  # Screen low-risk with nurse
            if self.nurses_available > 0 and len(self.patient_queue) > 0:
                patient = self._get_patient_by_risk(RiskLevel.LOW)
                if patient:
                    self.patient_queue.remove(patient)
                    self.nurses_available -= 1
                    cost = self.cost_nurse_screening
                    reward = 10  # Ruled out disease
        
        elif action == 3:  # Deploy mobile unit to high-risk region
            if self.mobile_units_available > 0:
                self.mobile_units_available -= 1
                cost = self.cost_mobile_unit
                reward = 30  # Access improvement bonus
        
        elif action == 4:  # Deploy mobile unit to medium-risk region
            if self.mobile_units_available > 0:
                self.mobile_units_available -= 1
                cost = self.cost_mobile_unit
                reward = 15
        
        elif action == 5:  # Doppler on high-risk
            if self.doppler_available > 0 and len(self.patient_queue) > 0:
                patient = self._get_patient_by_risk(RiskLevel.HIGH)
                if patient:
                    self.patient_queue.remove(patient)
                    self.doppler_available -= 1
                    cost = self.cost_doppler
                    # Very high accuracy with Doppler
                    if np.random.random() > 0.15:
                        reward = 200  # Excellent early detection
                        self.total_detections += 1
                    else:
                        reward = 100
        
        elif action == 6:  # Doppler on medium-risk
            if self.doppler_available > 0 and len(self.patient_queue) > 0:
                patient = self._get_patient_by_risk(RiskLevel.MEDIUM)
                if patient:
                    self.patient_queue.remove(patient)
                    self.doppler_available -= 1
                    cost = self.cost_doppler
                    reward = 50  # Good but may be overkill
        
        elif action == 7:  # Refer to specialist
            if len(self.patient_queue) > 0:
                # Prefer severe patients
                patient = self._get_patient_by_risk(RiskLevel.SEVERE)
                if patient:
                    self.patient_queue.remove(patient)
                    cost = self.cost_specialist_referral
                    reward = 80
        
        elif action == 8:  # Awareness campaign
            cost = self.cost_awareness
            reward = 20  # Long-term benefit
        
        elif action == 9:  # Train CHW
            cost = self.cost_training
            reward = 25  # Capacity building
        
        elif action == 10:  # Request budget
            if self.current_day % 15 == 0:  # Only every 15 days
                self.budget += 2000
                reward = 0
        
        elif action == 11:  # Wait/do nothing
            reward = -5  # Small penalty for inaction
        
        return reward, cost
    
    def _update_patient_queue(self):
        """Update patient queue: increase wait times, check for progressions"""
        reward = 0
        to_remove = []
        
        for patient in self.patient_queue:
            patient['wait_days'] += 1
            
            # Patient drops out if waiting too long
            if patient['wait_days'] > 30:
                to_remove.append(patient)
                reward -= 50
                self.patients_dropped += 1
            
            # Disease progression for high-risk patients
            elif patient['risk'] == RiskLevel.HIGH and patient['wait_days'] > 20:
                if np.random.random() > 0.7:
                    patient['risk'] = RiskLevel.SEVERE
                    reward -= 200
                    self.severe_progressions += 1
            
            # Medium-risk can become high-risk
            elif patient['risk'] == RiskLevel.MEDIUM and patient['wait_days'] > 25:
                if np.random.random() > 0.8:
                    patient['risk'] = RiskLevel.HIGH
                    reward -= 30
        
        # Remove dropped patients
        for patient in to_remove:
            self.patient_queue.remove(patient)
        
        return reward
    
    def _recover_resources(self):
        """Simulate resource recovery (shifts, equipment return)"""
        # Nurses return from screenings
        if np.random.random() > 0.5:
            self.nurses_available = min(self.max_nurses, self.nurses_available + 1)
        
        # Mobile units return
        if np.random.random() > 0.7:
            self.mobile_units_available = min(self.max_mobile_units, self.mobile_units_available + 1)
        
        # Doppler becomes available
        if np.random.random() > 0.6:
            self.doppler_available = min(self.max_doppler, self.doppler_available + 1)
    
    def _generate_patients(self, count):
        """Generate random patients"""
        patients = []
        for _ in range(count):
            risk_probs = [0.4, 0.35, 0.20, 0.05]  # LOW, MED, HIGH, SEVERE
            risk = np.random.choice([0, 1, 2, 3], p=risk_probs)
            
            patients.append({
                'risk': risk,
                'symptoms': np.random.randint(3, 11),  # 3-10 scale
                'wait_days': np.random.randint(0, 10),
                'distance': np.random.randint(5, 51),  # km
            })
        return patients
    
    def _generate_regional_stats(self):
        """Generate regional statistics"""
        stats = []
        for _ in range(self.num_regions):
            stats.append({
                'population_pct': np.random.uniform(0.15, 0.25),
                'avg_risk': np.random.uniform(0.3, 0.7)
            })
        return stats
    
    def _get_patient_by_risk(self, target_risk):
        """Get first patient matching risk level"""
        for patient in self.patient_queue:
            if patient['risk'] == target_risk:
                return patient
        # If no exact match, return first patient
        return self.patient_queue[0] if self.patient_queue else None
    
    def _get_observation(self):
        """Get current observation vector"""
        obs = np.zeros(35, dtype=np.float32)
        
        # Normalize state variables
        obs[0] = self.current_day / self.max_days
        obs[1] = max(0, self.budget / self.initial_budget)
        obs[2] = self.nurses_available / self.max_nurses
        obs[3] = self.mobile_units_available / self.max_mobile_units
        obs[4] = self.doppler_available / self.max_doppler
        
        # Next 5 patients in queue (or zeros if queue smaller)
        for i in range(5):
            if i < len(self.patient_queue):
                p = self.patient_queue[i]
                obs[5 + i*4] = p['risk'] / 3  # Normalize risk
                obs[6 + i*4] = p['symptoms'] / 10
                obs[7 + i*4] = min(p['wait_days'] / 30, 1.0)
                obs[8 + i*4] = p['distance'] / 50
        
        # Regional stats
        for i in range(5):
            if i < len(self.regional_stats):
                obs[25 + i*2] = self.regional_stats[i]['population_pct']
                obs[26 + i*2] = self.regional_stats[i]['avg_risk']
        
        return obs
    
    def _get_info(self):
        """Get additional info dict"""
        return {
            'day': self.current_day,
            'budget': self.budget,
            'queue_size': len(self.patient_queue),
            'total_detections': self.total_detections,
            'severe_progressions': self.severe_progressions,
            'patients_dropped': self.patients_dropped,
            'episode_reward': self.episode_reward
        }
    
    def render(self):
        """Render environment (optional)"""
        if self.render_mode == 'human':
            print(f"\n=== Day {self.current_day} ===")
            print(f"Budget: ${self.budget}")
            print(f"Resources: Nurses={self.nurses_available}, Mobile={self.mobile_units_available}, Doppler={self.doppler_available}")
            print(f"Queue: {len(self.patient_queue)} patients")
            print(f"Stats: Detections={self.total_detections}, Severe={self.severe_progressions}, Dropped={self.patients_dropped}")
    
    def close(self):
        """Cleanup"""
        pass


if __name__ == "__main__":
    # Test the environment
    env = gym.make('HealthcareEnv-v0', render_mode='human')
    
    print("Testing Healthcare Environment...")
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    
    # Run a few random steps
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        print(f"Action: {action}, Reward: {reward:.2f}")
        
        if terminated or truncated:
            break
    
    env.close()
    print("\nEnvironment test complete!")