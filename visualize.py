"""
visualize.py - Pygame visualization for healthcare environment
Watch the agent perform actions in real-time with visual feedback
"""

import pygame
import gymnasium as gym
import healthcare_env
import numpy as np
from enum import IntEnum

# Initialize Pygame
pygame.init()

# Constants
WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 800
FPS = 3  # 3 actions per second (slow enough to watch)

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
LIGHT_GRAY = (240, 240, 240)
DARK_GRAY = (100, 100, 100)
RED = (220, 53, 69)
GREEN = (40, 167, 69)
BLUE = (0, 123, 255)
YELLOW = (255, 193, 7)
ORANGE = (253, 126, 20)
PURPLE = (111, 66, 193)
LIGHT_RED = (255, 200, 200)
LIGHT_GREEN = (200, 255, 200)
LIGHT_BLUE = (200, 230, 255)

class RiskLevel(IntEnum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    SEVERE = 3

class HealthcareVisualizer:
    def __init__(self, use_trained_model=False, model_path=None):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("AI-Veins: Healthcare Resource Allocation - Live Agent")
        self.clock = pygame.time.Clock()
        
        # Fonts
        self.font_title = pygame.font.Font(None, 36)
        self.font_large = pygame.font.Font(None, 28)
        self.font_medium = pygame.font.Font(None, 22)
        self.font_small = pygame.font.Font(None, 18)
        
        # Environment
        self.env = gym.make('HealthcareEnv-v0')
        self.obs, self.info = self.env.reset()
        
        # Model (if using trained agent)
        self.use_trained_model = use_trained_model
        self.model = None
        if use_trained_model and model_path:
            try:
                from stable_baselines3 import PPO, DQN, A2C
                # Try to load model (will implement after training)
                if 'ppo' in model_path.lower():
                    self.model = PPO.load(model_path)
                elif 'dqn' in model_path.lower():
                    self.model = DQN.load(model_path)
                elif 'a2c' in model_path.lower():
                    self.model = A2C.load(model_path)
                print(f"✓ Loaded trained model: {model_path}")
            except Exception as e:
                print(f"Could not load model: {e}")
                print("Using random agent instead")
                self.use_trained_model = False
        
        # Action names
        self.action_names = [
            "Screen HIGH-RISK with Nurse",
            "Screen MEDIUM-RISK with Nurse",
            "Screen LOW-RISK with Nurse",
            "Deploy Mobile Unit (High-Risk Region)",
            "Deploy Mobile Unit (Medium-Risk Region)",
            "Use DOPPLER on High-Risk",
            "Use DOPPLER on Medium-Risk",
            "Refer to Specialist",
            "Launch Awareness Campaign",
            "Train Community Health Worker",
            "Request Additional Budget",
            "WAIT (Conserve Resources)"
        ]
        
        self.last_action = 11
        self.last_reward = 0
        self.action_history = []
        self.total_reward = 0
        
    def get_risk_color(self, risk_level):
        if risk_level == RiskLevel.LOW:
            return GREEN
        elif risk_level == RiskLevel.MEDIUM:
            return YELLOW
        elif risk_level == RiskLevel.HIGH:
            return ORANGE
        else:
            return RED
    
    def draw_header(self):
        """Draw title and episode info"""
        pygame.draw.rect(self.screen, BLUE, (0, 0, WINDOW_WIDTH, 80))
        
        title = self.font_title.render("AI-Veins: Healthcare Resource Allocation", True, WHITE)
        agent_type = "TRAINED AGENT" if self.use_trained_model else "RANDOM AGENT"
        subtitle = self.font_medium.render(f"Watching {agent_type} Perform", True, LIGHT_BLUE)
        
        self.screen.blit(title, (20, 15))
        self.screen.blit(subtitle, (20, 50))
        
        # Day counter
        day_text = self.font_large.render(f"Day {self.info['day']}/90", True, WHITE)
        self.screen.blit(day_text, (WINDOW_WIDTH - 150, 25))
    
    def draw_patient_queue(self):
        """Draw patient queue with risk levels"""
        panel_x, panel_y = 20, 100
        panel_width, panel_height = 400, 500
        
        pygame.draw.rect(self.screen, WHITE, (panel_x, panel_y, panel_width, panel_height))
        pygame.draw.rect(self.screen, DARK_GRAY, (panel_x, panel_y, panel_width, panel_height), 2)
        
        title = self.font_large.render(f"Patient Queue ({self.info['queue_size']})", True, BLACK)
        self.screen.blit(title, (panel_x + 10, panel_y + 10))
        
        # Get patient data from observation
        y_offset = panel_y + 50
        
        for i in range(min(5, self.info['queue_size'])):
            # Extract patient data from observation (indices 5-24 for 5 patients)
            idx = 5 + (i * 4)
            if idx + 3 < len(self.obs):
                risk_norm = self.obs[idx]
                symptoms_norm = self.obs[idx + 1]
                wait_norm = self.obs[idx + 2]
                distance_norm = self.obs[idx + 3]
                
                # Denormalize
                risk_level = int(risk_norm * 3)
                symptoms = int(symptoms_norm * 10)
                wait_days = int(wait_norm * 30)
                distance = int(distance_norm * 50)
                
                row_y = y_offset + (i * 85)
                
                # Patient card
                card_color = LIGHT_GRAY if i % 2 == 0 else WHITE
                pygame.draw.rect(self.screen, card_color, (panel_x + 10, row_y, panel_width - 20, 75))
                pygame.draw.rect(self.screen, GRAY, (panel_x + 10, row_y, panel_width - 20, 75), 1)
                
                # Patient icon
                risk_color = self.get_risk_color(risk_level)
                pygame.draw.circle(self.screen, risk_color, (panel_x + 40, row_y + 35), 20)
                
                # Risk label
                risk_names = ["LOW", "MED", "HIGH", "SEV"]
                risk_text = self.font_small.render(risk_names[min(risk_level, 3)], True, WHITE)
                text_rect = risk_text.get_rect(center=(panel_x + 40, row_y + 35))
                self.screen.blit(risk_text, text_rect)
                
                # Patient details
                details_x = panel_x + 75
                patient_name = self.font_medium.render(f"Patient #{i+1}", True, BLACK)
                self.screen.blit(patient_name, (details_x, row_y + 5))
                
                details = [
                    f"Symptoms: {symptoms}/10",
                    f"Waiting: {wait_days} days",
                    f"Distance: {distance} km"
                ]
                
                for j, detail in enumerate(details):
                    detail_text = self.font_small.render(detail, True, DARK_GRAY)
                    self.screen.blit(detail_text, (details_x, row_y + 30 + j * 15))
    
    def draw_resources(self):
        """Draw available resources"""
        panel_x, panel_y = 440, 100
        panel_width, panel_height = 480, 240
        
        pygame.draw.rect(self.screen, LIGHT_BLUE, (panel_x, panel_y, panel_width, panel_height))
        pygame.draw.rect(self.screen, DARK_GRAY, (panel_x, panel_y, panel_width, panel_height), 2)
        
        title = self.font_large.render("Resources Available", True, BLACK)
        self.screen.blit(title, (panel_x + 10, panel_y + 10))
        
        # Budget bar
        budget_norm = self.obs[1]
        budget_actual = budget_norm * 10000
        
        bar_width = 440
        bar_height = 30
        bar_x = panel_x + 20
        bar_y = panel_y + 50
        
        # Background
        pygame.draw.rect(self.screen, WHITE, (bar_x, bar_y, bar_width, bar_height))
        # Fill
        fill_width = int(bar_width * budget_norm)
        color = GREEN if budget_norm > 0.5 else ORANGE if budget_norm > 0.25 else RED
        pygame.draw.rect(self.screen, color, (bar_x, bar_y, fill_width, bar_height))
        # Border
        pygame.draw.rect(self.screen, DARK_GRAY, (bar_x, bar_y, bar_width, bar_height), 2)
        
        budget_text = self.font_medium.render(f"Budget: ${budget_actual:.0f} / $10,000", True, BLACK)
        self.screen.blit(budget_text, (bar_x + 10, bar_y + 5))
        
        # Resources
        y_offset = bar_y + 50
        
        nurses = int(self.obs[2] * 5)
        mobile = int(self.obs[3] * 2)
        doppler = int(self.obs[4] * 3)
        
        resources = [
            ("Nurses Available", nurses, 5, GREEN),
            ("Mobile Units", mobile, 2, PURPLE),
            ("Doppler Machines", doppler, 3, BLUE)
        ]
        
        for i, (name, available, total, color) in enumerate(resources):
            row_y = y_offset + (i * 50)
            
            label = self.font_medium.render(f"{name}:", True, BLACK)
            self.screen.blit(label, (panel_x + 20, row_y))
            
            # Draw circles for each resource
            for j in range(total):
                circle_color = color if j < available else GRAY
                pygame.draw.circle(self.screen, circle_color, 
                                 (panel_x + 250 + j * 50, row_y + 12), 15)
            
            count_text = self.font_large.render(f"{available}/{total}", True, BLACK)
            self.screen.blit(count_text, (panel_x + 400, row_y - 5))
    
    def draw_statistics(self):
        """Draw performance statistics"""
        panel_x, panel_y = 440, 360
        panel_width, panel_height = 480, 240
        
        pygame.draw.rect(self.screen, LIGHT_GREEN, (panel_x, panel_y, panel_width, panel_height))
        pygame.draw.rect(self.screen, DARK_GRAY, (panel_x, panel_y, panel_width, panel_height), 2)
        
        title = self.font_large.render("Performance Statistics", True, BLACK)
        self.screen.blit(title, (panel_x + 10, panel_y + 10))
        
        stats = [
            ("Total Detections", self.info['total_detections'], GREEN),
            ("Severe Progressions", self.info['severe_progressions'], RED),
            ("Patients Dropped", self.info['patients_dropped'], ORANGE),
            ("Episode Reward", int(self.total_reward), BLUE)
        ]
        
        for i, (label, value, color) in enumerate(stats):
            row_y = panel_y + 60 + (i * 45)
            
            label_text = self.font_medium.render(f"{label}:", True, BLACK)
            self.screen.blit(label_text, (panel_x + 20, row_y))
            
            value_text = self.font_large.render(str(value), True, color)
            self.screen.blit(value_text, (panel_x + 380, row_y - 3))
    
    def draw_action_panel(self):
        """Draw current action and history"""
        panel_x, panel_y = 940, 100
        panel_width, panel_height = 440, 500
        
        pygame.draw.rect(self.screen, LIGHT_RED, (panel_x, panel_y, panel_width, panel_height))
        pygame.draw.rect(self.screen, DARK_GRAY, (panel_x, panel_y, panel_width, panel_height), 2)
        
        title = self.font_large.render("Agent Actions", True, BLACK)
        self.screen.blit(title, (panel_x + 10, panel_y + 10))
        
        # Current action (large)
        current_label = self.font_medium.render("CURRENT ACTION:", True, BLACK)
        self.screen.blit(current_label, (panel_x + 20, panel_y + 50))
        
        action_text = self.action_names[self.last_action]
        # Split long text into two lines
        if len(action_text) > 30:
            words = action_text.split()
            line1 = " ".join(words[:3])
            line2 = " ".join(words[3:])
            action1 = self.font_large.render(line1, True, BLUE)
            action2 = self.font_large.render(line2, True, BLUE)
            self.screen.blit(action1, (panel_x + 20, panel_y + 85))
            self.screen.blit(action2, (panel_x + 20, panel_y + 115))
        else:
            action = self.font_large.render(action_text, True, BLUE)
            self.screen.blit(action, (panel_x + 20, panel_y + 85))
        
        # Reward
        reward_label = self.font_medium.render("REWARD:", True, BLACK)
        self.screen.blit(reward_label, (panel_x + 20, panel_y + 160))
        
        reward_color = GREEN if self.last_reward > 0 else RED if self.last_reward < 0 else GRAY
        reward_text = self.font_title.render(f"{self.last_reward:+.0f}", True, reward_color)
        self.screen.blit(reward_text, (panel_x + 20, panel_y + 190))
        
        # Total reward
        total_label = self.font_medium.render("TOTAL REWARD:", True, BLACK)
        self.screen.blit(total_label, (panel_x + 200, panel_y + 160))
        
        total_color = GREEN if self.total_reward > 0 else RED
        total_text = self.font_title.render(f"{self.total_reward:.0f}", True, total_color)
        self.screen.blit(total_text, (panel_x + 200, panel_y + 190))
        
        # Recent action history
        history_y = panel_y + 260
        history_label = self.font_medium.render("Recent Actions:", True, BLACK)
        self.screen.blit(history_label, (panel_x + 20, history_y))
        
        for i, action_idx in enumerate(self.action_history[-8:]):
            row_y = history_y + 35 + (i * 25)
            action_short = self.action_names[action_idx][:35]
            history_text = self.font_small.render(f"• {action_short}", True, DARK_GRAY)
            self.screen.blit(history_text, (panel_x + 30, row_y))
    
    def draw_instructions(self):
        """Draw control instructions"""
        panel_x, panel_y = 20, 620
        panel_width, panel_height = 900, 160
        
        pygame.draw.rect(self.screen, LIGHT_GRAY, (panel_x, panel_y, panel_width, panel_height))
        pygame.draw.rect(self.screen, DARK_GRAY, (panel_x, panel_y, panel_width, panel_height), 2)
        
        title = self.font_large.render("Controls", True, BLACK)
        self.screen.blit(title, (panel_x + 10, panel_y + 10))
        
        instructions = [
            "SPACE - Pause/Resume",
            "R - Reset Episode",
            "ESC - Quit",
            f"Speed: {FPS} actions/second"
        ]
        
        for i, instruction in enumerate(instructions):
            text = self.font_medium.render(instruction, True, DARK_GRAY)
            self.screen.blit(text, (panel_x + 20, panel_y + 50 + i * 30))
    
    def get_action(self):
        """Get action from model or random"""
        if self.use_trained_model and self.model:
            action, _ = self.model.predict(self.obs, deterministic=True)
            return int(action)
        else:
            return self.env.action_space.sample()
    
    def run(self):
        """Main visualization loop"""
        running = True
        paused = False
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_r:
                        # Reset episode
                        self.obs, self.info = self.env.reset()
                        self.total_reward = 0
                        self.action_history = []
            
            # Clear screen
            self.screen.fill(WHITE)
            
            # Draw all panels
            self.draw_header()
            self.draw_patient_queue()
            self.draw_resources()
            self.draw_statistics()
            self.draw_action_panel()
            self.draw_instructions()
            
            # Update display
            pygame.display.flip()
            self.clock.tick(FPS)
            
            # Take action if not paused
            if not paused:
                action = self.get_action()
                self.obs, reward, terminated, truncated, self.info = self.env.step(action)
                
                self.last_action = action
                self.last_reward = reward
                self.total_reward += reward
                self.action_history.append(action)
                
                if terminated or truncated:
                    print(f"\n{'='*60}")
                    print(f"Episode Complete!")
                    print(f"Days: {self.info['day']}")
                    print(f"Total Reward: {self.total_reward:.2f}")
                    print(f"Detections: {self.info['total_detections']}")
                    print(f"Severe Cases: {self.info['severe_progressions']}")
                    print(f"{'='*60}\n")
                    
                    # Wait 2 seconds then reset
                    pygame.time.wait(2000)
                    self.obs, self.info = self.env.reset()
                    self.total_reward = 0
                    self.action_history = []
        
        self.env.close()
        pygame.quit()


if __name__ == "__main__":
    import sys
    
    print("="*60)
    print("AI-Veins Healthcare Resource Allocation Visualizer")
    print("="*60)
    
    # Check for trained model argument
    use_trained = False
    model_path = None
    
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        use_trained = True
        print(f"Loading trained model: {model_path}")
    else:
        print("No model specified - using RANDOM agent")
        print("To use trained model: python visualize.py saved_models/ppo_model.zip")
    
    print("\nControls:")
    print("  SPACE - Pause/Resume")
    print("  R - Reset episode")
    print("  ESC - Quit")
    print("\nStarting visualization...\n")
    
    visualizer = HealthcareVisualizer(use_trained_model=use_trained, model_path=model_path)
    visualizer.run()