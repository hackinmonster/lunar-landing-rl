import gymnasium as gym
import numpy as np
import torch
import argparse
from lunar_landing import QNetwork
import os
import time
import sys

# Set OpenGL environment variables
os.environ['MESA_LOADER_DRIVER_OVERRIDE'] = 'i965'
os.environ['LIBGL_DRI3_DISABLE'] = '1'

def run_model(model_path, episodes=10, render=True):
    """Run a trained model."""
    try:
        # Create environment with appropriate render mode
        if render:
            env = gym.make('LunarLander-v3', render_mode='human')
        else:
            env = gym.make('LunarLander-v3')
    
        # Set up device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
    
        # Check if model exists
        if not os.path.exists(model_path):
            print(f"Model file not found at {model_path}")
            return None
    
        # Create network and load weights
        try:
            qnetwork = QNetwork(state_size=8, action_size=4, seed=0).to(device)
            qnetwork.load_state_dict(torch.load(model_path, map_location=device))
            qnetwork.eval()
            print(f"Successfully loaded model from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
        scores = []
        try:
            for i_episode in range(episodes):
                state, _ = env.reset()
                score = 0
                done = False
            
                while not done:
                    # Select action
                    state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
                    with torch.no_grad():
                        action_values = qnetwork(state_tensor)
                    action = np.argmax(action_values.cpu().numpy())
                
                    # Take action
                    state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    score += reward
                    
                    # Add a small delay to make visualization easier to follow
                    if render:
                        time.sleep(0.01)  # 10ms delay
                
                scores.append(score)
                print(f'Episode {i_episode+1}\tScore: {score:.2f}')
        
            print(f'\nAverage Score: {np.mean(scores):.2f}')
            print(f'Max Score: {np.max(scores):.2f}')
            print(f'Min Score: {np.min(scores):.2f}')
            
        except KeyboardInterrupt:
            print("\nSimulation interrupted by user")
        except Exception as e:
            print(f"Error during simulation: {e}")
        finally:
            env.close()
            
        return scores
            
    except Exception as e:
        print(f"Error initializing environment: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='models/solved_model.pth', 
                      help='Path to the model file')
    parser.add_argument('--episodes', type=int, default=10, 
                      help='Number of episodes to run')
    parser.add_argument('--no-render', action='store_true', 
                      help='Disable rendering')
    args = parser.parse_args()
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    try:
        run_model(args.model, args.episodes, not args.no_render)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1) 