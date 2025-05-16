import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import torch
import os
import time
from datetime import datetime

from src.agent import Agent

def dqn(agent, env, n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995, save_dir='models'):
    """Deep Q-Learning.

    Params
    ======
        agent: The DQN agent
        env: The environment
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
        save_dir (str): directory to save model weights
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon

    # Create save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    start_time = time.time()

    for i_episode in range(1, n_episodes+1):
        state, _ = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon

        # Print progress
        print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}', end="")
        if i_episode % 100 == 0:
            print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}')

        # Save model when performance is good enough
        if np.mean(scores_window)>=200.0:
            print(f'\nEnvironment solved in {i_episode-100} episodes!\tAverage Score: {np.mean(scores_window):.2f}')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            torch.save(agent.qnetwork_local.state_dict(), f'{save_dir}/checkpoint_{timestamp}.pth')
            break

    # Save final model regardless of performance
    if i_episode == n_episodes:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        torch.save(agent.qnetwork_local.state_dict(), f'{save_dir}/checkpoint_final_{timestamp}.pth')

    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds")

    return scores

def plot_scores(scores, filename='scores_plot.png'):
    """Plot scores and save the figure."""
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig(filename)
    plt.close()

def main():
    """Main function to train the agent."""
    try:
        # Try to import Box2D
        try:
            import Box2D
            print("Box2D is installed. Proceeding with training...")
        except ImportError:
            print("\nBox2D is not installed. To install it, run the following commands:")
            print("pip install swig")
            print("pip install gymnasium[box2d]")
            print("\nIf you're on Windows and having issues with the installation, you might need to:")
            print("1. Install Visual C++ Build Tools: https://visualstudio.microsoft.com/visual-cpp-build-tools/")
            print("2. Try installing a pre-built wheel for Box2D")
            return

        # Create environment
        print("Creating LunarLander environment...")
        env = gym.make('LunarLander-v3')

        # Get state and action dimensions
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        print(f"State size: {state_size}, Action size: {action_size}")

        # Create agent
        agent = Agent(state_size=state_size, action_size=action_size, seed=0)

        # Train agent
        print("Starting training...")
        scores = dqn(agent, env)

        # Plot results
        plot_scores(scores, filename='data/scores_plot.png')

        # Close environment
        env.close()

    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
