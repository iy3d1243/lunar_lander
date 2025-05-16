import gymnasium as gym
import torch
import numpy as np
import argparse
import os

from src.agent import Agent

def evaluate_agent(agent, env, n_episodes=10, render=True):
    """Evaluate the agent's performance.

    Params
    ======
        agent: The DQN agent
        env: The environment
        n_episodes (int): Number of episodes to evaluate
        render (bool): Whether to render the environment
    """
    scores = []

    for i in range(n_episodes):
        state, _ = env.reset()
        score = 0
        done = False

        while not done:
            if render:
                env.render()

            action = agent.act(state, eps=0.0)  # Greedy policy
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            score += reward

        scores.append(score)
        print(f"Episode {i+1}/{n_episodes}, Score: {score:.2f}")

    avg_score = np.mean(scores)
    print(f"Average Score over {n_episodes} episodes: {avg_score:.2f}")

    return avg_score

def main():
    parser = argparse.ArgumentParser(description='Test a trained DQN agent on LunarLander')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model weights')
    parser.add_argument('--episodes', type=int, default=10, help='Number of episodes to evaluate')
    parser.add_argument('--no_render', action='store_true', help='Disable rendering')

    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"Error: Model file {args.model_path} does not exist.")
        return

    try:
        # Try to import Box2D
        try:
            import Box2D
            print("Box2D is installed. Proceeding with evaluation...")
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
        env = gym.make('LunarLander-v3', render_mode='human' if not args.no_render else None)

        # Get state and action dimensions
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        print(f"State size: {state_size}, Action size: {action_size}")

        # Create agent
        agent = Agent(state_size=state_size, action_size=action_size, seed=0)

        # Load trained weights
        agent.qnetwork_local.load_state_dict(torch.load(args.model_path))

        # Evaluate agent
        print(f"Evaluating agent using model: {args.model_path}")
        evaluate_agent(agent, env, n_episodes=args.episodes, render=not args.no_render)

        # Close environment
        env.close()

    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
