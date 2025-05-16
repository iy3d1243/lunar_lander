# Reinforcement Learning Agent for LunarLander

Demo : https://youtu.be/ScqVPRYRF-A

This project implements a Deep Q-Network (DQN) agent that successfully learns to land a lunar module in the LunarLander-v3 environment from OpenAI Gymnasium. The agent was trained to achieve an average score of 200+ over 100 consecutive episodes in approximately 900 episodes.

## Project Overview

The lunar lander simulation presents a challenging control problem where an agent must learn to:
- Control thrust from the main engine and side boosters
- Navigate to the landing pad
- Land softly without crashing
- Minimize fuel consumption

This implementation uses a Deep Q-Network (DQN) algorithm, which combines Q-learning with deep neural networks to handle high-dimensional state spaces.

## Project Structure

```
.
├── data/                  # Directory for storing training data and plots
├── models/                # Directory for storing trained model weights
├── src/                   # Source code
│   ├── __init__.py        # Makes src a Python package
│   ├── agent.py           # DQN agent implementation
│   ├── model.py           # Neural network architecture
│   ├── train.py           # Training script
│   └── test.py            # Testing/evaluation script
├── main.py                # Main script to run training or testing
├── README.md              # Project documentation
└── requirements.txt       # Python dependencies
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/lunar-lander-rl.git
cd lunar-lander-rl
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. For the Box2D dependency (required for LunarLander), you may need:
```bash
pip install swig
pip install gymnasium[box2d]
```

Note: If you're on Windows and having issues with the Box2D installation, you might need to install Visual C++ Build Tools from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

## Training the Agent

To train the agent:

```bash
python main.py train
```

This will:
1. Create a DQN agent
2. Train it on the LunarLander-v3 environment
3. Save the model weights to the `models/` directory
4. Generate a plot of scores in the `data/` directory

## Testing the Agent

To test a trained agent:

```bash
python main.py test --model_path models/checkpoint_YYYYMMDD_HHMMSS.pth
```

Optional arguments:
- `--episodes N`: Number of episodes to run (default: 10)
- `--no_render`: Disable rendering the environment

## Implementation Details

### DQN Algorithm

The implementation uses the following DQN features:
- **Experience Replay Buffer**: Stores transitions (state, action, reward, next_state, done) and randomly samples from this buffer to break correlations between consecutive samples.
- **Target Network**: A separate network that is updated less frequently to provide stable Q-value targets.
- **Epsilon-Greedy Exploration**: Balances exploration and exploitation by sometimes taking random actions.
- **Soft Updates**: Gradually updates the target network to track the primary network, improving stability.

### Neural Network Architecture

The Q-network consists of:
- Input layer: 8 neurons (state size)
- Two hidden layers: 64 neurons each with ReLU activation
- Output layer: 4 neurons (action size)

### Hyperparameters

- Replay buffer size: 100,000
- Batch size: 64
- Discount factor (gamma): 0.99
- Soft update parameter (tau): 0.001
- Learning rate: 0.0005
- Network update frequency: Every 4 steps
- Epsilon start: 1.0
- Epsilon end: 0.01
- Epsilon decay: 0.995

## The Learning Process

1. The agent interacts with the environment, collecting experiences.
2. Experiences are stored in a replay buffer.
3. Periodically, the agent samples random batches from this buffer.
4. For each sampled experience, the agent:
   - Computes the target Q-value using the Bellman equation
   - Updates the Q-network to minimize the difference between predicted and target Q-values
5. The target network is slowly updated to track the Q-network.
6. As training progresses, epsilon decreases, shifting from exploration to exploitation.
7. Training continues until the agent solves the environment or reaches the maximum number of episodes.

## Results

The agent solved the environment in approximately 900 episodes, achieving an average score of 200+ over 100 consecutive episodes. This demonstrates the effectiveness of the DQN algorithm for this control task.

## Future Improvements

Potential enhancements to this implementation:
- Implement Double DQN to reduce overestimation of Q-values
- Add Prioritized Experience Replay for more efficient learning
- Implement Dueling DQN architecture
- Try different neural network architectures

## License

This project is open source and available under the MIT License.

## Acknowledgments

- OpenAI Gymnasium for the LunarLander environment
- Deep Q-Learning algorithm paper: "Human-level control through deep reinforcement learning" (Mnih et al., 2015)
