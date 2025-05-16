import argparse
import sys
import os

# Add the current directory to the path so we can import the src package
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.train import main as train_main
from src.test import main as test_main

def main():
    parser = argparse.ArgumentParser(description='DQN for LunarLander')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train the agent')

    # Test command
    test_parser = subparsers.add_parser('test', help='Test the agent')
    test_parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model weights')
    test_parser.add_argument('--episodes', type=int, default=10, help='Number of episodes to evaluate')
    test_parser.add_argument('--no_render', action='store_true', help='Disable rendering')

    args = parser.parse_args()

    if args.command == 'train':
        train_main()
    elif args.command == 'test':
        # Pass arguments to test_main through sys.argv
        import sys
        sys.argv = [sys.argv[0]]
        sys.argv.append('--model_path')
        sys.argv.append(args.model_path)
        sys.argv.append('--episodes')
        sys.argv.append(str(args.episodes))
        if args.no_render:
            sys.argv.append('--no_render')
        test_main()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
