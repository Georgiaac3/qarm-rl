#!/usr/bin/env python3
"""
Testing for bin picking policy.

Quick reference:
  python3 test_all.py              # All tests
  python3 test_all.py --eval       # Batch evaluation (10 episodes)
  python3 test_all.py --interactive # Interactive single episode
  python3 test_all.py --visualize  # Generate prediction visualizations
"""

import sys
import argparse
from pathlib import Path

# Add parent dirs to path
sys.path.insert(0, str(Path(__file__).parent))


def main():
    parser = argparse.ArgumentParser(
        description="Test trained bin picking policy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 test_all.py --eval                    # Run 10 evaluation episodes
  python3 test_all.py --interactive --steps 50  # Interactive test with 50 steps
  python3 test_all.py --visualize               # Generate prediction visualizations
  python3 test_all.py --eval --episodes 20      # Run 20 evaluation episodes
        """)
    
    parser.add_argument("--eval", action="store_true",
                        help="Run batch evaluation on multiple episodes")
    parser.add_argument("--interactive", action="store_true",
                        help="Run interactive single-episode test")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate prediction visualizations")
    parser.add_argument("--random", action="store_true",
                        help="Use random policy baseline instead of trained model")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of evaluation episodes (default: 10)")
    parser.add_argument("--steps", type=int, default=50,
                        help="Max steps for interactive test (default: 50)")
    parser.add_argument("--model", type=str, default="training/checkpoints/policy.pt",
                        help="Path to saved model checkpoint")
    parser.add_argument("--output", type=str, default="visualization_outputs",
                        help="Output directory for visualizations")
    
    args = parser.parse_args()
    
    # Default to eval if no test type specified
    if not (args.eval or args.interactive or args.visualize):
        args.eval = True
    
    if args.eval:
        from evaluate import evaluate_policy
        evaluate_policy(
            model_path=args.model,
            n_eval_episodes=args.episodes,
            use_random=args.random,
        )
    
    if args.interactive:
        from evaluate import interactive_test
        interactive_test(
            model_path=args.model,
            n_steps=args.steps,
        )
    
    if args.visualize:
        from visualize import visualize_predictions
        visualize_predictions(
            model_path=args.model,
            save_dir=args.output,
        )


if __name__ == "__main__":
    main()
