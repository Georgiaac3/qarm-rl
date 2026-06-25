# To learn how to use rsl-rl, see https://leggedrobotics.github.io/rsl_rl/guide/overview.html
import argparse

import yaml
from PIDEnv import PIDEnv
from rsl_rl.runners import OnPolicyRunner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-walking")

    # 1) Creating the environment
    env = PIDEnv()

    # 2) Load the YAML configuration used during training
    with open("config/my_training.yaml", "r", encoding="utf-8") as f:
        full_cfg = yaml.safe_load(f)
    train_cfg = full_cfg["runner"]

    # 3) Build the runner
    runner = OnPolicyRunner(
        env=env,
        train_cfg=train_cfg,
        log_dir="logs/my_experiment",  # Directory for saving checkpoints and logs
        device="cuda:0",  # Device to run the training on
    )

    # 4) Start training
    runner.learn(num_learning_iterations=1500)  # Specify the number of desired iterations

    # 5) Export the trained policy for deployment
    runner.export_policy_to_jit("logs/my_experiment/exported", filename="policy.pt")
    runner.export_policy_to_onnx("logs/my_experiment/exported", filename="policy.onnx")


if __name__ == "__main__":
    main()
