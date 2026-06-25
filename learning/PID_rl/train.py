# To learn how to use rsl-rl, see https://leggedrobotics.github.io/rsl_rl/guide/overview.html
import argparse

import yaml
from PIDEnv import PIDEnv
from rsl_rl.runners import OnPolicyRunner

import genesis as gs


def main():
    #################################
    # Parsing command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="PID")
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=101)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    log_dir = f"logs/{args.exp_name}"

    ###########################################################
    # Getting the training and environment configuration files
    with open("train_cfg.yaml", "r", encoding="utf-8") as f:
        full_cfg = yaml.safe_load(f)
    train_cfg = full_cfg["runner"]

    with open("env_cfg.yaml", "r", encoding="utf-8") as f:
        full_cfg = yaml.safe_load(f)
    env_cfg = full_cfg["env"]

    ###########################
    # Creating the environment
    gs.init(
        backend=gs.gpu,
        precision="32",
        logging_level="warning",
        seed=args.seed,
        performance_mode=True,
    )
    env = PIDEnv()

    # 3) Build the runner
    runner = OnPolicyRunner(
        env=env,
        train_cfg=train_cfg,
        log_dir=log_dir,  # Directory for saving checkpoints and logs
        device="cuda:0",  # Device to run the training on
    )

    # 4) Start training
    runner.learn(num_learning_iterations=1500)  # Specify the number of desired iterations

    # 5) Export the trained policy for deployment
    runner.export_policy_to_jit(log_dir + "/exported", filename="policy.pt")
    runner.export_policy_to_onnx(log_dir + "/exported", filename="policy.onnx")


if __name__ == "__main__":
    main()
