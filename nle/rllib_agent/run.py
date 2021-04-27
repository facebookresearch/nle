import argparse
import os

import ray
from ray import tune
from ray.rllib.agents import impala
from ray.tune.integration.wandb import WandbLoggerCallback

from nle.rllib_agent.environments import RLLibNLEEnv  # noqa: F401
from nle.rllib_agent.models import RLLibNLENetwork  # noqa: F401


def generate_arguments():
    parser = argparse.ArgumentParser(description="RLLib NLE Agent")

    # Experiment Details
    parser.add_argument("--id", type=str, default="default", help="Experiment ID")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument(
        "--timesteps_total",
        type=int,
        default=int(1e7),
        help="How many timesteps to run for",
    )

    # Environment Details
    parser.add_argument(
        "--env", type=str, default="NetHackScore-v0", help="Which NLE setting to play"
    )
    parser.add_argument(
        "--env_observation_keys",
        type=str,
        default="blstats,glyphs",
        help="What keys to request from the NLE environment, comma-separated",
    )

    # WandB
    parser.add_argument(
        "--use_wandb", action="store_true", help="Whether to use wandb logging"
    )
    parser.add_argument(
        "--wandb_tags",
        type=str,
        default="rllib",
        help="Wandb tags for this run, comma separated",
    )
    parser.add_argument("--wandb_project", type=str, help="Wandb project for this run")
    parser.add_argument("--wandb_group", type=str, help="Wandb group for this run")
    parser.add_argument("--wandb_entity", type=str, help="Wandb entity for this run")

    # Training details
    parser.add_argument("--batch_size", type=int, default=2048, help="batch size")
    parser.add_argument("--lr", type=float, default=2.5e-4, help="learning rate")

    args = parser.parse_args()
    return args


def train(args) -> None:
    os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"  # Only log to wandb
    config = impala.DEFAULT_CONFIG.copy()
    config.update(
        {
            "framework": "torch",
            "num_gpus": 0,
            "seed": args.seed,
            "env": "rllib_nle_env",
            "env_config": {
                "env": args.env,
                "observation_keys": args.env_observation_keys.split(","),
            },
            "train_batch_size": args.batch_size,
            "lr": args.lr,
            "model": {
                "custom_model": "rllib_nle_model",
                "custom_model_config": {
                    "embedding_dim": 32,
                    "crop_dim": 9,
                    "num_layers": 5,
                    "hidden_dim": 512,
                },
                "use_lstm": True,
                "lstm_use_prev_reward": True,
                "lstm_use_prev_action": True,
                "lstm_cell_size": 512,  # same as h_dim in models.NetHackNet
            },
            "evaluation_interval": 100,
            "evaluation_num_episodes": 50,
            "evaluation_config": {"explore": False},
        }
    )
    callbacks = []
    if args.use_wandb:
        callbacks.append(
            WandbLoggerCallback(
                project=args.wandb_project,
                api_key_file="~/.wandb_api_key",
                entity=args.wandb_entity,
                group=args.wandb_group,
                tags=args.wandb_tags.split(","),
            )
        )
    tune.run(
        impala.ImpalaTrainer,
        stop={"timesteps_total": args.timesteps_total},
        config=config,
        name=args.id,
        callbacks=callbacks,
    )


if __name__ == "__main__":
    parsed_args = generate_arguments()
    ray.init()
    train(parsed_args)
