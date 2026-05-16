#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FINDER Main Training Script

Usage:
    python train.py --variant cn

Variant options: cn, cn_cost, nd, nd_cost
"""

from trainers.vector_trainer import FinderVectorTrainer
from trainers.config import load_variant_config, FinderVariant
import argparse
import sys
from pathlib import Path
import os
import platform
import random
import multiprocessing as mp

import numpy as np

try:
    import torch
except Exception:
    torch = None

# Add project root directory to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def _setup_determinism_and_spawn(seed: int = 42) -> None:
    """
    Configure multiprocessing and random seeds before vector environments spawn.
    """
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ["FINDER_SEED"] = str(seed)

    if platform.system() == "Linux":
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

    random.seed(seed)
    np.random.seed(seed)

    if torch is None:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cuda"):
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
        except Exception:
            pass
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
    except Exception:
        pass


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train FINDER DQN variants.")
    parser.add_argument(
        "--variant",
        choices=["cn", "cn_cost", "nd", "nd_cost"],
        default="cn",
        help="FINDER variant to train.",
    )
    parser.add_argument(
        "--full-tricks",
        action="store_true",
        help="Use configs/cn_full_config.json. Only applies to the cn variant.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.environ.get("FINDER_SEED", "42")),
        help="Global seed used before vector environments are constructed.",
    )
    parser.add_argument(
        "--device", help="Override config device: cpu, cuda, or cuda:N.")
    parser.add_argument(
        "--cuda-device",
        type=int,
        help="Select CUDA GPU index, e.g. 1 maps to cuda:1.",
    )
    parser.add_argument("--max-iterations", type=int,
                        help="Override training.max_iterations.")
    parser.add_argument("--batch-size", type=int,
                        help="Override training.batch_size.")
    parser.add_argument("--learning-rate", type=float,
                        help="Override training.learning_rate.")
    parser.add_argument("--num-envs", type=int,
                        help="Override vector_env.num_envs.")
    parser.add_argument(
        "--sync-env",
        action="store_true",
        help="Use SyncVectorEnv instead of AsyncVectorEnv.",
    )
    parser.add_argument("--eval-freq", type=int,
                        help="Override training.eval_freq.")
    parser.add_argument("--save-freq", type=int,
                        help="Override training.save_freq.")
    parser.add_argument("--eval-episodes", type=int,
                        help="Override training.num_eval_episodes.")
    parser.add_argument("--eval-envs", type=int,
                        help="Override training.num_eval_envs.")
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Disable training-time evaluation.",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        help="Override training.max_grad_norm.",
    )
    parser.add_argument(
        "--base-dir",
        help="Override experiment base directory.",
    )
    parser.add_argument(
        "--experiment-name",
        help="Override experiment name under base-dir.",
    )
    return parser


def _validate_cli_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.device:
        valid_device = args.device in ("cpu", "cuda") or (
            args.device.startswith("cuda:") and args.device[5:].isdigit()
        )
        if not valid_device:
            parser.error("--device must be one of: cpu, cuda, cuda:N")

    if args.cuda_device is not None:
        if args.cuda_device < 0:
            parser.error("--cuda-device must be >= 0")
        if args.device not in (None, "cuda"):
            parser.error(
                "--cuda-device can only be combined with --device cuda or used alone")


def _infer_experiment_paths(config):
    save_dir = Path(config.save_dir)
    if save_dir.name == "models" and save_dir.parent.name:
        exp_dir = save_dir.parent
        return exp_dir.parent, exp_dir.name
    return Path("./experiments"), f"{config.variant.value}_experiment"


def _apply_cli_overrides(config, args: argparse.Namespace) -> None:
    if args.cuda_device is not None:
        config.device = f"cuda:{args.cuda_device}"
    elif args.device:
        config.device = args.device
    if args.max_iterations is not None:
        config.training.max_iterations = args.max_iterations
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.learning_rate is not None:
        config.training.learning_rate = args.learning_rate
    if args.num_envs is not None:
        config.vector_env.num_envs = args.num_envs
    if args.sync_env:
        config.vector_env.async_env = False
    if args.eval_freq is not None:
        config.training.eval_freq = args.eval_freq
    if args.save_freq is not None:
        config.training.save_freq = args.save_freq
    if args.eval_episodes is not None:
        config.training.num_eval_episodes = args.eval_episodes
    if args.eval_envs is not None:
        config.training.num_eval_envs = args.eval_envs
    if args.no_eval:
        config.training.enable_training_eval = False
    if args.max_grad_norm is not None:
        config.training.max_grad_norm = args.max_grad_norm

    if args.base_dir or args.experiment_name:
        default_base_dir, default_exp_name = _infer_experiment_paths(config)
        base_dir = Path(args.base_dir) if args.base_dir else default_base_dir
        exp_name = args.experiment_name if args.experiment_name else default_exp_name
        exp_dir = base_dir / exp_name
        config.save_dir = str(exp_dir / "models")
        config.log_dir = str(exp_dir / "logs")


def load_full_tricks_config():
    """
    Load full-tricks CN configuration.

    Includes all advanced DQN techniques:
    - Dueling DQN: Separate value and advantage functions
    - Double DQN: Reduce Q-value overestimation
    - Huber Loss: More stable gradients
    - Prioritized experience replay: Train on important experiences first
    - N-step learning: Multi-step temporal difference
    """
    import json
    from trainers.config import NetworkConfig, ModelConfig, TrainingConfig, VectorEnvConfig, FinderTrainingConfig

    config_path = project_root / 'configs' / 'cn_full_config.json'

    if not config_path.exists():
        raise FileNotFoundError(
            f"Full-tricks config file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    # Create configuration object
    network_config = NetworkConfig(
        min_nodes=cfg['network']['min_nodes'],
        max_nodes=cfg['network']['max_nodes'],
        graph_type=cfg['network']['graph_type'],
        training_type=cfg['network']['training_type'],
        n_train_graphs=cfg['network']['n_train_graphs'],
        n_valid_graphs=cfg['network']['n_valid_graphs']
    )

    model_config = ModelConfig(
        embedding_size=cfg['model']['embedding_size'],
        reg_hidden=cfg['model']['reg_hidden'],
        aux_dim=cfg['model']['aux_dim'],
        max_bp_iter=cfg['model']['max_bp_iter'],
        aggregator_id=cfg['model']['aggregator_id'],
        embedding_method=cfg['model']['embedding_method'],
        initialization_stddev=cfg['model']['initialization_stddev']
    )

    training_config = TrainingConfig(
        gamma=cfg['training']['gamma'],
        learning_rate=cfg['training']['learning_rate'],
        batch_size=cfg['training']['batch_size'],
        memory_size=cfg['training']['memory_size'],
        n_step=cfg['training']['n_step'],
        max_iterations=cfg['training']['max_iterations'],
        update_target_freq=cfg['training']['update_target_freq'],
        eval_freq=cfg['training']['eval_freq'],
        save_freq=cfg['training']['save_freq'],
        graph_pool_update_freq=cfg['training']['graph_pool_update_freq'],
        sampling_freq=cfg['training']['sampling_freq'],
        episodes_per_sampling=cfg['training']['episodes_per_sampling'],
        eps_start=cfg['training']['eps_start'],
        eps_end=cfg['training']['eps_end'],
        eps_decay_steps=cfg['training']['eps_decay_steps'],
        alpha=cfg['training']['alpha'],
        max_grad_norm=cfg['training'].get('max_grad_norm', 5.0),
        is_huber_loss=cfg['training']['is_huber_loss'],
        is_double_dqn=cfg['training']['is_double_dqn'],
        is_dueling_dqn=cfg['training']['is_dueling_dqn'],
        is_prioritized_sampling=cfg['training']['is_prioritized_sampling'],
        is_multi_step_dqn=cfg['training']['is_multi_step_dqn'],
        priority_epsilon=cfg['training']['priority_epsilon'],
        priority_alpha=cfg['training']['priority_alpha'],
        priority_beta=cfg['training']['priority_beta'],
        priority_beta_increment=cfg['training']['priority_beta_increment'],
        td_err_upper=cfg['training']['td_err_upper'],
        enable_training_eval=cfg['training'].get('enable_training_eval', True),
        num_eval_episodes=cfg['training'].get('num_eval_episodes', 32),
        num_eval_envs=cfg['training'].get('num_eval_envs', 8),
        eval_epsilon=cfg['training'].get('eval_epsilon', 0.0)
    )

    vector_env_config = VectorEnvConfig(
        num_envs=cfg['vector_env']['num_envs'],
        async_env=cfg['vector_env']['async_env'],
        max_episode_steps=cfg['vector_env']['max_episode_steps']
    )

    # Set experiment path
    base_dir = cfg.get('base_dir', './experiments')
    exp_name = cfg.get('experiment_name', 'cn_full_experiment')
    exp_dir = Path(base_dir) / exp_name

    return FinderTrainingConfig(
        variant=FinderVariant.CN,
        network=network_config,
        model=model_config,
        training=training_config,
        vector_env=vector_env_config,
        save_dir=str(exp_dir / 'models'),
        log_dir=str(exp_dir / 'logs'),
        device=cfg.get('device', 'cuda')
    )


def main(argv=None):
    parser = _build_arg_parser()
    args = parser.parse_args([] if argv is None else argv)
    _validate_cli_args(parser, args)
    variant_name = args.variant
    use_full_tricks = args.full_tricks

    try:
        # Must be called before FinderVectorTrainer(config) construction; the constructor creates AsyncVectorEnv child processes.
        _setup_determinism_and_spawn(seed=args.seed)

        # Load different config files based on selection
        if use_full_tricks:
            if variant_name != 'cn':
                print(
                    f"Warning: Full-tricks config currently only supports CN variant, will use standard {variant_name} config")
                use_full_tricks = False
            else:
                print(f"Loading {variant_name} variant full-tricks config...")
                config = load_full_tricks_config()

        if not use_full_tricks:
            print(f"Loading {variant_name} variant standard config...")
            variant = FinderVariant(variant_name)
            config = load_variant_config(variant)

        _apply_cli_overrides(config, args)

        print(f"Training variant: {variant_name.upper()}")
        print(f"Random seed: {args.seed}")
        print(f"Training device: {config.device}")
        print(f"Experiment directory: {Path(config.save_dir).parent}")
        print("-" * 30)

        # Create and start trainer
        print("Initializing trainer...")
        trainer = FinderVectorTrainer(config)

        print("Starting training...")
        trainer.train()

        print("Training complete!")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")

    except FileNotFoundError as e:
        print(
            f"Error: Config file not found configs/{variant_name}_config.json")
        sys.exit(1)

    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    finally:
        # Cleanup resources
        try:
            if 'trainer' in locals():
                trainer.cleanup()
        except:
            pass


if __name__ == "__main__":
    main(sys.argv[1:])
