import argparse

import retro
import numpy as np
import gymnasium as gym
import os

from gymnasium.wrappers.time_limit import TimeLimit
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.atari_wrappers import ClipRewardEnv, WarpFrame
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecFrameStack,
    VecTransposeImage,
)
from stable_baselines3.common.monitor import Monitor
from utils import *

try:
    from tensorboard_logger import Logger
except ImportError:
    Logger = None

class StochasticFrameSkip(gym.Wrapper):
    def __init__(self, env, n, stickprob):
        gym.Wrapper.__init__(self, env)
        self.n = n
        self.stickprob = stickprob
        self.curac = None
        self.rng = np.random.RandomState()
        self.supports_want_render = hasattr(env, "supports_want_render")

    def reset(self, **kwargs):
        self.curac = None
        return self.env.reset(**kwargs)

    def step(self, ac):
        terminated = False
        truncated = False
        totrew = 0
        for i in range(self.n):
            # First step after reset, use action
            if self.curac is None:
                self.curac = ac
            # First substep, delay with probability=stickprob
            elif i == 0:
                if self.rng.rand() > self.stickprob:
                    self.curac = ac
            # Second substep, new action definitely kicks in
            elif i == 1:
                self.curac = ac
            if self.supports_want_render and i < self.n - 1:
                ob, rew, terminated, truncated, info = self.env.step(
                    self.curac,
                    want_render=False,
                )
            else:
                ob, rew, terminated, truncated, info = self.env.step(self.curac)
            totrew += rew
            if terminated or truncated:
                break
        return ob, totrew, terminated, truncated, info


def make_retro(*, game, state=None, max_episode_steps=1000000, **kwargs):
    if state is None:
        state = retro.State.DEFAULT
    env = retro.make(game, state, inttype=retro.data.Integrations.ALL, **kwargs)
    env = StochasticFrameSkip(env, n=4, stickprob=0.25)
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    return env


def wrap_deepmind_retro(env):
    """
    Configure environment for retro games, using config similar to DeepMind-style Atari in openai/baseline's wrap_deepmind
    """
    env = WarpFrame(env)
    env = ClipRewardEnv(env)
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--model_type", default="PPO")
    parser.add_argument("--model_policy", default="CnnPolicy")
    parser.add_argument("--game", default="FZero-Snes")
    # parser.add_argument("--state", default=retro.State.DEFAULT)
    parser.add_argument("--state", default="start-big-blue")
    parser.add_argument("--n_env", default=8)
    parser.add_argument("--tensorboard_log", default="./tboard_log")
    args = parser.parse_args()

    if Logger is not None:
        tensorboard_logger = Logger(args.tensorboard_log)

    chosenPolicy = args.model_policy
    def make_env():
        env = make_retro(game=args.game, state=args.state, scenario=None)
        env = Monitor(env, filename='monitor_log', allow_early_resets=True)
        env = FZeroDiscretizer(env)
        env = wrap_deepmind_retro(env)
        return env

    venv = VecTransposeImage(VecFrameStack(SubprocVecEnv([make_env] * args.n_env), n_stack=4))
    if args.model_path:
        match args.model_type:
            case "PPO":
                model = PPO.load(args.model_path, tensorboard_log=args.tensorboard_log)
                model.set_env(env)
    else:
        match args.model_type:
            case "PPO":
                model = PPO(
                policy=chosenPolicy,
                env=venv,
                learning_rate=lambda f: f * 2.5e-4,
                n_steps=128,
                batch_size=32,
                n_epochs=4,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.1,
                ent_coef=0.01,
                verbose=1,
                tensorboard_log=args.tensorboard_log
                )
            case "DQN":
                model = DQN(policy=chosenPolicy,
                env=venv,
                learning_rate=0.0001,
                buffer_size=100000,
                learning_starts=100,
                batch_size=32,
                tau=1.0,
                gamma=0.99,
                train_freq=4,
                gradient_steps=1,
                replay_buffer_class=None,
                replay_buffer_kwargs=None,
                optimize_memory_usage=False,
                target_update_interval=10000,
                exploration_fraction=0.1,
                exploration_initial_eps=1.0,
                exploration_final_eps=0.05,
                max_grad_norm=10,
                stats_window_size=100,
                verbose=1,
                tensorboard_log=args.tensorboard_log
                )
            case "A2C":
                model = A2C(
                policy=chosenPolicy,
                env=venv,
                learning_rate=lambda f: f * 2.5e-4,
                n_steps=128,
                gamma=0.99,
                gae_lambda=0.95,
                ent_coef=0.01,
                verbose=1,
                tensorboard_log=args.tensorboard_log
                )

    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./training_checkpoints',
                                             name_prefix=f"fzero-{args.model_type}")

    model.learn(
        total_timesteps=1_000_000,
        log_interval=1,
        callback=checkpoint_callback
    )
    model.save(f"fzero-{args.model_type}")

if __name__ == "__main__":
    main()