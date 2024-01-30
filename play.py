import argparse

import retro
import numpy as np
import gymnasium as gym
import os

from gymnasium.wrappers.time_limit import TimeLimit
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.atari_wrappers import ClipRewardEnv, WarpFrame
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecFrameStack,
    VecTransposeImage,
)
from utils import *


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


def make_retro(*, game, state=None, max_episode_steps=4500, **kwargs):
    if state is None:
        state = retro.State.DEFAULT
    env = retro.make(game, state, inttype=retro.data.Integrations.ALL, **kwargs)
    env = StochasticFrameSkip(env, n=4, stickprob=0.25)
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    return env


def wrap_deepmind_retro(env):
    env = WarpFrame(env)
    env = ClipRewardEnv(env)
    return env

def play(self, args, continuous=True):
        com_print('Start Play Loop')
        state = self.env.reset()
        while True:
            self.env.render(mode='human')

            p1_actions = self.p1_model.predict(state, deterministic=args.deterministic)
            
            state, reward, done, info = self.env.step(p1_actions[0])
            time.sleep(0.01)
            #print(reward)

            if done[0]:
                state = self.env.reset()

            if not continuous and done is True:
                return info

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="models/fzero-PPO-sand-ocean")
    parser.add_argument("--model_type", default="PPO")
    parser.add_argument("--game", default="FZero-Snes")
    parser.add_argument("--state", default="start-sand-ocean")
    # parser.add_argument("--state", default="start-race-practice")
    parser.add_argument("--n_env", default=8)
    parser.add_argument("--record", default=False)
    args = parser.parse_args()

    def make_env():
        if args.record != False:
            env = make_retro(game=args.game, state=args.state, scenario=None, record='.')
        else:
            env = make_retro(game=args.game, state=args.state, scenario=None)
        env = FZeroDiscretizer(env)
        env = wrap_deepmind_retro(env)
        return env

    venvs = VecTransposeImage(VecFrameStack(SubprocVecEnv([make_env] * args.n_env), n_stack=4))
    if args.model_path:
        match args.model_type:
            case "PPO":
                model = PPO.load(args.model_path)
                model.set_env(venvs)
            case "A2C":
                model = A2C.load(args.model_path)
                model.set_env(venvs)

    # create 1 env for testing
    venv = VecTransposeImage(VecFrameStack(SubprocVecEnv([make_env] * 1), n_stack=4))
    state = venv.reset()

    zero_completed_obs = np.zeros((args.n_env,) + venvs.observation_space.shape)
    zero_completed_obs[0, :] = state
    state = zero_completed_obs

    while True:

        actions = model.predict(state)
        
        state, reward, done, info = venv.step(actions[0])

        if done[0]:
            state = venv.reset()

        zero_completed_obs = np.zeros((args.n_env,) + venvs.observation_space.shape)
        zero_completed_obs[0, :] = state
        state = zero_completed_obs

if __name__ == "__main__":
    main()