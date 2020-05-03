import gym
import numpy as np
import tensorflow as tf
import argparse


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, default='../logs/output')
    parser.add_argument('--input_path', type=str, default='model/input')
    parser.add_argument('--max_epochs', type=str, default=1)
    return parser.parse_known_args()


print("Tuning model......")
args, unknown = _parse_args()
print(f"output_path: {args.output_path}")
print(f"max_epochs: {args.max_epochs}")
env = gym.make('CartPole-v1')
obs = env.reset()
print(f"obs: {obs}")
