import argparse
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

def make_env(env_id):
    def _thunk():
        return gym.make(env_id)
    return _thunk

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="Walker2d-v4")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--vecnorm_path", default=None)
    ap.add_argument("--episodes", type=int, default=5)
    args = ap.parse_args()

    venv = DummyVecEnv([make_env(args.env)])
    if args.vecnorm_path:
        venv = VecNormalize.load(args.vecnorm_path, venv)
        venv.training = False
        venv.norm_reward = False

    model = PPO.load(args.model_path)

    speeds = []
    for _ in range(args.episodes):
        obs = venv.reset()
        done = False
        ep_v = []
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, info = venv.step(action)
            inf0 = info[0] if isinstance(info, (list, tuple)) else info
            if "x_velocity" in inf0:
                ep_v.append(float(inf0["x_velocity"]))
        if ep_v:
            speeds.append(float(np.mean(ep_v)))

    v_cmd = float(np.mean(speeds)) if speeds else 0.6
    print(f"{v_cmd:.4f}")

if __name__ == "__main__":
    main()
