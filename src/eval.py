import os
import argparse
import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from wrappers import EnergyRewardWrapper, MPCStabilityFilterWrapper


def make_single_env(env_id: str,
                    seed: int,
                    w_energy: float,
                    w_jerk: float,
                    fall_penalty: float,
                    max_steps: int,
                    use_mpc: bool):
    env = gym.make(env_id)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_steps)
    env = EnergyRewardWrapper(env, w_energy=w_energy, w_jerk=w_jerk, fall_penalty=fall_penalty)
    if use_mpc:
        env = MPCStabilityFilterWrapper(env, kv=0.01, max_tilt=1.2, min_scale=0.6)
    env.reset(seed=seed)
    return env


def rollout_gym(model, env, n_episodes: int, deterministic: bool = True):
    rows = []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep)
        x0 = float(env.unwrapped.data.qpos[0])

        energy = 0.0
        jerk = 0.0
        steps = 0
        fallen = False
        truncated = False

        while True:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, r, terminated, trunc, info = env.step(action)
            steps += 1

            energy += float(info.get("energy_cost", 0.0))
            jerk += float(info.get("jerk_cost", 0.0))

            if terminated or trunc:
                fallen = bool(terminated)
                truncated = bool(trunc)
                break

        x1 = float(env.unwrapped.data.qpos[0])
        dist = max(1e-6, x1 - x0)

        rows.append({
            "episode": ep,
            "distance": dist,
            "energy_proxy": energy,
            "jerk_proxy": jerk,
            "energy_per_m": energy / dist,
            "steps": steps,
            "fallen": float(fallen),
            "truncated": float(truncated),
        })
    return pd.DataFrame(rows)


def rollout_vec(model, vec_env, n_episodes: int, deterministic: bool = True):
    rows = []
    for ep in range(n_episodes):
        obs = vec_env.reset()

        energy = 0.0
        jerk = 0.0
        steps = 0
        fallen = False
        truncated = False

        x0 = None
        x1 = None

        while True:
            if x0 is None:
                unwrapped = vec_env.get_attr("unwrapped")[0]
                x0 = float(unwrapped.data.qpos[0])

            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, infos = vec_env.step(action)
            steps += 1

            info0 = infos[0] if isinstance(infos, (list, tuple)) and len(infos) > 0 else {}
            energy += float(info0.get("energy_cost", 0.0))
            jerk += float(info0.get("jerk_cost", 0.0))

            if bool(done[0]) if isinstance(done, np.ndarray) else bool(done):
                finfo = info0.get("final_info", None)
                if isinstance(finfo, dict):
                    truncated = bool(finfo.get("TimeLimit.truncated", False))
                    fallen = not truncated

                    if "x_position" in finfo:
                        x1 = float(finfo["x_position"])
                break

        if x0 is None:
            x0 = 0.0
        if x1 is None:
            unwrapped = vec_env.get_attr("unwrapped")[0]
            x1 = float(unwrapped.data.qpos[0])

        dist = max(1e-6, x1 - float(x0))

        rows.append({
            "episode": ep,
            "distance": dist,
            "energy_proxy": energy,
            "jerk_proxy": jerk,
            "energy_per_m": energy / dist,
            "steps": steps,
            "fallen": float(fallen),
            "truncated": float(truncated),
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="Walker2d-v4")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--n_episodes", type=int, default=50)

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--vecnorm_path", type=str, default=None)
    parser.add_argument("--no_vecnorm", action="store_true")

    parser.add_argument("--outdir", type=str, default="./eval_out")
    parser.add_argument("--tag", type=str, default="run")

    parser.add_argument("--w_energy", type=float, default=0.0)
    parser.add_argument("--w_jerk", type=float, default=0.0)
    parser.add_argument("--fall_penalty", type=float, default=0.0)
    parser.add_argument("--use_mpc", action="store_true")

    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    deterministic = bool(args.deterministic)

    if args.no_vecnorm:
        env = make_single_env(
            env_id=args.env,
            seed=args.seed,
            w_energy=args.w_energy,
            w_jerk=args.w_jerk,
            fall_penalty=args.fall_penalty,
            max_steps=args.max_steps,
            use_mpc=args.use_mpc,
        )
        model = PPO.load(args.model_path)
        df = rollout_gym(model, env, n_episodes=args.n_episodes, deterministic=deterministic)
    else:
        if args.vecnorm_path is None:
            raise SystemExit("VecNormalize enabled but --vecnorm_path is missing. Pass --vecnorm_path or use --no_vecnorm.")

        vec_env = DummyVecEnv([lambda: make_single_env(
            env_id=args.env,
            seed=args.seed,
            w_energy=args.w_energy,
            w_jerk=args.w_jerk,
            fall_penalty=args.fall_penalty,
            max_steps=args.max_steps,
            use_mpc=args.use_mpc,
        )])
        vec_env = VecNormalize.load(args.vecnorm_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

        model = PPO.load(args.model_path, env=vec_env)
        df = rollout_vec(model, vec_env, n_episodes=args.n_episodes, deterministic=deterministic)

    summary = {
        "tag": args.tag,
        "use_mpc": bool(args.use_mpc),
        "use_vecnorm": (not args.no_vecnorm),
        "mean_energy_per_m": float(df["energy_per_m"].mean()),
        "std_energy_per_m": float(df["energy_per_m"].std()),
        "mean_distance": float(df["distance"].mean()),
        "fall_rate": float(df["fallen"].mean()),
        "time_limit_rate": float(df["truncated"].mean()),
        "mean_steps": float(df["steps"].mean()),
    }
    print(summary)

    suffix = f"{args.tag}{'_mpc' if args.use_mpc else ''}"
    df_path = os.path.join(args.outdir, f"metrics_{suffix}.csv")
    df.to_csv(df_path, index=False)

    plt.figure()
    plt.hist(df["energy_per_m"], bins=16)
    plt.xlabel("energy_per_m (proxy)")
    plt.ylabel("count")
    plt.title(f"Energy per meter ({suffix})")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, f"hist_energy_per_m_{suffix}.png"), dpi=200)
    plt.close()

    print("Saved:", df_path)


if __name__ == "__main__":
    main()
