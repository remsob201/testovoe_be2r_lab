import os
import argparse
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import set_random_seed

from wrappers import EnergyRewardWrapper


def make_env(env_id: str,
             seed: int,
             max_steps: int,
             use_energy_wrapper: bool,
             w_energy: float,
             w_jerk: float,
             fall_penalty: float):
    def _thunk():
        env = gym.make(env_id)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_steps)

        if use_energy_wrapper:
            env = EnergyRewardWrapper(
                env,
                w_energy=w_energy,
                w_jerk=w_jerk,
                fall_penalty=fall_penalty,
            )

        env.reset(seed=seed)
        return env
    return _thunk


def main():
    parser = argparse.ArgumentParser()

    # env
    parser.add_argument("--env", type=str, default="Walker2d-v4")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--n_envs", type=int, default=8)

    # длительность тренировки
    parser.add_argument("--timesteps", type=int, default=600_000)

    parser.add_argument("--use_energy_wrapper", action="store_true",
                        help="Enable EnergyRewardWrapper for energy-aware training/fine-tuning.")
    parser.add_argument("--w_energy", type=float, default=0.0)
    parser.add_argument("--w_jerk", type=float, default=0.0)
    parser.add_argument("--fall_penalty", type=float, default=0.0)

    # гиперпараметры PPO
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n_steps", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--ent_coef", type=float, default=0.0)
    parser.add_argument("--clip_range", type=float, default=0.2)


    parser.add_argument("--no_vecnorm", action="store_true",
                        help="Disable VecNormalize. Recommended for fast, robust baseline training.")
    parser.add_argument("--load_vecnorm", type=str, default=None,
                        help="Load vecnormalize.pkl (only if VecNormalize enabled).")

    parser.add_argument("--load_model", type=str, default=None,
                        help="Path to SB3 .zip model to load and continue training (fine-tune).")

    parser.add_argument("--outdir", type=str, default="./runs")
    parser.add_argument("--run_name", type=str, default="ppo_run")

    parser.add_argument("--device", type=str, default="cpu",
                        help="cpu recommended for SB3 PPO MLP. Use 'auto' if you want.")
    args = parser.parse_args()

    set_random_seed(args.seed)

    os.makedirs(args.outdir, exist_ok=True)
    run_dir = os.path.join(args.outdir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    use_vecnorm = (not args.no_vecnorm)
    env_fns = [
        make_env(
            env_id=args.env,
            seed=args.seed + i,
            max_steps=args.max_steps,
            use_energy_wrapper=args.use_energy_wrapper,
            w_energy=args.w_energy,
            w_jerk=args.w_jerk,
            fall_penalty=args.fall_penalty,
        )
        for i in range(args.n_envs)
    ]
    venv = DummyVecEnv(env_fns)
    venv = VecMonitor(venv)

    if use_vecnorm:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)
        if args.load_vecnorm is not None:
            venv = VecNormalize.load(args.load_vecnorm, venv)
            venv.training = True
            venv.norm_reward = False

    eval_env = DummyVecEnv([
        make_env(
            env_id=args.env,
            seed=args.seed + 10_000,
            max_steps=args.max_steps,
            use_energy_wrapper=args.use_energy_wrapper,
            w_energy=args.w_energy,
            w_jerk=args.w_jerk,
            fall_penalty=args.fall_penalty,
        )
    ])
    eval_env = VecMonitor(eval_env)
    if use_vecnorm:
        if args.load_vecnorm is not None:
            eval_env = VecNormalize.load(args.load_vecnorm, eval_env)
        else:
            eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
        eval_env.training = False
        eval_env.norm_reward = False

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=run_dir,
        log_path=run_dir,
        eval_freq=max(1, (20_000 // args.n_envs)),
        n_eval_episodes=10,
        deterministic=True,
    )

    if args.load_model is None:
        model = PPO(
            "MlpPolicy",
            venv,
            learning_rate=args.lr,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            ent_coef=args.ent_coef,
            clip_range=args.clip_range,
            verbose=1,
            tensorboard_log=run_dir,
            device=args.device,
        )
        reset_num_timesteps = True
    else:
        model = PPO.load(args.load_model, env=venv, device=args.device)
        reset_num_timesteps = False

    model.learn(
        total_timesteps=args.timesteps,
        callback=eval_cb,
        reset_num_timesteps=reset_num_timesteps,
    )

    model.save(os.path.join(run_dir, "final_model.zip"))
    if use_vecnorm:
        venv.save(os.path.join(run_dir, "vecnormalize.pkl"))

    print("Saved to:", run_dir)


if __name__ == "__main__":
    main()

