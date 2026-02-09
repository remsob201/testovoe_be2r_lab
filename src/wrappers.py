import numpy as np
import gymnasium as gym


def quat_to_euler_xyzw(q):
    """
    Convert quaternion (x, y, z, w) to roll, pitch, yaw.
    """
    x, y, z, w = q
    # roll (вращение вокруг х)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # pitch (вокруг у)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.sign(sinp) * (np.pi / 2.0)
    else:
        pitch = np.arcsin(sinp)

    # yaw (вокру z)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


class EnergyRewardWrapper(gym.Wrapper):

    def __init__(self, env, w_energy=0.0, w_jerk=0.0, fall_penalty=0.0):
        super().__init__(env)
        self.w_energy = float(w_energy)
        self.w_jerk = float(w_jerk)
        self.fall_penalty = float(fall_penalty)
        self.prev_action = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_action = None
        return obs, info

    def step(self, action):
        obs, reward_env, terminated, truncated, info = self.env.step(action)

        a = np.array(action, dtype=np.float32)
        energy_cost = float(np.sum(a * a))
        jerk_cost = float(np.sum((a - self.prev_action) ** 2)) if self.prev_action is not None else 0.0
        self.prev_action = a.copy()

        reward = float(reward_env) - self.w_energy * energy_cost - self.w_jerk * jerk_cost

        # штраф за окончение не по плану
        if terminated:
            reward -= self.fall_penalty

        info = dict(info) if info is not None else {}
        info["reward_env"] = float(reward_env)
        info["energy_cost"] = energy_cost
        info["jerk_cost"] = jerk_cost

        return obs, reward, terminated, truncated, info


class MPCStabilityFilterWrapper(gym.Wrapper):

    def __init__(self, env, kv=0.05, max_tilt=0.8, min_scale=0.25):
        super().__init__(env)
        self.kv = float(kv)
        self.max_tilt = float(max_tilt)
        self.min_scale = float(min_scale)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def _extract_tilt_and_joint_vel(self):
        qpos = self.env.unwrapped.data.qpos
        qvel = self.env.unwrapped.data.qvel


        tilt = 0.0
        if len(qpos) >= 7:
            try:
                quat = np.array(qpos[3:7], dtype=np.float64)  
                roll, pitch, _ = quat_to_euler_xyzw(quat)
                tilt = float(np.sqrt(roll**2 + pitch**2))
            except Exception:
                tilt = 0.0
        elif len(qpos) > 2:
            tilt = float(abs(qpos[2]))

        act_dim = self.env.action_space.shape[0]
        joint_vel = np.array(qvel[-act_dim:], dtype=np.float32)

        return tilt, joint_vel

    def step(self, action):
        a = np.array(action, dtype=np.float32)
        tilt, joint_vel = self._extract_tilt_and_joint_vel()

        a_f = a - self.kv * joint_vel

        if tilt > self.max_tilt:
            a_f *= self.min_scale
        else:

            s = 1.0 - (tilt / self.max_tilt) * (1.0 - self.min_scale)
            a_f *= float(np.clip(s, self.min_scale, 1.0))

        low, high = self.action_space.low, self.action_space.high
        a_f = np.clip(a_f, low, high)

        return self.env.step(a_f)
