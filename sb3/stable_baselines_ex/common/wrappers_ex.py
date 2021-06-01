import gym
import numpy as np
from gym import spaces

from stable_baselines3.common.running_mean_std import RunningMeanStd


class ScaleRewardEnv(gym.RewardWrapper):
    def __init__(self, env: gym.Env, scale):
        gym.RewardWrapper.__init__(self, env)
        self.scale = scale

    def reward(self, reward: float) -> float:
        return reward * self.scale


class RepeatGoalEnv(gym.Wrapper):
    def __init__(
            self,
            env: gym.Env,
            gamma,
            max_d,
            max_t,
            lambda_dt,
            anoise_type=None,
            anoise_prob=0.,
            anoise_std=0.,
    ):
        gym.Wrapper.__init__(self, env)
        self.epsilon_std = 1e-3
        self.gamma = gamma
        self.max_d = max_d
        self.max_t = max_t
        self.lambda_dt = lambda_dt
        self.anoise_type = anoise_type
        self.anoise_prob = anoise_prob
        self.anoise_std = anoise_std

        self.body_key = None
        part_keys = set(self.env.sim.model._body_name2id.keys())
        target_keys = ['torso', 'cart', 'body1']
        for target_key in target_keys:
            if target_key in part_keys:
                self.body_key = target_key
                break

        if self.anoise_type in ['ext_fpc']:
            low = np.concatenate([self.observation_space.low, [-np.inf] * 3])
            high = np.concatenate([self.observation_space.high, [np.inf] * 3])
            self.observation_space = spaces.Box(
                low=low, high=high,
                shape=(self.observation_space.shape[0] + 3,), dtype=self.observation_space.dtype,
            )
            self.obs_dim = self.observation_space.shape[0] + 3
            self.cur_force = np.zeros(3)
        else:
            self.obs_dim = self.observation_space.shape[0]

        action_dim = self.env.action_space.shape[0]
        self.ori_action_dim = action_dim
        low = self.env.action_space.low
        high = self.env.action_space.high
        if self.max_d is not None or self.max_t is not None:
            action_dim += 1
            low = np.r_[low, -1.]
            high = np.r_[high, 1.]

        self.action_space = spaces.Box(
            low=low, high=high, shape=(action_dim,), dtype=env.action_space.dtype
        )

        self.cur_obs = None

        self.obs_rms = RunningMeanStd(shape=self.observation_space.shape)
        self.reset_update_obs_estimate = False
        self.num_steps = 0

        self.eval_mode = False

    def _update_obs_estimate(self, obs):
        if not self.eval_mode:
            self.obs_rms.update(obs[:, :self.obs_dim])

    def step(self, aug_action):
        cur_idx = self.ori_action_dim
        action = aug_action[:self.ori_action_dim]

        if self.anoise_type == 'action':
            if np.random.rand() < self.anoise_prob:
                action = action + np.random.randn(*action.shape) * self.anoise_std
                action = np.clip(action, self.action_space.low[:len(action)], self.action_space.high[:len(action)])
        elif self.anoise_type is not None and 'ext' in self.anoise_type:
            if np.random.rand() < self.anoise_prob:
                if self.env.spec.id == 'Reacher-v2':
                    force = np.zeros(3)
                    torque = np.random.randn(3) * self.anoise_std
                    cur_info = torque
                else:
                    force = np.random.randn(3) * self.anoise_std
                    torque = np.zeros(3)
                    cur_info = force

                if self.anoise_type == 'ext_fpc':
                    self.cur_force = np.clip(cur_info, -1, 1)
                self.env.sim.data.xfrc_applied[self.env.sim.model._body_name2id[self.body_key], :] = np.r_[
                    force, torque]
            else:
                self.env.sim.data.xfrc_applied[self.env.sim.model._body_name2id[self.body_key], :] = [0] * 6

        if self.max_d is not None or self.max_t is not None:
            u = aug_action[cur_idx]
            cur_idx += 1
            norm_u = (u + 1) / 2
            u = norm_u
        else:
            u = None

        lambda_dt = self.lambda_dt

        total_reward = 0.0
        done = None
        cur_gamma = 1.0
        first_obs = self.cur_obs
        for i in range(100000000):
            obs, reward, done, info = self.env.step(action)

            if self.anoise_type in ['ext_fpc']:
                obs = np.concatenate([obs, self.cur_force])

            if not done:
                self._update_obs_estimate(obs[np.newaxis, ...])
                self.reset_update_obs_estimate = True
            total_reward += reward * cur_gamma
            cur_gamma *= self.gamma
            if done:
                break

            if self.max_d is None and self.max_t is None:
                break

            if self.max_t is not None:
                t_delta = (i + 1) * self.env.dt

            if self.max_d is not None:
                norm_obs = (obs - self.obs_rms.mean) / (np.sqrt(self.obs_rms.var) + self.epsilon_std)
                norm_first_obs = (first_obs - self.obs_rms.mean) / (np.sqrt(self.obs_rms.var) + self.epsilon_std)

                d_delta = np.linalg.norm(norm_obs - norm_first_obs, ord=1) / len(obs)

            if self.max_d is not None and self.max_t is not None:
                if lambda_dt is None:
                    if d_delta >= u * self.max_d:
                        break
                    if t_delta >= self.max_t:
                        break
                else:
                    ori_t_delta = t_delta
                    t_delta = t_delta / self.max_t
                    d_delta = d_delta / self.max_d
                    delta = lambda_dt * d_delta + (1 - lambda_dt) * t_delta
                    if delta >= u:
                        break
                    if ori_t_delta >= self.max_t:
                        break
            elif self.max_t is not None:
                if t_delta >= u * self.max_t:
                    break
            elif self.max_d is not None:
                if d_delta >= u * self.max_d:
                    break

        self.cur_obs = obs
        info['w'] = i + 1
        info['t_diff'] = (i + 1) * self.env.dt
        if u is not None:
            if self.max_d is not None and self.max_t is not None:
                pass
            elif self.max_t is not None:
                info['t'] = u * self.max_t
            elif self.max_d is not None:
                info['d'] = u * self.max_d
            info['u'] = u
        if lambda_dt is not None:
            info['lambda_dt'] = lambda_dt

        self.num_steps += 1

        return self.cur_obs, total_reward, done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)

        if self.anoise_type in ['ext_fpc']:
            self.cur_force = np.zeros(3)
            obs = np.concatenate([obs, self.cur_force])

        if self.reset_update_obs_estimate:
            self._update_obs_estimate(obs[np.newaxis, ...])
            self.reset_update_obs_estimate = False
        self.cur_obs = obs
        return self.cur_obs
