import argparse
import datetime
import os

import better_exceptions
import torch.multiprocessing as mp
from torch import nn

from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines_ex.common.callbacks_ex import EvalCallbackEx, CheckpointCallbackEx
from stable_baselines_ex.common.dummy_vec_env_ex import DummyVecEnvEx
from stable_baselines_ex.common.env_util_ex import make_vec_env_ex
from stable_baselines_ex.common.vec_normalize_ex import VecNormalizeEx
from stable_baselines_ex.common.wrappers_ex import RepeatGoalEnv
from stable_baselines_ex.rg.a2c_rg import A2CRg
from stable_baselines_ex.rg.evaluation_rg import evaluate_policy_rg
from stable_baselines_ex.rg.ppo_rg import PPORg

better_exceptions.hook()

EXP_DIR = 'exp'
GROUPS_DIR = 'groups'


def get_argparser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--run_group', type=str, default='Exp')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--load_path', type=str, default=None)

    parser.add_argument('--env', type=str, default='InvertedPendulum-v2')
    parser.add_argument('--algo', type=str, default='ppo_rg', choices=['ppo_rg', 'a2c_rg'])
    parser.add_argument('--frame_skip', type=int, default=None)
    parser.add_argument('--dt', type=float, default=None)
    parser.add_argument('--ref_gamma', type=float, default=0.99)

    parser.add_argument('--n_steps', type=int, default=int(1e6))
    parser.add_argument('--save_freq', type=int, default=int(1e9))
    parser.add_argument('--eval_freq', type=int, default=int(1e9))
    parser.add_argument('--n_eval_episodes', type=int, default=16)
    parser.add_argument('--log_freq', type=int, default=None)
    parser.add_argument('--deterministic_eval', type=int, default=1)

    parser.add_argument('--vec_norm', type=int, default=1, choices=[0, 1])
    parser.add_argument('--lr', type=float, default=1e-4)

    parser.add_argument('--model_dim', type=int, default=256)
    parser.add_argument('--model_layers', type=int, default=2)

    parser.add_argument('--max_d', type=float, default=None)
    parser.add_argument('--max_t', type=float, default=None)
    parser.add_argument('--lambda_dt', type=float, default=None)
    parser.add_argument('--anoise_prob', type=float, default=0)
    parser.add_argument('--anoise_std', type=float, default=0)
    parser.add_argument('--anoise_type', type=str, default=None, choices=['action', 'ext_f', 'ext_fpc'])

    return parser


args = get_argparser().parse_args()


def get_exp_name():
    g_start_time = int(datetime.datetime.now().timestamp())
    parser = get_argparser()

    exp_name = ''
    if args.seed is not None:
        exp_name += f'sd{args.seed:03d}_'
    if 'SLURM_JOB_ID' in os.environ:
        exp_name += f's_{os.environ["SLURM_JOB_ID"]}.'
    if 'SLURM_PROCID' in os.environ:
        exp_name += f'{os.environ["SLURM_PROCID"]}.'
    exp_name_prefix = exp_name
    if 'SLURM_RESTART_COUNT' in os.environ:
        exp_name += f'rs_{os.environ["SLURM_RESTART_COUNT"]}.'
    exp_name += f'{g_start_time}'

    exp_name_abbrs = set()
    exp_name_arguments = set()

    def list_to_str(arg_list):
        return str(arg_list).replace(",", "|").replace(" ", "").replace("'", "")

    def add_name(abbr, argument, value_dict=None, max_length=None, log_only_if_changed=False):
        nonlocal exp_name

        if abbr is not None:
            assert abbr not in exp_name_abbrs
            exp_name_abbrs.add(abbr)
        else:
            abbr = ''
        exp_name_arguments.add(argument)

        value = getattr(args, argument)
        if log_only_if_changed and parser.get_default(argument) == value:
            return
        if isinstance(value, list):
            if value_dict is not None:
                value = [value_dict.get(v, v) for v in value]
            value = list_to_str(value)
        elif value_dict is not None:
            value = value_dict.get(value, value)

        if value is None:
            value = 'X'

        if max_length is not None:
            value = str(value)[:max_length]

        if isinstance(value, str):
            value = value.replace('/', '-')

        exp_name += f'_{abbr}{value}'

    add_name(None, 'algo', {
        'ppo_rg': 'PR',
        'a2c_rg': 'AR',
    })
    add_name('e', 'env', {
        'InvertedPendulum-v2': 'Inp',
        'InvertedDoublePendulum-v2': 'Idp',
    }, max_length=3)

    add_name('vn', 'vec_norm')
    add_name('lr', 'lr')
    add_name('dt', 'dt')

    add_name('d', 'max_d')
    add_name('t', 'max_t')
    add_name('ldt', 'lambda_dt')
    if args.anoise_type is not None:
        add_name('ant', 'anoise_type', {
            'action': 'A',
            'ext_f': 'EF',
            'ext_fpc': 'EFC',
        })
        add_name('anp', 'anoise_prob')
        add_name('ans', 'anoise_std')

    return exp_name, exp_name_prefix


def get_log_dir():
    exp_name, exp_name_prefix = get_exp_name()

    log_dir = os.path.realpath(os.path.join(GROUPS_DIR, args.run_group, exp_name))

    return log_dir


def get_env(
        log_dir, wrapper_class=None, wrapper_kwargs=None,
        vec_env_cls=DummyVecEnvEx,
        max_episode_steps=None, frame_skip=None, dt=None, reward_scale=None, gamma=args.ref_gamma,
):
    env = make_vec_env_ex(
        env_id=args.env,
        n_envs=1,
        seed=args.seed or 0,
        monitor_dir=log_dir,
        wrapper_class=wrapper_class,
        wrapper_kwargs=wrapper_kwargs,
        vec_env_cls=vec_env_cls,
        frame_skip=frame_skip,
        dt=dt,
        reward_scale=reward_scale,
        max_episode_steps=max_episode_steps,
    )

    if args.vec_norm:
        env = VecNormalizeEx(env, gamma=gamma)

    return env


def main():
    log_dir = get_log_dir()

    if args.seed is not None:
        set_random_seed(args.seed, True)

    ref_gamma = args.ref_gamma
    dt = args.dt
    frame_skip = args.frame_skip
    vanilla_env = get_env(None)

    ori_action_dim = get_action_dim(vanilla_env.action_space)

    ref_total_dt = vanilla_env.envs[0].dt
    ref_max_episode_steps = vanilla_env.envs[0].spec.max_episode_steps
    if dt is None:
        dt = vanilla_env.envs[0].unwrapped.model.opt.timestep
    if frame_skip is None:
        frame_skip = vanilla_env.envs[0].unwrapped.frame_skip

    total_dt = dt * frame_skip
    gamma = ref_gamma ** (total_dt / ref_total_dt)
    max_episode_steps = int(ref_max_episode_steps * ref_total_dt / total_dt)

    env_info = dict(
        dt=dt,
        frame_skip=frame_skip,
        max_episode_steps=max_episode_steps,
        reward_scale=total_dt / ref_total_dt,
        vec_env_cls=DummyVecEnvEx,
        gamma=gamma,
    )

    wrapper_class = RepeatGoalEnv
    wrapper_kwargs = dict(
        gamma=gamma,
        max_d=args.max_d,
        max_t=args.max_t,
        lambda_dt=args.lambda_dt,
        anoise_type=args.anoise_type,
        anoise_prob=args.anoise_prob,
        anoise_std=args.anoise_std,
    )

    env = get_env(log_dir, wrapper_class, wrapper_kwargs, **env_info)

    callbacks = []
    if args.save_freq > 0:
        save_freq = max(args.save_freq, 1)

        callbacks.append(CheckpointCallbackEx(
            save_freq=save_freq,
            save_path=log_dir,
            name_prefix='rl_model',
            verbose=1,
        ))

    if args.eval_freq > 0:
        eval_freq = args.eval_freq
        eval_env = get_env(None, wrapper_class, wrapper_kwargs, **env_info)

        eval_func = evaluate_policy_rg

        callbacks.append(EvalCallbackEx(
            eval_env,
            best_model_save_path=log_dir,
            n_eval_episodes=args.n_eval_episodes,
            log_path=log_dir,
            eval_freq=eval_freq,
            eval_func=eval_func,
            deterministic=args.deterministic_eval,
        ))

    default_kwargs = dict(
        policy='MlpPolicy',
        env=env,
        verbose=1,
        device=args.device,
        tensorboard_log=log_dir,
        gamma=gamma,
        learning_rate=args.lr,
        seed=args.seed,
    )

    if 'ppo' in args.algo or 'a2c' in args.algo:
        default_kwargs.update(
            vf_coef=0.5,
            policy_kwargs=dict(
                squash_output=False,
                ortho_init=True,
                activation_fn=nn.ReLU,
                net_arch=[dict(pi=[args.model_dim] * args.model_layers, vf=[args.model_dim] * args.model_layers)],
            ),
        )

        if 'ppo' in args.algo:
            default_kwargs.update(
                n_steps=2048,
                n_epochs=10,
                batch_size=64,
                ent_coef=0.,
            )
            default_kwargs['policy_kwargs']['log_std_init'] = 0.
        elif 'a2c' in args.algo:
            default_kwargs.update(
                n_steps=256,
                use_rms_prop=True,
                gae_lambda=1.,
                normalize_advantage=False,
            )
            default_kwargs['policy_kwargs']['log_std_init'] = -1.5

    default_kwargs['policy_kwargs']['mean_type'] = 'full'
    default_kwargs['policy_kwargs']['std_type'] = 'mixed'
    default_kwargs['policy_kwargs']['ori_action_dim'] = ori_action_dim
    default_kwargs['policy_kwargs']['log_std_max'] = 2
    default_kwargs['policy_kwargs']['log_std_min'] = -20
    default_kwargs['policy_kwargs']['mean_init'] = 0.

    learn_kwargs = dict()
    if args.algo == 'ppo_rg':
        model = PPORg(
            **default_kwargs,
        )
        learn_kwargs = dict(
            log_interval=args.log_freq or 1,
        )
    elif args.algo == 'a2c_rg':
        model = A2CRg(
            **default_kwargs,
        )
    else:
        raise Exception('Unknown algorithm')

    model.learn(
        total_timesteps=args.n_steps,
        callback=callbacks,
        **learn_kwargs,
    )


if __name__ == '__main__':
    main()
