import json
import os
import logging
import gymnax
import jax
import numpy as np
import yaml
from os.path import basename

from conf.config import Config, EvoMapConfig, SweepConfig, TrainConfig
from envs.candy import Candy, CandyParams
from envs.pcgrl_env import PROB_CLASSES, PCGRLEnvParams, PCGRLEnv, ProbEnum, RepEnum, get_prob_cls
from envs.play_pcgrl_env import PlayPCGRLEnv, PlayPCGRLEnvParams
from models import ActorCritic, ActorCriticPCGRL, AutoEncoder, ConvForward, ConvForward2, Dense, \
    NCA, SeqNCA
from pcgrllm.scenario_preset import Scenario, ScenarioPreset
from pcgrllm.task import TaskType

log_level = os.getenv('LOG_LEVEL', 'INFO').upper()  # Add the environment variable ;LOG_LEVEL=DEBUG
logger = logging.getLogger(basename(__file__))
logger.setLevel(getattr(logging, log_level, logging.INFO))


def get_exp_dir_evo_map(config: EvoMapConfig):
    exp_dir = os.path.join(
        'saves_evo_map',
        config.problem,
        f'pop-{config.evo_pop_size}_' +
        f'parents-{config.n_parents}_' +
        f'mut-{config.mut_rate}_' +
        f'{config.seed}_{config.exp_name}',
    )
    return exp_dir


def is_default_hiddims(config: Config):
    # Hack, because we're not consistent about when we truncate the hidden dims argument relative to getting the exp_dir
    # path.
    return tuple(config.hidden_dims) == (64, 256)[:len(config.hidden_dims)]


def get_exp_group(config):
    if config.env_name == 'PCGRL':

        # task
        config_dict = {
            'pe': config.pe,
            'it': config.total_iterations,
            'fit': config.evaluator,
            'exp': config.exp_name,
        }

        if config.task != 'alphabet':
            task = config.task[:3]
            config_dict['t'] = task

        # key와 value를 '_'로 구분하여 join
        exp_group = os.path.join(
            '_'.join([f'{key}-{value}' for key, value in config_dict.items()])
        )

        flags_dict = {
            'fewshot': 'fs',
        }
        # Append suffixes for enabled flags


        for flag, suffix in flags_dict.items():
            if getattr(config, flag, False):  # Check if the flag exists and is True
                exp_group += f'_{suffix}'

    elif config.env_name == 'PlayPCGRL':
        exp_group = os.path.join(
            'saves',
            f'play_w-{config.map_width}_' + \
            f'{config.model}-{config.activation}_' + \
            f'vrf-{config.vrf_size}_arf-{config.arf_size}_' + \
            f'{config.exp_name}'
        )
    elif config.env_name == 'Candy':
        exp_group = os.path.join(
            'candy_' + \
            f'{config.exp_name}'
        )
    else:
        exp_group = os.path.join(
            config.env_name
        )
    return exp_group

def get_short_target(target: str) -> str:
    # Split the target string into words
    words = target.split()

    # If there's only one word, return it with the length
    if len(words) == 1:
        return f"{words[0]}_{len(target)}"

    # Otherwise, take the first and last words and include the length
    return f"{words[0]}X{words[-1]}{len(target)}"


def get_exp_name(config):
    exp_group = get_exp_group(config)

    target_character = get_short_target(config.target_character) if config.task == 'scenario' else config.target_character

    if config.feedback_type == "default":
        return f'{exp_group}_chr-{target_character}_s-{config.seed}'
    elif config.feedback_type == "no":
        return f'{exp_group}_chr-{target_character}_fb-{config.feedback_type}_s-{config.seed}'
    else:
        return f'{exp_group}_chr-{target_character}_fb-{config.feedback_type[:3]}_s-{config.seed}'


def get_exp_dir(config):
    return os.path.join('saves', get_exp_name(config))

def init_config(config: Config):
    config.n_gpus = jax.local_device_count()

    # problem
    if config.task == 'scenario' and config.problem != 'dungeon3':
        config.problem = 'dungeon3'
        logger.log(logging.INFO, f"Changing config.problem to dungeon3 for scenario task")

    # Validate if the evaluator is supported
    if config.task == TaskType.Alphabet and config.evaluator not in {'llm', 'hr', 'vit'}:
        raise ValueError(f"Unsupported evaluator for task {config.task}: {config.evaluator}")
    elif config.task == TaskType.Scenario and config.evaluator not in {'hr', 'llm'}:
        raise ValueError(f"Unsupported evaluator for task {config.task}: {config.evaluator}")

    if config.representation in set({'wide', 'nca'}):
        # TODO: Technically, maybe arf/vrf size should affect kernel widths in (we're assuming here) the NCA model?
        config.arf_size = config.vrf_size = config.map_width

    if config.representation == 'nca':
        config.act_shape = (config.map_width, config.map_width)

    else:
        config.arf_size = (2 * config.map_width -
                           1 if config.arf_size == -1 else config.arf_size)

        config.vrf_size = (2 * config.map_width -
                           1 if config.vrf_size == -1 else config.vrf_size)

    if hasattr(config, 'evo_pop_size') and hasattr(config, 'n_envs'):
        assert config.n_envs % (config.evo_pop_size * 2) == 0, "n_envs must be divisible by evo_pop_size * 2"
    if config.model == 'conv2':
        config.arf_size = config.vrf_size = min([config.arf_size, config.vrf_size])

    config.exp_group = get_exp_group(config)
    config.exp_dir = get_exp_dir(config)

    config._vid_dir = os.path.join(config.exp_dir, 'videos')
    config._img_dir = os.path.join(config.exp_dir, 'images')
    config._numpy_dir = os.path.join(config.exp_dir, 'numpy')

    if config.model == 'seqnca':
        config.hidden_dims = config.hidden_dims[:1]

    # if config.pe == 'io', set config.total_iterations < 2 assert!
    if config.pe == 'io':
        if config.total_iterations >= 2:
            print("Total iterations must be less than 2 for IO PE. Did you forget to change the 'pe=' argument?")
            exit(0)

    if config.task == 'scenario':
        try:
            ScenarioPreset().scenarios[str(config.target_character)]
        except:
            print(f"Could not find scenario with condition: {config.target_character}")
            exit(0)




    return config


def init_config_evo_map(config: EvoMapConfig):
    # FIXME: This is meaningless, should remove it eventually.
    config.arf_size = (2 * config.map_width -
                       1 if config.arf_size == -1 else config.arf_size)

    config.vrf_size = (2 * config.map_width -
                       1 if config.vrf_size == -1 else config.vrf_size)

    config.n_gpus = jax.local_device_count()
    config.exp_dir = get_exp_dir_evo_map(config)
    return config


def get_ckpt_dir(config: Config):
    return os.path.join(config.exp_dir, 'ckpts')


def init_network(env: PCGRLEnv, env_params: PCGRLEnvParams, config: Config):
    if config.env_name == 'Candy':
        # In the candy-player environment, action space is flat discrete space over all candy-direction combos.
        action_dim = env.action_space(env_params).n

    elif 'PCGRL' in config.env_name:
        action_dim = env.rep.action_space.n
        # First consider number of possible tiles
        # action_dim = env.action_space(env_params).n
        # action_dim = env.rep.per_tile_action_dim

    else:
        action_dim = env.num_actions

    if config.model == "dense":
        network = Dense(
            action_dim, activation=config.activation,
            arf_size=config.arf_size, vrf_size=config.vrf_size,
        )
    elif config.model == "conv":
        network = ConvForward(
            action_dim=action_dim, activation=config.activation,
            arf_size=config.arf_size, act_shape=config.act_shape,
            vrf_size=config.vrf_size,
            hidden_dims=config.hidden_dims,
        )
    elif config.model == "conv2":
        network = ConvForward2(
            action_dim=action_dim, activation=config.activation,
            act_shape=config.act_shape,
            hidden_dims=config.hidden_dims,
        )
    elif config.model == "seqnca":
        network = SeqNCA(
            action_dim, activation=config.activation,
            arf_size=config.arf_size, act_shape=config.act_shape,
            vrf_size=config.vrf_size,
            hidden_dims=config.hidden_dims,
        )
    elif config.model in {"nca", "autoencoder"}:
        if config.model == "nca":
            network = NCA(
                representation=config.representation,
                tile_action_dim=env.rep.tile_action_dim,
                activation=config.activation,
            )
        elif config.model == "autoencoder":
            network = AutoEncoder(
                representation=config.representation,
                action_dim=action_dim,
                activation=config.activation,
            )
    else:
        raise Exception(f"Unknown model {config.model}")
    # if config.env_name == 'PCGRL':
    if 'PCGRL' in config.env_name:
        network = ActorCriticPCGRL(network, act_shape=config.act_shape,
                                   n_agents=config.n_agents, n_ctrl_metrics=len(config.ctrl_metrics))
    # elif config.env_name == 'PlayPCGRL':
    #     network = ActorCriticPlayPCGRL(network)
    else:
        network = ActorCritic(network)
    return network


def get_env_params_from_config(config: Config):
    map_shape = ((config.map_width, config.map_width) if not config.is_3d
                 else (config.map_width, config.map_width, config.map_width))
    rf_size = max(config.arf_size, config.vrf_size)
    rf_shape = (rf_size, rf_size) if not config.is_3d else (rf_size, rf_size, rf_size)

    act_shape = tuple(config.act_shape)
    if config.is_3d:
        assert len(config.act_shape) == 3

    # Convert strings to enum ints
    problem = ProbEnum[config.problem.upper()]
    prob_cls = PROB_CLASSES[problem]
    ctrl_metrics = tuple([int(prob_cls.metrics_enum[c.upper()]) for c in config.ctrl_metrics])

    env_params = PCGRLEnvParams(
        problem=problem,
        representation=int(RepEnum[config.representation.upper()]),
        map_shape=map_shape,
        rf_shape=rf_shape,
        act_shape=act_shape,
        static_tile_prob=config.static_tile_prob,
        n_freezies=config.n_freezies,
        n_agents=config.n_agents,
        max_board_scans=config.max_board_scans,
        ctrl_metrics=ctrl_metrics,
        change_pct=config.change_pct,
        randomize_map_shape=config.randomize_map_shape,
        empty_start=config.empty_start,
        pinpoints=config.pinpoints,
    )
    return env_params


def get_play_env_params_from_config(config: Config):
    map_shape = (config.map_width, config.map_width)
    rf_size = max(config.arf_size, config.vrf_size)
    rf_shape = (rf_size, rf_size) if not config.is_3d else (rf_size, rf_size, rf_size)

    return PlayPCGRLEnvParams(
        map_shape=map_shape,
        rf_shape=rf_shape,
    )


def gymnax_pcgrl_make(env_name, config: Config, **env_kwargs):
    if env_name in gymnax.registered_envs:
        return gymnax.make(env_name)

    elif env_name == 'PCGRL':
        env_params = get_env_params_from_config(config)
        env = PCGRLEnv(env_params)

    elif env_name == 'PlayPCGRL':
        env_params = get_play_env_params_from_config(config)
        env = PlayPCGRLEnv(env_params)

    elif env_name == 'Candy':
        env_params = CandyParams()
        env = Candy(env_params)

    return env, env_params


def get_sweep_conf_path(cfg: SweepConfig):
    conf_sweeps_dir = os.path.join('conf', 'sweeps')
    # sweep_conf_path_json = os.path.join(conf_sweeps_dir, f'{cfg.name}.json')
    sweep_conf_path_yaml = os.path.join(conf_sweeps_dir, f'{cfg.name}.yaml')
    return sweep_conf_path_yaml


def write_sweep_confs(_hypers: dict, eval_hypers: dict):
    conf_sweeps_dir = os.path.join('conf', 'sweeps')
    os.makedirs(conf_sweeps_dir, exist_ok=True)
    for grid_hypers in _hypers:
        name = grid_hypers['NAME']
        save_grid_hypers = grid_hypers.copy()
        save_grid_hypers['eval_hypers'] = eval_hypers
        with open(os.path.join(conf_sweeps_dir, f'{name}.yaml'), 'w') as f:
            f.write(yaml.dump(save_grid_hypers))
        # with open(os.path.join(conf_sweeps_dir, f'{name}.json'), 'w') as f:
        #     f.write(json.dumps(grid_hypers, indent=4))


def load_sweep_hypers(cfg: SweepConfig):
    sweep_conf_path = get_sweep_conf_path(cfg)
    if os.path.exists(sweep_conf_path):
        hypers = yaml.load(open(sweep_conf_path), Loader=yaml.FullLoader)
        eval_hypers = hypers.pop('eval_hypers')
    else:
        raise FileNotFoundError(f"Could not find sweep config file {sweep_conf_path}")
    return hypers, eval_hypers

