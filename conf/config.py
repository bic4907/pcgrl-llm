from typing import Iterable, List, Optional, Tuple, Union
from hydra.core.config_store import ConfigStore
from dataclasses import dataclass, field


# @dataclass
# class EnvConfig:
#     problem: str = "binary"
#     representation: str = "nca"


@dataclass
class Config:
    lr: float = 1.0e-4
    n_envs: int = 4
    # How many steps do I take in all of my batched environments before doing a gradient update
    num_steps: int = 128
    total_timesteps: int = int(5e7)
    timestep_chunk_size: int = -1
    update_epochs: int = 10
    NUM_MINIBATCHES: int = 4
    GAMMA: float = 0.99
    GAE_LAMBDA: float = 0.95
    CLIP_EPS: float = 0.2
    ENT_COEF: float = 0.01
    VF_COEF: float = 0.5
    MAX_GRAD_NORM: float = 0.5
    activation: str = "relu"
    env_name: str = "PCGRL"
    ANNEAL_LR: bool = False
    DEBUG: bool = True
    exp_name: str = "def"
    seed: int = 0

    problem: str = "binary"
    representation: str = "narrow"
    model: str = "conv"

    map_width: int = 16
    randomize_map_shape: bool = False
    is_3d: bool = False
    # ctrl_metrics: Tuple[str] = ('diameter', 'n_regions')
    ctrl_metrics: Tuple[str] = ()
    # Size of the receptive field to be fed to the action subnetwork.
    vrf_size: Optional[int] = -1  # -1 means 2 * map_width - 1, i.e. full observation, 31 if map_width=16
    # Size of the receptive field to be fed to the value subnetwork.
    arf_size: Optional[int] = -1  # -1 means 2 * map_width - 1, i.e. full observation, 31 if map_width=16
    # TODO: actually take arf and vrf into account in models, where possible

    change_pct: float = -1.0

    # The shape of the (patch of) edit(s) to be made by the edited by the generator at each step.
    act_shape: Tuple[int, int] = (1, 1)

    static_tile_prob: Optional[float] = 0.0
    n_freezies: int = 0
    n_agents: int = 1  # multi-agent is fake and broken
    multiagent: bool = False
    max_board_scans: float = 3.0

    # How many milliseconds to wait between frames of the rendered gifs
    gif_frame_duration: int = 25

    """ DO NOT USE. WILL BE OVERWRITTEN. """
    exp_dir: Optional[str] = None
    n_gpus: int = 1

    # To make the task simpler, always start with an empty map
    empty_start: bool = False

    # In problems with tile-types with specified valid numbers, fix/freeze their random placement at the beginning of 
    # each episode.
    pinpoints: bool = False

    hidden_dims: Tuple[int] = (64, 256)

    # TODO: Implement this. Just a placeholder for now.
    reward_every: int = 1

    # A toggle, will add `n_envs` to the experiment name if we are profiling training FPS, so that we can distinguish 
    # results.
    profile_fps: bool = False

    # NOTE: DO NOT MODIFY THESE. WILL BE SET AUTOMATICALLY AT RUNTIME. ########
    initialize: Optional[bool] = None

    # Wandb
    wandb_key: Optional[str] = None
    wandb_project: Optional[str] = 'pcgrl-llm'
    wandb_resume: str = 'allow'
    evaluator: str = 'vit' # 'vit', 'hr' (heuristic)

    exp_group: Optional[str] = None

    _vid_dir: Optional[str] = None
    _img_dir: Optional[str] = None
    _numpy_dir: Optional[str] = None
    current_iteration: int = 0


    ###########################################################################
    # Note: BORDER, EMPTY, WALL is mandatory for all problems, so don't include them in below lists.
    available_tiles: List[int] = field(default_factory=lambda: [])
    task: str = "alphabet"

    randomize_start_pos: bool = False
    use_preset_level: bool = True


@dataclass
class EvoMapConfig(Config):
    n_generations: int = 100_000
    evo_pop_size: int = 100
    n_parents: int = 50
    mut_rate: float = 0.3
    render_freq: int = 10_000
    log_freq: int = 1_000
    callbacks: bool = True


@dataclass
class TrainConfig(Config):
    overwrite: bool = False

    # Save a checkpoint after (at least) this many timesteps
    ckpt_freq: int = int(1e7)

    # Render after this many update steps
    render_freq: int = 5
    n_render_eps: int = 3

    # eval the model on pre-made eval freezie maps to see how it's doing
    eval_freq: int = 100
    n_eval_maps: int = 6
    eval_map_path: str = "user_defined_freezies/binary_eval_maps.json"
    # discount factor for regret value calculation is the same as GAMMA

    # NOTE: DO NOT MODIFY THESE. WILL BE SET AUTOMATICALLY AT RUNTIME. ########
    NUM_UPDATES: Optional[int] = None
    MINIBATCH_SIZE: Optional[int] = None
    ###########################################################################

    reward_function_path: Optional[str] = None

    agents: int = 1
    target_character: str = 'A'
    pe: str = 'io' # 'zs', 'cot', 'cotsc', 'tot', 'got'
    total_iterations: int = 1
    n_self_alignment: int = 0


@dataclass

class DebugConfig(Config):
    overwrite: bool = True

    # Save a checkpoint after (at least) this many timesteps
    ckpt_freq: int = int(1)
    # Render after this many update steps
    render_freq: int = 1000
    n_render_eps: int = 3

    # eval the model on pre-made eval freezie maps to see how it's doing
    eval_freq: int = 1
    n_eval_maps: int = 6
    eval_map_path: str = "user_defined_freezies/binary_eval_maps.json"
    # discount factor for regret value calculation is the same as GAMMA

    # NOTE: DO NOT MODIFY THESE. WILL BE SET AUTOMATICALLY AT RUNTIME. ########
    NUM_UPDATES: Optional[int] = None
    MINIBATCH_SIZE: Optional[int] = None
    ###########################################################################

    total_timesteps: int = int(1e6)
    log_freq: int = 1

class MultiAgentConfig(TrainConfig):
    multiagent: bool = True
    # lr: float = 3e-4
    # update_epochs: int = 4
    # num_steps: int = 521
    # gamma: float = 0.99
    # gae_lambda: float = 0.95
    # clip_eps: float = 0.2
    # scale_clip_eps: bool = False
    # ent_coef: float = 0.0
    # vf_coef: float = 0.5
    # max_grad_norm: float = 0.25

    model: str = 'rnn'
    representation: str = "turtle"
    n_agents: int = 2
    n_envs: int = 300
    scale_clip_eps: bool = False
    hidden_dims: Tuple[int] = (512, -1)
    empty_start: bool = True

    # Save a checkpoint after (at least) this many ***update*** steps
    ckpt_freq: int = 40
    render_freq: int = 20

    # WandB Params
    WANDB_MODE: str = 'run'  # one of: 'offline', 'run', 'dryrun', 'shared', 'disabled', 'online'
    ENTITY: str = ''
    PROJECT: str = 'smearle_pcgrl_mappo'

    # NOTE: DO NOT MODIFY THESE. WILL BE SET AUTOMATICALLY AT RUNTIME. ########
    _num_actors: int = -1
    _minibatch_size: int = -1
    _num_updates: int = -1
    _exp_dir: Optional[str] = None
    _ckpt_dir: Optional[str] = None
    _vid_dir: Optional[str] = None
    _numpy_dir: Optional[str] = None
    ###########################################################################

@dataclass
class TrainAccelConfig(TrainConfig):
    evo_freq: int = 10
    evo_pop_size: int = 10
    evo_mutate_prob: float = 0.1


@dataclass
class EvalConfig(TrainConfig):
    reevaluate: bool = True
    random_agent: bool = False
    # In how many bins to divide up each metric being evaluated
    n_bins: int = 10
    n_eval_envs: int = 10
    n_eps: int = 5
    eval_map_width: Optional[int] = None
    eval_max_board_scans: Optional[int] = None
    eval_randomize_map_shape: Optional[bool] = None
    eval_seed: int = 0

    # Which eval metric to keep in our generated table if sweeping over eval hyperparams (in which case we want to 
    # save space). Only applied when runnins `cross_eval.py`
    # metrics_to_keep = [
    #     # 'min_min_loss',
    #     'mean_ep_reward',
    # ]
    metrics_to_keep: Tuple[str] = ('mean_ep_reward',)


@dataclass
class EnjoyConfig(EvalConfig):
    random_agent: bool = False
    # How many episodes to render as gifs
    n_eps: int = 5
    eval_map_width: Optional[int] = None
    render_stats: bool = True
    n_enjoy_envs: int = 1
    render_ims: bool = False


@dataclass
class EnjoyMultiAgentConfig(MultiAgentConfig, EnjoyConfig):
    pass
    

@dataclass
class ProfileEnvConfig(Config):
    N_PROFILE_STEPS: int = 5000
    reevaluate: bool = False


@dataclass
class SweepConfig(EnjoyConfig, EvalConfig):
    name: Optional[str] = None
    mode: str = 'train'
    slurm: bool = True


@dataclass
class TrainLLMConfig(Config):
    overwrite: bool = False

    target_character: Optional[str] = "A"
    # Save a checkpoint after (at least) this many timesteps
    ckpt_freq: int = int(5e6)
    # Render after this many update steps
    total_timesteps: int = int(5e7)

    render_freq: int = 40
    n_render_eps: int = 3

    # eval the model on pre-made eval freezie maps to see how it's doing
    eval_freq: int = 100
    n_eval_maps: int = 6
    eval_map_path: str = "user_defined_freezies/binary_eval_maps.json"
    # discount factor for regret value calculation is the same as GAMMA

    # NOTE: DO NOT MODIFY THESE. WILL BE SET AUTOMATICALLY AT RUNTIME. ########
    NUM_UPDATES: Optional[int] = None
    MINIBATCH_SIZE: Optional[int] = None
    ###########################################################################

    # LLM experiment setting
    total_iterations: int = int(1)
    n_self_alignment: int = int(0)

    # Eval rollout setting
    random_agent: bool = False
    eval_map_width: Optional[int] = 16
    eval_max_board_scans: Optional[int] = 3
    eval_randomize_map_shape: Optional[bool] = False
    eval_seed: int = 0
    n_eval_envs: int = 1
    reevaluate: bool = True
    n_eps: int = 2

    # validation setting
    reward_function_path: Optional[str] = None


    # reward generation setting
    n_generation_trials: int = 5

    bypass_reward_path: Optional[str] = None
    bypass_train_path: Optional[str] = None
    bypass_feedback_path: Optional[str] = None
    random_fitness: bool = False

    gpt_model: str = 'gpt-4o'
    feedback_input_type: str = 'array' # 'array' or 'image', image requires vision-language model
    pe: str = 'io' # 'zs', 'cot', 'cotsc', 'tot', 'got'
    evaluator: str = 'vit' # 'vit', 'hr' (heuristic)
    n_samples: int = 30
    reward_feature: str = 'array' # 'array', 'stats', 'array+stats'
    fewshot: bool = False
    feedback_type: str = 'default' # 'default', 'no', 'generic'

    branch_factor: Optional[int] = 2

    n_codegen_trials: int = 3
    n_codefix_trials: int = 3

cs = ConfigStore.instance()
cs.store(name="config", node=Config)
cs.store(name="ma_config", node=MultiAgentConfig)
cs.store(name="enjoy_ma_pcgrl", node=EnjoyMultiAgentConfig)
cs.store(name="evo_map_pcgrl", node=EvoMapConfig)
cs.store(name="train_pcgrl", node=TrainConfig)
cs.store(name="debug_pcgrl", node=DebugConfig)
cs.store(name="train_accel_pcgrl", node=TrainAccelConfig)
cs.store(name="enjoy_pcgrl", node=EnjoyConfig)
cs.store(name="eval_pcgrl", node=EvalConfig)
cs.store(name="profile_pcgrl", node=ProfileEnvConfig)
cs.store(name="batch_pcgrl", node=SweepConfig)

# PCGRLLM Configs
cs.store(name="train_pcgrllm", node=TrainLLMConfig)
