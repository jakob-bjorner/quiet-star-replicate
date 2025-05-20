from functools import partial
import os
from dotenv import load_dotenv
load_dotenv()
import contextlib
import pprint
import logging

from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from hydra_plugins.hydra_submitit_launcher.config import SlurmQueueConf
from omegaconf import OmegaConf, MISSING
from hydra import main as hydra_main
from typing import Dict, List, Any
import os
import wandb
cs = ConfigStore.instance()
@dataclass
class CustomKargoLauncherConfig(SlurmQueueConf): 
    """ https://hydra.cc/docs/1.3/plugins/submitit_launcher/ then go to github and look at config.py this is what I extend.
        to run things locally, use the option on launch `python run.py hydra/launcher=submitit_local`, 
        or in this case, without -m it launches things locally.
    """
    # submitit_folder: str = 
    # the default submitit_folder = "${hydra.sweep.dir}/.submitit/%j"
    # so reasonable and can't make it anything more reasonable it seems, because 
    # they launch with map_executor on the backend, which is the best for my 
    # parallel jobs, but prevents nicely putting the submitit loggs into more 
    # careful foldering. Perhaps in the future I can follow a per experiment 
    # foldering, and the specificity of the sweep.dir folder will be more useful 
    # to me.
    timeout_min: int = 2880 # 60 * 24 * 2
    # cpus_per_task: int|None = 6 # type: ignore
    cpus_per_gpu: int|None = 6
    # gpus_per_node: int|None = None
    # gpus_per_task: str = 1
    tasks_per_node: int =  1
    mem_gb: int|None =  None
    nodes: int = 1
    _target_: str = "hydra_plugins.hydra_submitit_launcher.submitit_launcher.SlurmLauncher"
    partition: str|None = "overcap" # kargo-lab
    qos: str|None = "short"
    exclude: str|None = "major,crushinator,nestor,voltron,megabot,samantha,uniblab,gundam,consu,puma,protocol"
    use_srun: bool = False
    additional_parameters: Dict[str, Any] = field(default_factory=lambda: {"gpus-per-task": "a40:1"})
                                                                        #    "requeue": True
    array_parallelism: int = 40
cs.store(name="custom_kargo_submitit", node=CustomKargoLauncherConfig, group="hydra/launcher")



@dataclass
class RunConfig:
    defaults: List[Any] = field(default_factory=lambda: [
        {"override hydra/launcher": os.getenv("HYDRA_LAUNCHER", "custom_kargo_submitit")},
        # {"override hydra/sweeper": "optuna"}, # https://hydra.cc/docs/plugins/optuna_sweeper/
        # {"override hydra/sweeper/sampler": "random"}, 
        "_self_",
        ])
    run_type: str = MISSING
    model_type: str = MISSING
    training_type: str = MISSING
    node_name: str = MISSING
    device: str = os.getenv('device', "cuda")
    experiment_logger: str = "wandb"
    info: str = ""
    seed: int = 7
    output_dir: str = "${hydra:runtime.output_dir}"
    seq_len: int = 128
    data_loader_batch_size: int = 128
    data_loader_num_workers: int = 0
    model_hidden_dim: int = 100
    nll_loss_beta: float = 1.0
    dataset: str = 'shake' # 'fw' #
    max_steps: int = -1
    base_lr: float = 0.001
    use_scheduler: bool = False
    linear_lm_head: bool = True
    mix_interpeter_base_lm: int = 0
    clip_gradients: float = 0.0
    entropy_encouragement_coef: float=0.0
    parameter_groups: int = 0
    neuter_dropout_base: float = 0.0
    infer_pretrained_base: Any = False
    infer_pretrained_reasoner: Any = False
    infer_pretrained_base_reasoner: Any = False
    
    reasoner_interpreter_vocab_size: int = MISSING
    base_lm_hidden_dim: int = MISSING
    reasoner_hidden_dim: int = MISSING
    interpreter_hidden_dim: int = "${reasoner_hidden_dim}" # type: ignore
    use_reasoner: bool = MISSING
    use_base_lm: bool = MISSING
    simple_lm_head: bool = MISSING
    stage_wise_downproject_interpreter_rep: bool = False
    weight_groups: list|None = None
    share_lm_head: bool = MISSING
    variable_len: bool = False
    punish_unfinished: float = 0.0
    length_penalty: float = 0.0
    schedule_punfinished: bool = True
    tokenizer_version: str = "v2" # update from not having a version attached, which is what I did for fineweb. This for unk

    model_n_layer: int = MISSING
    model_n_head: int = MISSING
    use_residual: bool = MISSING
    add_last_context_token: bool = MISSING
    add_surogate_loss_to_last_context_token: bool = MISSING
    different_eot: bool = MISSING
    last_context_loss_beta: float = MISSING
    increment_pos_id_for_last_context_token: bool = MISSING
    train_nll_num: int = "${trice_samples}" # type: ignore

    dropout: float = MISSING

    debug_cfg: str = ""

    policy_loss_beta: float = 1
    trice_samples: int = 10
    n_tokens_ahead: int = MISSING
    max_reasoning_len: int = 10
    auto_find_grad_acc: bool = False
    grad_acc: int = 1

    epochs: int = 50

    # known issue: the multirun.yaml is saved to the sweep dir, and not the subdirs, so it is not saved! 
    # (don't think I will need this to be saved tho, and makes folders easier to read)
    hydra: Any = field(default_factory=lambda: {
        "job":{"config":{"override_dirname":{"item_sep": "_"}}},
        "sweep":{"dir": "quiet_star_replicate_runs", 
                 "subdir": "${hydra.job.override_dirname}_${now:%Y-%m-%d}/${now:%H-%M-%S}"},
        "run":{"dir":  "quiet_star_replicate_runs/${hydra.job.override_dirname}_${now:%Y-%m-%d}/${now:%H-%M-%S}"},
    })
cs.store(name="RunConfig", node=RunConfig)

QuietSTaRGRURunConfig = {
                        # "defaults":  [{"override /dataset@trainer.datasetloaders.train_dataset": "MultilingualUnsupervisedAwesomeAlignDatasetTraining"},
                        #             {"override /datasetmap@trainer.datasetloaders.val_datasets": "nozhSupervisedAwesomeAlignDatasetsMapEval"}],
                       "model_type": "rgruhtilda",
                       "training_type": "quiet_starM",
                       "run_type": "train"}
cs.store(name="QuietSTaRGRURunConfig", node=QuietSTaRGRURunConfig, group="run_modifier", package="_global_")

ReasonerInterpreterGRURunConfig = {
                       "model_type": "reintG",
                    #    "reasoner_interpreter_vocab_size": 0, # this is the default it is the same as the char level llm for shakespeare + 1 (this for the start/end of thought token).
                       "base_lm_hidden_dim": 100,
                       "reasoner_hidden_dim": 100,
                    #    "interpreter_hidden_dim": 100,
                       "use_base_lm": True,
                       "use_reasoner": True,
                       "n_tokens_ahead": 1,
                       "training_type": "quiet_starM",
                       "run_type": "train",
                       "simple_lm_head": False,
                       "weight_groups": ['A', 'B', 'C', 'D'],
                       "share_lm_head": False,
                       "auto_find_grad_acc": True}
cs.store(name="ReasonerInterpreterGRURunConfig", node=ReasonerInterpreterGRURunConfig, group="run_modifier", package="_global_")

InterpreterLowVarGRURunConfig = {
                        "defaults":  [{"/run_modifier": ["ReasonerInterpreterGRURunConfig"]}], # must be a list for some reason
                        # "data_loader_batch_size": 8,
                        "use_base_lm": False,
                        "trice_samples": 100,
}
cs.store(name="InterpreterLowVarGRURunConfig", node=InterpreterLowVarGRURunConfig, group="run_modifier", package="_global_")

InterpreterGRURunConfig = {
                       "defaults":  [{"/run_modifier": ["ReasonerInterpreterGRURunConfig"]}], # must be a list for some reason
                       "use_reasoner": False,
                       "training_type": 'nll',}
cs.store(name="InterpreterGRURunConfig", node=InterpreterGRURunConfig, group="run_modifier", package="_global_")


QuietSTaRDiscreteRunConfig = {
                       "model_type": "quietdiscreteG",
                       "model_hidden_dim": 100,
                       "use_reasoner": True,
                       "use_base_lm": True,
                       "n_tokens_ahead": 1,
                       "max_reasoning_len": 10,
                       "training_type": "quiet_starM",
                       "run_type": "train",
                       "auto_find_grad_acc": True,
}
cs.store(name="QuietSTaRDiscreteRunConfig", node=QuietSTaRDiscreteRunConfig, group="run_modifier", package="_global_")

QuietSTaRDiscreteTRunConfig = {
                       "model_type": "quietdiscreteT",
                       "model_hidden_dim": 100,
                       "model_n_layer": 2,
                       "model_n_head": 5,
                       "n_tokens_ahead": 1,
                       "max_reasoning_len": 10,
                       "use_reasoner": True,
                       "use_residual": False,
                       "training_type": "quiet_starO",
                       "run_type": "train",
                       "auto_find_grad_acc": True,
                       "add_last_context_token": True,
                       "add_surogate_loss_to_last_context_token": False,
                       "different_eot": True,
                       "last_context_loss_beta": 1.0,
                       "increment_pos_id_for_last_context_token": False,
}
cs.store(name="QuietSTaRDiscreteTRunConfig", node=QuietSTaRDiscreteTRunConfig, group="run_modifier", package="_global_")

TLMRunConfig = {
    "model_type": "T",
    "model_hidden_dim": 100,
    "model_n_layer": 2,
    "model_n_head": 5,
    "dropout": 0.0,
    "training_type": "nll",
    "run_type": "train",
    "auto_find_grad_acc": True,
}
cs.store(name="TLMRunConfig", node=TLMRunConfig, group="run_modifier", package="_global_")

SameHdim = { # this was to make the file names shorter lol
    "base_lm_hidden_dim": MISSING,
    "reasoner_hidden_dim": "${base_lm_hidden_dim}",
    "interpreter_hidden_dim": "${base_lm_hidden_dim}",

}
cs.store(name="SameHdim", node=SameHdim, group="run_modifier", package="_global_")

GLMRunConfig = {
    "model_type": 'G',
    "training_type": "nll",
    "base_lm_hidden_dim": MISSING,
    "model_n_layer": 1,
    "run_type": "train",
    "auto_find_grad_acc": True,
}
cs.store(name="GLMRunConfig", node=GLMRunConfig, group="run_modifier", package="_global_")

GRUSeparateScaleExperiment = {
    "defaults":  [{"/run_modifier": ["SameHdim"]}],
    "info": "[GRU separate scale]",
    "dataset": "fw",
    "trice_samples": 2,
    "max_steps": 2700,
    "simple_lm_head": True,
    "share_lm_head": True,
    "use_scheduler": True,
    "base_lr": 0.005,
}
cs.store(name="GRUSeparateScaleExperiment", node=GRUSeparateScaleExperiment, group="run_modifier", package="_global_")

GRUShakeSeparateScaleExperiment = dict(
    defaults=[{"/run_modifier": ["ReasonerInterpreterGRURunConfig"]}],
    info="[debug nll 1.6 1.8 discrep v2]", # just a name they already had. whatever easier to find the other experiments this way.
    simple_lm_head=True,
    share_lm_head=True,
    linear_lm_head=False,
    base_lm_hidden_dim=32,
    reasoner_hidden_dim=256,
    interpreter_hidden_dim="${reasoner_hidden_dim}",
    # seed=1,2,3,4,5,6,7,8,9,10,
    # policy_loss_beta=0,1,
)
cs.store(name="GRUShakeSeparateScaleExperiment", node=GRUShakeSeparateScaleExperiment, group="run_modifier", package="_global_")

GRUVarLenScaleExperiment = dict(
    defaults=[{"/run_modifier": ["ReasonerInterpreterGRURunConfig"]}],
    info="[varlen gru v1]", # for easy filtering
    base_lm_hidden_dim=32,
    reasoner_hidden_dim=256,
    simple_lm_head=True, 
    share_lm_head=False, # main difference from above experiment
    linear_lm_head=False,
    interpreter_hidden_dim="${reasoner_hidden_dim}",
    use_scheduler=True,
    variable_len=True,
    base_lr=0.005,
    mix_interpeter_base_lm=2, 
    entropy_encouragement_coef=0.0,
    trice_samples=10,
    policy_loss_beta=1,
    clip_gradients=1.0,
    dataset='fw',
    max_steps=2700
    # trice_samples=2, # this is mainly a speedup, and shouldn't impact variance of gradient estimator too much is the hope for short sequences ie len 10
    # did seem to matter actually.
)
cs.store(name="GRUVarLenScaleExperiment", node=GRUVarLenScaleExperiment, group="run_modifier", package="_global_")

GRUBaseInterpreterScaleExperiment = dict(
    defaults=[{"/run_modifier": ["ReasonerInterpreterGRURunConfig"]}],
    info="[gru biscale v0]", # for easy filtering
    base_lm_hidden_dim=32,
    reasoner_hidden_dim="${base_lm_hidden_dim}",
    simple_lm_head=True, 
    share_lm_head=False, # main difference from above experiment
    linear_lm_head=False,
    interpreter_hidden_dim="${base_lm_hidden_dim}",
    use_scheduler=True,
    variable_len=True,
    base_lr=0.005,
    mix_interpeter_base_lm=2, 
    entropy_encouragement_coef=0.0,
    trice_samples=10,
    policy_loss_beta=1,
    clip_gradients=1.0,
    dataset='fw',
    max_steps=2700,
    max_reasoning_len=10,
    # trice_samples=2, # this is mainly a speedup, and shouldn't impact variance of gradient estimator too much is the hope for short sequences ie len 10
    # did seem to matter actually.
)
cs.store(name="GRUBaseInterpreterScaleExperiment", node=GRUBaseInterpreterScaleExperiment, group="run_modifier", package="_global_")

GRUConcatExperiment = dict(
    defaults=[{"/run_modifier": ["ReasonerInterpreterGRURunConfig"]}],
    info="[gru concat v0]", # for easy filtering
    base_lm_hidden_dim=32,
    reasoner_hidden_dim=32,
    simple_lm_head=False, # now simple head is false meaning we concat instead of element wise add interpreter repr and base lm.
    share_lm_head=False, 
    linear_lm_head=False,
    interpreter_hidden_dim="${reasoner_hidden_dim}",
    use_scheduler=True,
    variable_len=True,
    base_lr=0.005,
    mix_interpeter_base_lm=0, 
    entropy_encouragement_coef=0.0,
    trice_samples=10,
    policy_loss_beta=1,
    clip_gradients=1.0,
    dataset='fw',
    max_steps=2700
    # trice_samples=2, # this is mainly a speedup, and shouldn't impact variance of gradient estimator too much is the hope for short sequences ie len 10
    # did seem to matter actually.
)
cs.store(name="GRUConcatExperiment", node=GRUConcatExperiment, group="run_modifier", package="_global_")

GRUPretrainReasonerExperiment = dict(
    defaults=[{"/run_modifier": ["GLMRunConfig"]}],
    info="[gru pretrain v0]", # for easy filtering
    simple_lm_head=True, # now simple head is false meaning we concat instead of element wise add interpreter repr and base lm.
    linear_lm_head=True, # this because the reasoner only has a simple LM head as well, so need to load this in.
    use_scheduler=True,
    base_lr=0.005,
    clip_gradients=1.0,
    dataset='fw',
    max_steps=2700
)
cs.store(name="GRUPretrainReasonerExperiment", node=GRUPretrainReasonerExperiment, group="run_modifier", package="_global_")



DebugRunConfig = {
                        # "defaults": [{"/run_modifier": ["QuietSTaRDiscreteRunConfig"]}],
                        # "defaults": [{"/run_modifier": ["GRUShakeSeparateScaleExperiment"]}],
                        # "defaults": [{"/run_modifier": ["InterpreterGRURunConfig"]}],
                        # "defaults": [{"/run_modifier": ["QuietSTaRDiscreteTRunConfig"]}],
                        # "defaults": [{"/run_modifier": ["TLMRunConfig"]}],
                        # "defaults": [{"/run_modifier": ["GRUPretrainReasonerExperiment"]}],
                        "defaults": [{"/run_modifier": ["GRUConcatExperiment"]}],
                        "base_lm_hidden_dim": 32,
                        "max_reasoning_len": 10,
                        "reasoner_hidden_dim": 256,
                        # "infer_pretrained_reasoner": True,
                        "infer_pretrained_base_reasoner": True,
                        # "dataset": 'fw',
                        # "max_steps": 2700,
                        # "neuter_dropout_base": 0.875,
                        # "data_loader_batch_size": 128,
                        # "infer_pretrained_base": True, # either True or a checkpoint pointing to the path. if true, I'll implement it later...
                        # "parameter_groups": 1,
                        # "stage_wise_downproject_interpreter_rep": True,
                        # "entropy_encouragement_coef": 0.0005,
                        # "use_reasoner": False,
                        # "variable_len": True,
                        # "punish_unfinished": 0.0,
                        # "length_penalty": 0.0004,
                        # "share_lm_head": False,
                        # "use_scheduler": True,
                        # "base_lr": 0.005,
                        # "mix_interpeter_base_lm": 2, 
                        # 'use_reasoner': False,
                        # "model_hidden_dim": 32,
                        # "model_n_layer": 2,
                        # "model_n_head": 1,
                        # "trice_samples": 10,

                        # "policy_loss_beta": 0, # must be together with use_reasoner = false due to numerical issues with the reward baseline being not exactly zero.
                        # "use_residual": True,

                        # "add_last_context_token": True,
                        # "add_surogate_loss_to_last_context_token": True,
                        # "increment_pos_id_for_last_context_token": True,
                        "seed": 1,
                        # "simple_lm_head": True,
                        # 'device': 'cpu',
                        # "use_base_lm": False,
                        # "info": "seperateInterpreterDEBUG_",
                        # "debug_cfg": "seperateInterpreter",
                        # "auto_find_grad_acc": True,
                        "experiment_logger": 'offlinelogger',
                }
cs.store(name="DebugRunConfig", node=DebugRunConfig, group="run_modifier", package="_global_")

def get_run_name_from_cfg(cfg: RunConfig):
    run_name = f"{cfg.info}model={cfg.model_type}"
    if cfg.run_type == "train":
        run_name += f"_train={cfg.training_type}_ds={cfg.dataset}_bz={cfg.data_loader_batch_size}_gclip={cfg.clip_gradients}"
        if cfg.nll_loss_beta != 1.0:
            run_name += f"_nlb={cfg.nll_loss_beta}"
        if cfg.base_lr != 0.001:
            run_name += f"_lr={cfg.base_lr}"
        if cfg.use_scheduler:
            run_name += f"_sched={cfg.use_scheduler}"
        
    if cfg.model_type == "reintG":
        # removed this, for ease of experimentation: _rivocab={cfg.reasoner_interpreter_vocab_size}
        run_name += f"_trice={cfg.trice_samples}_blmhdim={cfg.base_lm_hidden_dim}_rhdim={cfg.reasoner_hidden_dim}_ihdim={cfg.interpreter_hidden_dim}_maxr={cfg.max_reasoning_len}_linhead={cfg.linear_lm_head}_mix={cfg.mix_interpeter_base_lm}"
        if cfg.use_reasoner == False:
            run_name += f"_use_r={cfg.use_reasoner}"
        if cfg.use_base_lm == False:
            run_name += f"_use_lm={cfg.use_base_lm}"
        if cfg.simple_lm_head:
            run_name += f"_simplehead"
        if cfg.weight_groups:
            run_name += f"_weightg={''.join(cfg.weight_groups)}"
        if cfg.share_lm_head:
            run_name += f"_sharehead"
        if cfg.variable_len:
            run_name += f"_varlen=True_pununfin={cfg.punish_unfinished}_sched={cfg.schedule_punfinished}_lenpen={cfg.length_penalty}"
        if cfg.entropy_encouragement_coef != 0.0:
            run_name += f"_ent={cfg.entropy_encouragement_coef}"
    if cfg.model_type == "quietdiscreteG":
        if cfg.linear_lm_head == False:
            raise NotImplementedError("Need to implement this for discreteG")
        run_name += f"_hdim={cfg.model_hidden_dim}_maxr={cfg.max_reasoning_len}"
        if cfg.use_reasoner == False:
            run_name += f"_use_r={cfg.use_reasoner}"
        if cfg.use_base_lm == False:
            run_name += f"_use_lm={cfg.use_base_lm}"
    if cfg.model_type == "quietdiscreteT":
        run_name += f"_hdim={cfg.model_hidden_dim}_depth={cfg.model_n_layer}_heads={cfg.model_n_head}_maxr={cfg.max_reasoning_len}"
        if cfg.use_reasoner == False:
            run_name += f"_use_r={cfg.use_reasoner}"
        if cfg.use_residual:
            run_name += f"_use_res={cfg.use_residual}"
        if cfg.add_last_context_token:
            run_name += f"_add_last"
            if cfg.increment_pos_id_for_last_context_token:
                run_name += f"_incposid"
            if cfg.add_surogate_loss_to_last_context_token:
                run_name += f"_surogate"
                if cfg.last_context_loss_beta != 1.0:
                    run_name += f"_lclb={cfg.last_context_loss_beta}"
        if cfg.different_eot:
            run_name += f"_dif_eot"
    if cfg.model_type == "T":
        run_name += f"_hdim={cfg.model_hidden_dim}_depth={cfg.model_n_layer}_heads={cfg.model_n_head}_drop={cfg.dropout}"

    if cfg.model_type == "G":
        run_name += f"_hdim={cfg.base_lm_hidden_dim}_depth={cfg.model_n_layer}_linhead={cfg.linear_lm_head}"
        

    if "quiet_star" in cfg.training_type:
        run_name += f"_trice={cfg.trice_samples}_nahead={cfg.n_tokens_ahead}"
        if cfg.train_nll_num != cfg.trice_samples:
            run_name += f"_num_nll={cfg.train_nll_num}"
    run_name += f"_seed={cfg.seed}"
        
    return run_name


@hydra_main(version_base=None, config_name='RunConfig')
def main(cfg: RunConfig):
    # make the logger here instead of at top level due to hydra auto complete.
    format_str = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
    logging.basicConfig(encoding="utf-8", level=logging.DEBUG, format=format_str)
    logger = logging.getLogger(__name__)
    cfg.node_name = os.getenv("SLURMD_NODENAME", "NO_NODE_NAME_FOUND")
    run_name = get_run_name_from_cfg(cfg)
    cfg.output_dir
    cfg_for_logging = OmegaConf.to_container(cfg)
    device = cfg.device
    data_loader_batch_size = cfg.data_loader_batch_size
    data_loader_num_workers = cfg.data_loader_num_workers
    seq_len = cfg.seq_len
    training_type = cfg.training_type # "quiet_star"
    model_type = cfg.model_type # 'rgruhtilda'
    model_hidden_dim = cfg.model_hidden_dim
    epochs = cfg.epochs

    import torch
    import random
    import numpy as np


    def set_seed(seed):
        # Set the seed for Python's built-in random module
        random.seed(seed)

        # Set the seed for NumPy's random module
        np.random.seed(seed)

        # Set the seed for PyTorch's CPU random number generator
        torch.manual_seed(seed)

        # Set the seed for PyTorch's GPU random number generator (if available)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # Ensure deterministic behavior for CuDNN (if available)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Set the environment variable for Python hash seed
        os.environ["PYTHONHASHSEED"] = str(seed)
    set_seed(cfg.seed)

    logger.info(f"run_name = {run_name}")
    logger.info(f"config: {pprint.pformat(cfg_for_logging)}")
    if cfg.experiment_logger == "wandb":
        experiment_logger = wandb
    elif cfg.experiment_logger == "offlinelogger":
        class OfflineLogger:
            def __init__(self,):
                ...
            def init(self, *args, **kwargs):
                print(args, kwargs)
                return contextlib.nullcontext()
            def log(self, *args, **kwargs):
                # print(args, kwargs) # we already default to printing, so this would be redundant.
                ...
        experiment_logger = OfflineLogger()
    else:
        raise Exception(f"invalid experiment_logger {cfg.experiment_logger} should be either wandb or offlinelogger")
    with experiment_logger.init(project="quiet_star_replicate", name=run_name,  config=cfg_for_logging): # type: ignore
        # import here for lazy loading so hydra command line autocomplete remains fast.
        from quiet_star_replicate.trainer.trainer import LatentRationaleTrainer, Trainer
        from quiet_star_replicate.model.model import LanguageModelLSTM, get_nll, QuietStarLanguageModelrGRU, get_quiet_star_loss_reason_multiple, get_quiet_star_loss_reason_once, ReasonerInterpreterGRULM, QuietStarDiscreteGRULM, QuietStarDiscreteTransformerLM, CustomGPT2LMHeadModel, GPT2Config, LanguageModelLSTM
        from quiet_star_replicate.data.data import CustomTokenizer, get_train_val_test_datasets, get_shakespeare_collate_fn, FWDataset
        from torch.utils.data import Dataset, DataLoader, random_split

        if cfg.dataset == 'shake':
            data_str = open(os.path.join(os.environ.get("data_file_path", ""),'./tiny_shakespeare.txt'), 'r').read()
            tokenizer = CustomTokenizer(data_str)
            train_ds, val_ds, test_ds = get_train_val_test_datasets(data_str, seq_len=seq_len)
            collate_fn = get_shakespeare_collate_fn(tokenizer)
        elif cfg.dataset == 'fw':
            dataset = FWDataset.load(os.environ.get("data_file_path", ""), seq_len=seq_len)
            # data_str = ''.join(dataset.data)
            tokenizer = CustomTokenizer.load(os.environ.get("data_file_path", ""), cfg.dataset + cfg.tokenizer_version)
            train_ds, val_ds, test_ds = random_split(dataset, [0.8, 0.0001, 0.2 - 0.0001], generator=torch.Generator().manual_seed(42))
            collate_fn = get_shakespeare_collate_fn(tokenizer)
        else:
            raise Exception(f"{cfg.dataset=} is not defined.")



        train_dl = DataLoader(train_ds, batch_size=data_loader_batch_size, collate_fn=collate_fn, shuffle=True if cfg.max_steps != -2 else False, num_workers=data_loader_num_workers)
        val_dl = DataLoader(val_ds, batch_size=data_loader_batch_size, collate_fn=collate_fn, shuffle=False, num_workers=data_loader_num_workers)

        
        # q_star_lm = QuietStarLanguageModelrGRU(len(tokenizer.vocab), model_hidden_dim, 1, model_type).to(device)
        if cfg.model_type == "quietdiscreteG":
            model = QuietStarDiscreteGRULM(len(tokenizer.vocab)+1, cfg.model_hidden_dim, cfg.use_base_lm, cfg.use_reasoner, cfg.max_reasoning_len, len(tokenizer.vocab), cfg.debug_cfg)
        elif cfg.model_type == "quietdiscreteT":
            if cfg.different_eot:
                model = QuietStarDiscreteTransformerLM(len(tokenizer.vocab)+2, cfg.model_hidden_dim, cfg.model_n_layer, cfg.model_n_head, cfg.use_reasoner, cfg.use_residual, cfg.max_reasoning_len, cfg.seq_len + 1, len(tokenizer.vocab), len(tokenizer.vocab) + 1, cfg.add_last_context_token, cfg.add_surogate_loss_to_last_context_token, cfg.increment_pos_id_for_last_context_token)
            else:
                model = QuietStarDiscreteTransformerLM(len(tokenizer.vocab)+1, cfg.model_hidden_dim, cfg.model_n_layer, cfg.model_n_head, cfg.use_reasoner, cfg.use_residual, cfg.max_reasoning_len, cfg.seq_len + 1, len(tokenizer.vocab), len(tokenizer.vocab), cfg.add_last_context_token, cfg.add_surogate_loss_to_last_context_token, cfg.increment_pos_id_for_last_context_token)
        elif cfg.model_type == "reintG":
            model = ReasonerInterpreterGRULM(len(tokenizer.vocab)+2, len(tokenizer.vocab)+2, cfg.base_lm_hidden_dim, cfg.reasoner_hidden_dim, cfg.interpreter_hidden_dim, cfg.use_base_lm, cfg.use_reasoner, cfg.mix_interpeter_base_lm, cfg.simple_lm_head, cfg.linear_lm_head, cfg.weight_groups, cfg.share_lm_head, cfg.variable_len, len(tokenizer.vocab)+1, cfg.stage_wise_downproject_interpreter_rep, cfg.infer_pretrained_base, cfg.infer_pretrained_reasoner, cfg.infer_pretrained_base_reasoner, cfg.parameter_groups, max_reasoning_len=cfg.max_reasoning_len, start_of_thought_token=len(tokenizer.vocab))
            if cfg.variable_len:
                tokenizer.add_token("<sot>")
                tokenizer.add_token("<pad>")
        elif cfg.model_type == "T":
            model = CustomGPT2LMHeadModel(GPT2Config(vocab_size=len(tokenizer.vocab) + 2, # just give some useless tokens to make it fair comparison when using eot and sot
                                                     n_embd=cfg.model_hidden_dim,
                                                     n_layer=cfg.model_n_layer,
                                                     n_head=cfg.model_n_head,
                                                     resid_pdrop=cfg.dropout, 
                                                     embd_pdrop=cfg.dropout, 
                                                     attn_pdrop=cfg.dropout))
        elif cfg.model_type == "G":
            model = LanguageModelLSTM(len(tokenizer.vocab)+2, # just to be fair to the reasoner interpreter. I guess because I could implement ignoring the gradient, but I am lazy and I figure this won't do much?
                                      hidden_dim=cfg.base_lm_hidden_dim,
                                      num_layers=cfg.model_n_layer,
                                      model_type="gru",
                                      linear_lm_head=cfg.linear_lm_head)
        else:
            raise Exception(f"model {cfg.model_type} not implemented")
        model = model.to(device)
        if training_type == "nll":
            get_train_loss = get_nll
        elif training_type == "quiet_starM":
            get_train_loss = partial(get_quiet_star_loss_reason_multiple, policy_loss_beta=cfg.policy_loss_beta, nll_loss_beta=cfg.nll_loss_beta, trice_samples=cfg.trice_samples, train_nll_num=cfg.train_nll_num,
                                    n_tokens_ahead=cfg.n_tokens_ahead, punish_unfinished=cfg.punish_unfinished, length_penalty=cfg.length_penalty, entropy_encouragement_coef=cfg.entropy_encouragement_coef) 
        elif training_type == "quiet_starO":
            get_train_loss = partial(get_quiet_star_loss_reason_once, policy_loss_beta=cfg.policy_loss_beta, nll_loss_beta=cfg.nll_loss_beta, trice_samples=cfg.trice_samples, train_nll_num=cfg.train_nll_num,
                                     last_context_loss_beta=cfg.last_context_loss_beta, n_tokens_ahead=cfg.n_tokens_ahead, entropy_encouragement_coef=cfg.entropy_encouragement_coef) 
        else:
            raise Exception(f"invalid training type {training_type} must be either nll or quiet_star.")
        # TODO: Change learning rate?? smaller because more influence of loss from early hidden representations is increased, potentially unstable?
        def run_with_grad_acc(grad_acc): 
            assert cfg.data_loader_batch_size % grad_acc == 0, "training loop assumes a nicely divisible batch size and gradient accumulation"
            trainer = Trainer(model, tokenizer, device, get_train_loss, get_nll, train_dl, val_dl, base_lr=cfg.base_lr, epochs=epochs, max_steps=cfg.max_steps, eval_every=100, save_directory=cfg.output_dir, experiment_logger=experiment_logger, grad_acc=grad_acc, use_scheduler=cfg.use_scheduler, schedule_punfinished=cfg.schedule_punfinished, punish_unfinished=cfg.punish_unfinished, clip_gradients=cfg.clip_gradients, neuter_dropout_base=cfg.neuter_dropout_base)
            # with torch.autograd.set_detect_anomaly(True):
            trainer.train()
        from accelerate.utils.memory import clear_device_cache
        from accelerate.utils.memory import should_reduce_batch_size as should_reduce_grad_acc_size
        def find_runnable(func):
            for grad_acc in [cfg.grad_acc * 2 ** i for i in range(int(np.log2(cfg.data_loader_batch_size / cfg.grad_acc)))] + [cfg.data_loader_batch_size]:
                try:
                    clear_device_cache(garbage_collection=True)
                    func(grad_acc)
                    return
                except Exception as e:
                    print(f"FAILED to run for grad acc {grad_acc}. Batch size is {cfg.data_loader_batch_size}")
                    if not should_reduce_grad_acc_size(e):
                        raise

            raise RuntimeError("No executable grad accumulation found")
        if cfg.auto_find_grad_acc:
            find_runnable(run_with_grad_acc)
        else:
            run_with_grad_acc(cfg.grad_acc)

        #  i =  log2(cfg.batch_size /cfg.grad_acc) for what i?

        # rnn_lm = LanguageModelLSTM(vocab_size=len(tokenizer.vocab), hidden_dim=model_hidden_dim, num_layers=1, model_type=model_type).to(device) 
        # trainer = Trainer(rnn_lm, tokenizer, device, get_nll, get_nll, train_dl, val_dl, epochs=epochs, eval_every=100, save_directory=save_directory)
        # trainer.train()


if __name__ == "__main__":
    main()