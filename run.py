from functools import partial
import os
from dotenv import load_dotenv
load_dotenv()
import sys
import contextlib
import io
import pprint
import logging

class Tee(io.TextIOWrapper):
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
        return 0





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
    exclude: str|None = "major,crushinator,nestor,voltron,megabot,samantha,uniblab,gundam,consu,puma"
    additional_parameters: Dict[str, str] = field(default_factory=lambda: {"gpus-per-task": "a40:1"})
    array_parallelism: int = 20
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
    
    reasoner_interpreter_vocab_size: int = MISSING
    base_lm_hidden_dim: int = MISSING
    reasoner_hidden_dim: int = MISSING
    interpreter_hidden_dim: int = "${reasoner_hidden_dim}" # type: ignore
    use_reasoner: bool = MISSING
    use_base_lm: bool = MISSING
    simple_lm_head: bool = MISSING
    weight_groups: list|None = None
    share_lm_head: bool = MISSING

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
                       "training_type": "quiet_star",
                       "run_type": "train"}
cs.store(name="QuietSTaRGRURunConfig", node=QuietSTaRGRURunConfig, group="run_modifier", package="_global_")

ReasonerInterpreterGRURunConfig = {
                       "model_type": "reint",
                       "reasoner_interpreter_vocab_size": 67, # this is the default it is the same as the char level llm for shakespeare + 1 (this for the start/end of thought token).
                       "base_lm_hidden_dim": 100,
                       "reasoner_hidden_dim": 100,
                    #    "interpreter_hidden_dim": 100,
                       "use_base_lm": True,
                       "use_reasoner": True,
                       "n_tokens_ahead": 1,
                       "training_type": "quiet_star",
                       "run_type": "train",
                       "simple_lm_head": False,
                       "weight_groups": ['A', 'B', 'C', 'D'],
                       "share_lm_head": False,
                       "auto_find_grad_acc": True}
cs.store(name="ReasonerInterpreterGRURunConfig", node=ReasonerInterpreterGRURunConfig, group="run_modifier", package="_global_")

InterpreterLowVarGRURunConfig = {
                        "defaults":  [{"/run_modifier": ["ReasonerInterpreterGRURunConfig"]}], # must be a list for some reason
                        "data_loader_batch_size": 8,
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
                       "model_type": "quietdiscrete",
                       "model_hidden_dim": 100,
                       "use_reasoner": True,
                       "use_base_lm": True,
                       "n_tokens_ahead": 1,
                       "max_reasoning_len": 10,
                       "training_type": "quiet_star",
                       "run_type": "train",
                       "auto_find_grad_acc": True,
}
cs.store(name="QuietSTaRDiscreteRunConfig", node=QuietSTaRDiscreteRunConfig, group="run_modifier", package="_global_")


DebugRunConfig = {
                        # "defaults": [{"/run_modifier": ["QuietSTaRDiscreteRunConfig"]}],
                        "defaults": [{"/run_modifier": ["ReasonerInterpreterGRURunConfig"]}],
                        # "max_reasoning_len": 50,
                        "data_loader_batch_size": 128,
                        # "use_reasoner": False,
                        "seed": 7,
                        "simple_lm_head": True,
                        # 'device': 'cpu',
                        # "use_base_lm": True,
                        # "info": "seperateInterpreterDEBUG_",
                        # "debug_cfg": "seperateInterpreter",
                        "auto_find_grad_acc": True,
                        "experiment_logger": 'offlinelogger',
                }
cs.store(name="DebugRunConfig", node=DebugRunConfig, group="run_modifier", package="_global_")

def get_run_name_from_cfg(cfg: RunConfig):
    run_name = f"{cfg.info}model={cfg.model_type}"
    if cfg.run_type == "train":
        run_name += f"_train={cfg.training_type}_bz={cfg.data_loader_batch_size}"
    if cfg.model_type == "reint":
        run_name += f"blmhdim={cfg.base_lm_hidden_dim}_rhdim={cfg.reasoner_hidden_dim}_ihdim={cfg.interpreter_hidden_dim}_rivocab={cfg.reasoner_interpreter_vocab_size}_maxr={cfg.max_reasoning_len}"
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
    if cfg.model_type == "quietdiscrete":
        run_name += f"hdim={cfg.model_hidden_dim}_maxr={cfg.max_reasoning_len}"
        if cfg.use_reasoner == False:
            run_name += f"_use_r={cfg.use_reasoner}"
        if cfg.use_base_lm == False:
            run_name += f"_use_lm={cfg.use_base_lm}"
    if cfg.training_type == "quiet_star":
        run_name += f"_trice={cfg.trice_samples}_nahead={cfg.n_tokens_ahead}"
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
        # with open(os.path.join(, 'logging.txt'), 'w') as f:
        #     output_files = Tee(sys.stdout, f)
        #     with contextlib.redirect_stdout(output_files):

        # import here for lazy loading so hydra command line autocomplete remains fast.
        from quiet_star_replicate.trainer.trainer import LatentRationaleTrainer, Trainer
        from quiet_star_replicate.model.model import LanguageModelLSTM, get_nll, QuietStarLanguageModelrGRU, get_quiet_star_loss, ReasonerInterpreterModelGRU, QuietStarDiscreteGRULM
        from quiet_star_replicate.data.data import CustomTokenizer, get_train_val_test_datasets, get_shakespeare_collate_fn
        from torch.utils.data import Dataset, DataLoader

        data_str = open(os.path.join(os.environ.get("data_file_path", ""),'./tiny_shakespeare.txt'), 'r').read()
        tokenizer = CustomTokenizer(data_str)
        train_ds, val_ds, test_ds = get_train_val_test_datasets(data_str, seq_len=seq_len)
        collate_fn = get_shakespeare_collate_fn(tokenizer)
        train_dl = DataLoader(train_ds, batch_size=data_loader_batch_size, collate_fn=collate_fn, shuffle=True, num_workers=data_loader_num_workers)
        val_dl = DataLoader(val_ds, batch_size=data_loader_batch_size, collate_fn=collate_fn, shuffle=False, num_workers=data_loader_num_workers)

        
        # q_star_lm = QuietStarLanguageModelrGRU(len(tokenizer.vocab), model_hidden_dim, 1, model_type).to(device)
        if cfg.model_type == "quietdiscrete":
            model = QuietStarDiscreteGRULM(len(tokenizer.vocab)+1, cfg.model_hidden_dim, cfg.use_base_lm, cfg.use_reasoner, cfg.max_reasoning_len, len(tokenizer.vocab), cfg.debug_cfg)
        elif cfg.model_type == "reint":
            model = ReasonerInterpreterModelGRU(len(tokenizer.vocab)+1, cfg.reasoner_interpreter_vocab_size, cfg.base_lm_hidden_dim, cfg.reasoner_hidden_dim, cfg.interpreter_hidden_dim, cfg.use_base_lm, cfg.use_reasoner, cfg.simple_lm_head, cfg.weight_groups, cfg.share_lm_head, cfg.max_reasoning_len, len(tokenizer.vocab))
        else:
            raise Exception(f"model {cfg.model_type} not implemented")
        model = model.to(device)
        if training_type == "nll":
            get_train_loss = get_nll
        elif training_type == "quiet_star":
            get_train_loss = partial(get_quiet_star_loss, policy_loss_beta=cfg.policy_loss_beta, trice_samples=cfg.trice_samples, 
                                    n_tokens_ahead=cfg.n_tokens_ahead) 
        else:
            raise Exception(f"invalid training type {training_type} must be either nll or quiet_star.")
        # TODO: Change learning rate?? smaller because more influence of loss from early hidden representations is increased, potentially unstable?
        def run_with_grad_acc(grad_acc):
            assert cfg.data_loader_batch_size % grad_acc == 0, "training loop assumes a nicely divisible batch size and gradient accumulation"
            trainer = Trainer(model, tokenizer, device, get_train_loss, get_nll, train_dl, val_dl, epochs=epochs, eval_every=100, save_directory=cfg.output_dir, experiment_logger=experiment_logger, grad_acc=grad_acc)
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