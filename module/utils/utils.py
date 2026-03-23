from ..metric import METRIC_MAP
from ..datamodule import DATAMODULE_MAP
from ..model import MODEL_MAP
from typing import List, Union, Dict, Any, TypeVar, Type
from torchmetrics.metric import Metric
from torchmetrics import MetricCollection
import inspect
from torch import nn
import torch
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
import os
import pprint

T = TypeVar("T")

def update_dict_with_recursive_update(
    base_dict: Dict[str, Any], update_dict: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Update a base dictionary with a recursive update.

    Args:
        base_dict (Dict[str, Any]): Base dictionary to update.
        update_dict (Dict[str, Any]): Update dictionary.

    Returns:
        Dict[str, Any]: Updated dictionary.
    """
    for key, value in update_dict.items():
        if isinstance(value, dict):
            base_dict[key] = update_dict_with_recursive_update(
                base_dict.get(key, {}), value
            )
        else:
            base_dict[key] = value
    return base_dict


def load_params_from_dict(config: Dict[str, Any], module: Type[T]) -> T:
    """
    Load parameters from a dictionary and initialize a module.

    Args:
        config (Dict[str, Any]): Configuration dictionary.
        module (Type[T]): Module class to initialize.

    Returns:
        T: Initialized module instance.

    Raises:
        ValueError: If a required parameter is not found in the config.
    """
    init_signature = inspect.signature(module.__init__)
    init_params = init_signature.parameters
    init_args = {}
    var_args = []
    var_kwargs = {}

    for param_name, param in init_params.items():
        if param_name == "self" or param_name == "name":
            continue
        elif param_name in config:
            init_args[param_name] = config[param_name]
        elif param.default is not param.empty:
            init_args[param_name] = param.default
        elif param.kind == param.VAR_POSITIONAL:  # Handles *args
            if param_name in config:
                var_args.extend(config[param_name])
        elif param.kind == param.VAR_KEYWORD:  # Handles **kwargs
            if param_name in config:
                var_kwargs.update(config[param_name])
            else:
                remaining_kwargs = {
                    k: v for k, v in config.items() if k not in init_args
                }
                var_kwargs.update(remaining_kwargs)
        else:
            raise ValueError(f"Required parameter '{param_name}' not found in config.")

    return module(*var_args, **init_args, **var_kwargs)


def load_from_file(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a JSON or YAML file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        Dict[str, Any]: Loaded configuration dictionary.

    Raises:
        FileNotFoundError: If the config file is not found.
        ValueError: If the file format is not supported.
    """
    if config_path is None:
        return {}
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    config_path = str(config_path)
    if config_path.endswith(".json"):
        import json

        with open(config_path, "r") as f:
            config = json.load(f)
    elif config_path.endswith((".yaml", ".yml")):
        import yaml

        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    else:
        raise ValueError(f"Unsupported file format: {config_path}")
    return config


def initialize_datamodule_from_dict(config: Dict[str, Any]) -> LightningDataModule:
    """
    Initialize a LightningDataModule from a configuration dictionary.

    Args:
        config (Dict[str, Any]): Configuration dictionary for the datamodule.

    Returns:
        LightningDataModule: Initialized datamodule.

    Raises:
        ValueError: If the datamodule name is not found in the config or not recognized.
    """
    if "name" not in config:
        raise ValueError("Datamodule name not found in config")
    if config["name"] not in DATAMODULE_MAP:
        raise ValueError(
            f"Datamodule name {config['name']} is not recognized in {DATAMODULE_MAP.keys()}"
        )
    return load_params_from_dict(config, DATAMODULE_MAP[config["name"]])


def init_metrics_from_list(config: Dict[str,Dict]) -> MetricCollection:
    """
    Initialize metrics from various input formats.

    Args:
        config (Any): Metric configuration in various formats.

    Returns:
        Dict[str, Metric]: Initialized metrics.

    Raises:
        ValueError: If the input format is not supported.
    """
    if isinstance(config, dict):
        return MetricCollection(
            {
                key: METRIC_MAP[value.get("name", key)](**value.get("params", {}))
                for key, value in config.items()
            }
        )
    else:
        raise ValueError(
            f"metrics for {type(config)} is not supported, please use Dict[str,Dict]"
        )


def init_metrics_from_dict(config: Dict[str, Any]) -> Dict[str, MetricCollection]:
    """
    Initialize metrics from a dictionary configuration.

    Args:
        config (Dict[str, Any]): Metric configuration dictionary.

    Returns:
        Dict[str, Dict[str, MetricCollection]]: Initialized metrics.

    Raises:
        ValueError: If the input format is not supported.
    """
    if isinstance(config, dict):
        return {key: init_metrics_from_list(config[key]) for key in config}
    else:
        raise ValueError(
            f"metrics for {type(config)} is not supported, please use Dict[str,Any]"
        )


def initialize_model_from_dict(
    model_config: Dict[str, Any], metrics_config: Union[Dict[str, Any], None] = None
) -> Union[nn.Module, LightningModule]:
    """
    Initialize a model from a configuration dictionary.

    Args:
        model_config (Dict[str, Any]): Model configuration dictionary.
        metrics_config (Union[Dict[str, Any], None], optional): Metrics configuration dictionary.

    Returns:
        Union[nn.Module, LightningModule]: Initialized model.

    Raises:
        ValueError: If the model name is not found in the config or not recognized.
    """
    if "name" not in model_config:
        raise ValueError("Model name not found in config")
    if model_config["name"] not in MODEL_MAP:
        raise ValueError(
            f"Model name {model_config['name']} is not recognized in {MODEL_MAP.keys()}"
        )
    if metrics_config is not None:
        model_config.update(init_metrics_from_dict(metrics_config))
    return load_params_from_dict(model_config, MODEL_MAP[model_config["name"]])


def initialize_trainer_from_dict(config: Dict[str, Any]) -> Trainer:
    """
    Initialize a PyTorch Lightning Trainer from a configuration dictionary.

    Args:
        config (Dict[str, Any]): Trainer configuration dictionary.

    Returns:
        Trainer: Initialized PyTorch Lightning Trainer.

    Raises:
        ValueError: If a callback module fails to import.
    """
    trainer_config = config.copy()
    init_args = {}
    if "callbacks" in trainer_config:
        import importlib

        init_args["callbacks"] = []
        for module in trainer_config["callbacks"]:
            try:
                module_path, class_name = module.get("name").rsplit(".", 1)
                module_class = importlib.import_module(module_path)
                module_class = getattr(module_class, class_name)
                init_args["callbacks"].append(load_params_from_dict(module, module_class))
            except:
                raise ValueError(
                    f"Failed to import callback module name: {module}, you should follow the format 'module.class'"
                )
    if "logger" in trainer_config:
        from pytorch_lightning.loggers import TensorBoardLogger
        init_args["logger"] = load_params_from_dict(trainer_config["logger"], TensorBoardLogger)
    for key in init_args:
        trainer_config[key] = init_args[key]
    # from pytorch_lightning.strategies import DDPStrategy
    # trainer_config["strategy"] = DDPStrategy(
    #     process_group_backend="gloo",
    #     find_unused_parameters=False
    # )
    return load_params_from_dict(trainer_config, Trainer)


def continue_model(config_path: str=None,config_dict: Dict[str, Any]={}):
    """
    Continue training or testing a model based on a configuration file.

    Args:
        config_path (str): Path to the configuration file.
        config_dict (Dict[str, Any]): Configuration dictionary.

    Raises:
        ValueError: If required configuration items are missing or invalid.
        FileNotFoundError: If the config file is not found.
    """
    if config_path is None:
        config = {} 
    else:
        config = load_from_file(config_path)
    config = update_dict_with_recursive_update(config, config_dict)
    if config.get("datamodule", None) is None:
        raise ValueError("Datamodule not found in config")
    if config.get("model", None) is None:
        raise ValueError("Model not found in config")
    if config.get("trainer", None) is None:
        raise ValueError("Trainer not found in config")
    print("Configuration:")
    pprint.pprint(config,compact=False,sort_dicts=False,width=1)
    datamodule: LightningDataModule = initialize_datamodule_from_dict(
        config["datamodule"]
    )
    model: LightningModule = initialize_model_from_dict(
        config["model"], config.get("metrics", None)
    )
    trainer: Trainer = initialize_trainer_from_dict(config["trainer"])
    
    ckpt = config.get("resume_from_checkpoint", None)
    train = config.get("train", True)
    test = config.get("test", True)

    if ckpt is not None:        
        if os.path.exists(ckpt):
            print(f"Loading checkpoint from {ckpt}")
        else:
            print("Failed to load checkpoint, training from scratch.(This may be unstable.)")
            print("Please avoid using --resume_from_checkpoint when training from scratch.")
            ckpt = None        
            

            
    if train:
        if config.get("continue_training", False) and ckpt is not None:
            trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt)
            ckpt = None 
        else:
            trainer.fit(model=model, datamodule=datamodule, ckpt_path=None)

    if test:
        if not train and ckpt:
            trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt)
        else:
            trainer.test(model=model, datamodule=datamodule)
