from module.utils.utils import continue_model, load_from_file, update_dict_with_recursive_update
import torch
import argparse
torch.set_float32_matmul_precision("high")
# ignore user warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="My training script, it's recommended to use the config files to specify the parameters."
    )
    # config files paths
    parser.add_argument(
        "-cfg",
        "--config_path",
        type=str,
        default=None,
        help="Path to default config file, load this file first",
    )
    parser.add_argument(
        "-data_cfg",
        "--data_config_path",
        type=str,
        default="config/data_config/default_data.yaml",
        help="Path to data config file, load this file before the initial config file, override the parameters in the previous config file",
    )
    parser.add_argument(
        "-trainer_cfg",
        "--trainer_config_path",
        type=str,
        default="config/trainer_config/default_trainer.yaml",
        help="Path to trainer config file, load this file after the data config file, override the parameters in the previous config file",
    )
    parser.add_argument(
        "-model_cfg",
        "--model_config_path",
        type=str,
        default="config/model_config/DBMIF.yaml",
        help="Path to model config file, load this file after the trainer config file, override the parameters in the previous config file",
    )
    parser.add_argument(
        "-additional_cfg",
        "--additional_config_path",
        type=str,
        default=None,
        help="Path to additional config file, this file will be loaded after all other config files, and will override the parameters in the previous files",
    )
    # some additional aguments for training
    # trainer arguments
    parser.add_argument(
        "-r",
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Resume from checkpoint path, can be find in trainer_config/default_trainer.yaml with the key 'checkpoint'.",
    )
    parser.add_argument(
        "-e",
        "--num_epochs",
        type=int,
        default=None,
        help="Number of epochs to train, can be find in trainer_config/default_trainer.yaml with the key 'trainer.max_epochs'",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=None,
        help="Batch size for training, will not effect the batch size for testing, can be find in data_config/abcs_full.yaml with the key 'datamodule.bs_train', 'datamodule.bs_val'",
    )
    parser.add_argument(
        "-j",
        "--num_workers",
        type=int,
        default=None,
        help="Number of workers for data loading, can be find in data_config/abcs_full.yaml with the key 'datamodule.num_workers'",
    )
    parser.add_argument(
        "-l",
        "--log_dir",
        type=str,
        default=None,
        help="Directory to save logs, can be find in trainer_config/default_trainer.yaml with the key 'logger.save_dir'",
    )
    parser.add_argument(
        "-n",
        "--log_name",
        type=str,
        default=None,
        help="Name of the experiment, used for logging, can be find in trainer_config/default_trainer.yaml with the key 'loggers.name', if you want to specify other loggers, you can modify the config file",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default=None,
        help="Precision for training(matmul), can be 32, 16, bf-16, also with fixed, default is bf16-mixed, can be find in trainer_config/default_trainer.yaml with the key 'trainer.precision'",
    )
    parser.add_argument(
        "-d",
        "--num_nodes",
        type=int,
        default=1,
        help="Number of nodes for distributed training, if num_nodes > 1, will use ddp for training, can be find in trainer_config/default_trainer.yaml with the key 'trainer.num_nodes'",
    )
    args = parser.parse_args()
    # load config files
    config_dict = load_from_file(args.config_path)
    config_dict = update_dict_with_recursive_update(
        config_dict, load_from_file(args.data_config_path)
    )
    config_dict = update_dict_with_recursive_update(
        config_dict, load_from_file(args.trainer_config_path)
    )
    config_dict = update_dict_with_recursive_update(
        config_dict, load_from_file(args.model_config_path)
    )
    config_dict = update_dict_with_recursive_update(
        config_dict, load_from_file(args.additional_config_path)
    )
    # update config with arguments
    if args.resume_from_checkpoint is not None:
        config_dict = update_dict_with_recursive_update(
            config_dict, {"checkpoint": args.resume_from_checkpoint, "continue_training": True}
        )
    if args.num_epochs is not None:
        config_dict = update_dict_with_recursive_update(
            config_dict, {"trainer": {"max_epochs": args.num_epochs}}
        )
    if args.batch_size is not None:
        config_dict = update_dict_with_recursive_update(
            config_dict, {"datamodule": {"bs_train": args.batch_size, "bs_val": args.batch_size}}
        )
    if args.num_workers is not None:
        config_dict = update_dict_with_recursive_update(
            config_dict, {"datamodule": {"num_workers": args.num_workers}}
        )
    if args.log_dir is not None:
        config_dict = update_dict_with_recursive_update(
            config_dict, {"trainer": {"logger": {"save_dir": args.log_dir}}}
        )
    if args.log_name is not None:
        config_dict = update_dict_with_recursive_update(
            config_dict,
            {"trainer": {"logger": {"name": args.log_name}}},
        )
    if args.num_nodes is not None:
        config_dict = update_dict_with_recursive_update(
            config_dict, {"trainer": {"num_nodes": args.num_nodes}}
        )
        if args.num_nodes > 1:
            config_dict = update_dict_with_recursive_update(
                config_dict, {"trainer": {"accelerator": "auto", "strategy": "ddp"}}
            )
    if args.precision is not None:
        config_dict = update_dict_with_recursive_update(
            config_dict, {"trainer": {"precision": args.precision}}
        )
    # continue training
    trainer = continue_model(config_dict=config_dict)
