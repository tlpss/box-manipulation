import inspect
from argparse import ArgumentParser, Namespace
from typing import Tuple

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.trainer.trainer import Trainer

import wandb
from keypoint_detection import BoxKeypointsDataModule, KeypointDetector
from keypoint_detection.src.datamodule import BoxDatasetPreloaded

default_config = {
    ## system params
    # Data params
    "image_dataset_path": "/workspaces/box-manipulation/datasets/box_dataset2",
    "json_dataset_path": "/workspaces/box-manipulation/datasets/box_dataset2/dataset.json",
    "batch_size": 4,
    "train_val_split_ratio": 0.1,
    # logging info
    "wandb_entity": "airo-box-manipulation",
    "wandb_project": "test-project",
    # Trainer params
    "seed": 2021,
    "max_epochs": 2,
    "gpus": 0,
    # model params -> default values in the model.
}


def add_system_args(parent_parser: ArgumentParser) -> ArgumentParser:
    """
    function that adds all system configuration (hyper)parameters to the provided argumentparser
    """
    parser = parent_parser.add_argument_group("Trainer")
    parser.add_argument("--batch_size", required=False, type=int)
    parser.add_argument("--train_val_split_ratio", required=False, type=float)
    parser.add_argument("--image_dataset_path", required=False, type=str)
    parser.add_argument("--json_dataset_path", required=False, type=str)

    return parent_parser


def create_pl_trainer_from_args(hparams: dict, wandb_logger: WandbLogger) -> Trainer:
    """
    function that creates a pl.Trainer instance from the given global hyperparameters and logger.

    pl only supports constructing from an Argparser or its output by default, but also allows to pass additional **kwargs.
    However, these kwargs must be present in the __init__ function, since ther is no additional **kwargs argument in the function
    to catch other kwargs (unlike in the Detector Module for example).
    Hence the global config dict is filtered to only include parameters that are present in the init function.

    To comply with the pl.Trainer.from_argparse_args declaration, an empty Nampespace (the result of argparser) is created and added to the call.
    """

    valid_kwargs = inspect.signature(Trainer.__init__).parameters
    trainer_kwargs = {name: hparams[name] for name in valid_kwargs if name in hparams}
    trainer_kwargs.update({"logger": wandb_logger})

    # this call will add all relevant hyperparameters to the trainer, overwriting the empty namespace
    # and the default values in the Trainer class
    trainer = pl.Trainer.from_argparse_args(Namespace(), **trainer_kwargs)
    return trainer


def main(hparams: dict) -> Tuple[KeypointDetector, pl.Trainer]:
    """
    Initializes the datamodule, model and trainer based on the global hyperparameters.
    calls trainer.fit(model, module) afterwards and returns both model and trainer.
    """
    pl.seed_everything(hparams["seed"], workers=True)
    model = KeypointDetector(**hparams)

    dataset = BoxDatasetPreloaded(hparams["json_dataset_path"], hparams["image_dataset_path"], n_io_attempts=5)

    module = BoxKeypointsDataModule(
        dataset,
        hparams["batch_size"],
        hparams["train_val_split_ratio"],
    )
    wandb_logger = WandbLogger(project=default_config["wandb_project"], entity=default_config["wandb_entity"])
    trainer = create_pl_trainer_from_args(hparams, wandb_logger)
    trainer.fit(model, module)
    return model, trainer


if __name__ == "__main__":
    """
    1. loads default configuration parameters
    2. creates argumentparser with Model, Trainer and system paramaters; which can be used to overwrite default parameters
    when running python train.py --<param> <param_value>
    3. sets ups wandb and loads the local config in wandb.
    4. pulls the config from the wandb instance, which allows wandb to update this config when a sweep is used to set some config parameters
    5. calls the main function to start the training process
    """

    # start with the default config hyperparameters
    config = default_config

    # create the parser, add module arguments and the system arguments
    parser = ArgumentParser()
    parser = add_system_args(parser)
    parser = KeypointDetector.add_model_argparse_args(parser)
    parser = Trainer.add_argparse_args(parser)

    # get parser arguments and filter the specified arguments
    args = vars(parser.parse_args())
    # remove the unused optional items without default, which have None as key

    # include those that are at their default values
    # this is could result in cluttering the list of hyperparameters with the unused arguments of the Trainer, but
    # those of the model are logged anyways (see init of the model)
    # however it is done for completeness.
    args = {k: v for k, v in args.items() if v is not None}  # and v is not parser.get_default(k)}

    print(f" argparse arguments ={args}")

    # update the hyperparameters with the argparse parameters
    # this adds new <key,value> pairs if the keys did not exist and
    # updates the key with the new value pairs otherwise.
    # (so argparse > default)
    config.update(args)

    print(f" updated config parameters before wandb  = {config}")

    # initialize wandb here, this allows for using wandb sweeps.
    # with sweeps, wandb will send hyperparameters to the current agent after the init
    # these can then be found in the 'config'
    # (so wandb params > argparse > default)
    wandb.init(project=config["wandb_project"], entity=config["wandb_entity"], config=config)

    # get (possibly updated by sweep) config parameters
    config = wandb.config
    print(f" config after wandb init: {config}")

    # actual training.
    print("starting trainig")
    main(config)
