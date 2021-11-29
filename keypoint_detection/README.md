# Keypoint Detection Package
This package provides pytorch modules for a model to detect the box keypoints and for a dataset and dataloader module. It uses Pytorch-Lightning to create an abstraction on top of pytorch and to avoid boilerplate code, and weights and biases for logging and experiment orchestration.

## Training

The `train.py` file is the main entrypoint for using the package. It collects configuration parameters (defaults - argparser - wandb); creates the model, trainer and dataloader and starts the training.

It collects configuration parameters in 3 phases (where each phase overrides the params from previous phases if the same parameter is passed)
1. from the default config dict
2. from the argparser
3. from wandb (in case of a sweep)

and then starts the training.

### Local Training
- `python train.py --help` for all configurable parameters
- `python train.py -- <custom_param_values> `
- wandb will provide a url to monitor the training process

### Wandb Sweep
to create a wandb sweep (for hyperparam tuning or just conveniently running experiments):

- go to the box-manipulation page on wandb and create a new project (or open an existing one)
- create a new sweep and define the configuration (see https://docs.wandb.ai/guides/sweeps/configuration), always provide a path to the dataset image folder and the json file.
- start the sweep
- start the desired number of agents (can be on any device that has this package + the requirements) using the sweepID : `wandb agent airo-box-manipulation/<project>/<sweepID>`

#### starting agents on GPULab
The hacky way:
- start a jupyterhub instance using the project and the Jupyterlab-pytorch docker image from https://gitlab.ilabt.imec.be/tlips/gpulab
- pull this repository anywhere on the machine (except the project mount), pip install `box-manipulation/keypoint_dection`
- navigate to the folder relative to which you specified the python script in the sweep
- start the agent(s)
- paste the wandb authorization key
the nice way:
- #TODO create job and submit to GPULab CLI.

### known issues
- if you get a wandb Connection error inside a docker environment, consider restarting the docker container as discussed [here](https://stackoverflow.com/questions/44761246/temporary-failure-in-name-resolution-errno-3-with-docker)
## Notebooks
the top-level folder also contains a `notebooks` folder, in which an (undocumented) example can be found of how to train the system.

## Development Guide
### Package structure
package is used to manage imports. Not ment as standalone python package.
hence no dependencies are listed etc in the setup.py.

### Parameter configuration  setup

There were a few requirements for the hyperparam setup
1. models can declare their params explicitly in the init function and provide default values (separation of concerns -> other code does not need to manage this modulde)
2. ease-of-use both from CLI and with wandb (no lock-in to the wandb interface)

PL suggests the use of Argparsers which declare default values for all arguments to then pass this Namespace (or dict after conversion) as the single parameter to the init function of all models. The trainer can take in the namespace from the argparser to create an instance.

Wandb is rather agnostic but needs to be able to override local parameters to orchestrate sweeps. This happens by calling `wandb.init(config= local_config)` and then getting the (possibly updated) `wandb.config` object that is basically a dict of all hyperparameters.
Wandb also starts up sweeps by updating both the config and passing them as `-- <name>=<value>` args to the python executable, which allows to catch them with an argparser.

So given these constraints and requirements:

#### Models
- define all hyperparameters as keyword arguments with default values and take one additional **kwargs argument that catches hyperparameters that are not for this class. (alternative would be to filter them out, but this requires more code and provides no clear benefit?). The class can then be initialized by calling `cls(** params_dict)` which will override any kwargs default value and bring the other key,value pairs in the **kwargs argument.
- The init function calls the pl `save_parameters` to log the params, even if they are on their default value (kwargs is excluded to avoid logging other module's parameters)
- provide a function to add all their kwargs to an ArugmentParser, as optional and without a default value (which results in None if they are not specified)

#### Trainer
for the trainer class, a util function was made in `trainer.py` which creates the trainer from a argument dict by creating a dummy Namespace and filtering out all non-relevant kwargs from the params dict. the existing function to add them to a parser is used.

#### System params
all non-local-module parameters can be given default values in the default_dict (could be replaced by a default yaml file) and should be added manually to the function that adds them to a parser.

#### collecting the params
1. use default dict
2. update with arguments from the argparser that are not None (unspecified optional)
3. init wandb with these local params
4. fetch (updated if sweep) params from wandb instance (not that this is not stricly required since wandb sweep also adds them as arguments to the script, which means they are already caught by the argparser.., it is however more modular since now the code could be used even without argparser..; Using the argparsers also makes sure that no typos can occur in the wandb parameters since this would result in an 'unknown argument error' in the argparser)

#### alternatives:
- specify everything locally in yaml files, however this requires to create new yaml files locally for each specific run
-
### Testing
- pip install pytest
- run pytest (manually or using vscode)
