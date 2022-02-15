# Keypoint Detection Package
This folder uses the [keypoint detector](https://github.com/tlpss/keypoint-detection) package and adds some scripts for submitting jobs to GPULab(UGent infrastructure) and some notebooks.

## Notebooks
the top-level folder also contains a `notebooks` folder, in which an (undocumented) example can be found of how to train the system.

### Wandb Sweep
to create a wandb sweep (for hyperparam tuning or just conveniently running experiments):

- go to the box-manipulation page on wandb and create a new project (or open an existing one)
- create a new sweep and define the configuration (see https://docs.wandb.ai/guides/sweeps/configuration), always provide a path to the dataset image folder and the json file.
- start the sweep
- start the desired number of agents (can be on any device that has this package + the requirements) using the sweepID : `wandb agent airo-box-manipulation/<project>/<sweepID>`

### starting agents on GPULab
**The hacky way (JupyterHub)**
- start a jupyterhub instance using the project and the Jupyterlab-pytorch docker image from https://gitlab.ilabt.imec.be/tlips/gpulab
- pull this repository anywhere on the machine (except the project mount), and run `setup.sh`.
- activate the conda environment (`conda init` and `conda activate <env>)
- navigate to the folder relative to which you specified the python script in the sweep.
- start the agent(s)
- paste the wandb authorization key


**the nice way (Job submission)**
- copy the `run-sweep-on-wall.sh` and `setup-keypoint-detector.sh` files to your /project folder and chmod +x them to make executable.
- run the `submit_sweep_agent_to_wall.py` file, which will submit a job request with one GPU to cluster 4 and start an agent for the specified

### known issues
- if you get a wandb Connection error inside a docker environment, consider restarting the docker container as discussed [here](https://stackoverflow.com/questions/44761246/temporary-failure-in-name-resolution-errno-3-with-docker)
