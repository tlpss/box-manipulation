#!/bin/bash

#####################################################################
##script to run keypoint detector wandb sweep from job submission  ##
#####################################################################
# do not forget to chmod + x the sh files on the mount, so that they become executable.


# $1 is the auth key for wandb
# $2 is the weep ID to start an agent for.

source /project/setup-keypoint-detector.sh

wandb login "$1"
wandb agent "$2"
