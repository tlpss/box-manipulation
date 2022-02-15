#!/bin/bash

###################################################
##script to use for creating env in a Job request##
###################################################

# move to user home dir
cd ~
# activate conda env
source /opt/conda/bin/activate
conda activate python39

#clone repo
if [[ -d ~/box-manipulation ]]; then
    echo "removing box dir"
    sudo rm -r ~/box-manipulation
fi
git clone https://github.com/tlpss/box-manipulation.git git clone --recurse-submodules

# install keypoint_detection package
# install in dev mode to enable symlinks
pip install -e  box-manipulation/keypoint_detection/keypoint_detection
