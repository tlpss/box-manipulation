#!/bin/bash

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
git clone https://github.com/tlpss/box-manipulation.git

# install keypoint_detection package
pip install box-manipulation/keypoint_detection
