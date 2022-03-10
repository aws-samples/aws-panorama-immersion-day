#!/bin/bash

# install dev foundation

sudo yum install git -y
sudo yum install gcc-c++ -y         # for cmake
sudo yum install openssl-devel -y   # for cmake

# install Python libraries and CLI tools

sudo pip3 install boto3
sudo pip3 install awscli
sudo pip3 install sagemaker
sudo pip3 install panoramacli
sudo pip3 install numpy
sudo pip3 install matplotlib
sudo pip3 install jupyterlab

./scripts/install-docker.sh

./scripts/install-cmake3.sh

./scripts/install-dlr.sh

./scripts/install-mxnet.sh

./scripts/install-samples.sh

./scripts/create-opt-aws-panorama.sh

./scripts/configure-aws.sh
