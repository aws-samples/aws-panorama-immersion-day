#!/bin/bash

cd $HOME/aws-panorama-samples

export AWS_REGION=us-east-1

export LD_LIBRARY_PATH=$HOME/glibc-2.27-subset:$LD_LIBRARY_PATH

jupyter-lab --no-browser --allow-root --port 8888 --notebook-dir ~

