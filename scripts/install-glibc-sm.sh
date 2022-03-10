#!/bin/bash

# remove existing installation
rm $HOME/glibc-2.27-subset/*
rmdir $HOME/glibc-2.27-subset

tar xvzf packages/glibc-2.27-subset.tgz
mv glibc-2.27-subset $HOME/

#sudo touch /etc/profile.d/jupyter-env.sh
echo 'export LD_LIBRARY_PATH=/home/ec2-user/glibc-2.27-subset:$LD_LIBRARY_PATH' | sudo tee /etc/profile.d/jupyter-env.sh
