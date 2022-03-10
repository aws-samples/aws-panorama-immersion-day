#!/bin/bash

sudo yum install docker -y

sudo yum install qemu binfmt-support qemu-user-static -y
# binfmt-support is not found, but that error is ignorable

sudo docker run --rm --privileged multiarch/qemu-user-static --reset -p yes

sudo gpasswd -a ec2-user docker
#sudo newgrp docker

