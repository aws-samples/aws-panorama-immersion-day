#!/bin/bash

sudo yum install bison -y

wget http://ftp.gnu.org/gnu/libc/glibc-2.27.tar.gz
tar xvzf glibc-2.27.tar.gz
mkdir glibc-2.27-build
mkdir glibc-2.27-install
cd glibc-2.27-build
../glibc-2.27/configure --prefix=$HOME/glibc-2.27-full
make
make install


# Cherry pick libm

mkdir $HOME/glibc-2.27-subset
cp $HOME/glibc-2.27-full/lib/libm.so.6 $HOME/glibc-2.27-subset/

