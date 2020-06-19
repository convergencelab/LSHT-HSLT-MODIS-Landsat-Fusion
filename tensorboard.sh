#!/bin/bash

# connect ssh with remote port 6006 transferred to local port 6006
ssh gra-login1 -L 6006:127.0.0.1:6006 ${GRAHAMSSH} -n -N -f
# activate env
source "/project/6026587/x2017sre/virtualenv/tensorflow/bin/activate"
# launch tensorboard
tensorboard --logdir=/home/x2017sre/logs/gradient_tape/20200619-080625