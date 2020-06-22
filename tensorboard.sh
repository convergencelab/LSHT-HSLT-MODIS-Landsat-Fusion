#!/bin/bash
# activate env
source /home/x2017sre/projects/def-jhughe54/x2017sre/virtualenv/tensorflow/bin/activate

# connect ssh with remote port 6006 transferred to local port 6006
ssh gra-login1 -L 6006:127.0.0.1:6006 -n -N -f


# launch tensorboard
tensorboard --logdir=/home/x2017sre/logs/gradient_tape --port=6006