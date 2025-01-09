#!/bin/bash
export TORCH_CUDA_ARCH_LIST="8.6"
export MAX_JOBS=1
cd ~/workspace/ISBNet/isbnet/pointnet2/dist || exit
pip3 install pointnet2-0.0.0-cp37-cp37m-linux_x86_64.whl
cd ~/workspace/ISBNet || exit
python3 setup.py build_ext develop
