FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel

ENV TORCH_CUDA_ARCH_LIST="8.0"

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub \
    && apt-get update \
    && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6
RUN pip install torchvision==0.13.1
RUN pip install spconv-cu113==2.1.25
RUN pip install torch-scatter==2.0.9 -f https://data.pyg.org/whl/torch-1.12.1+cu113.html

RUN pip install \
    munch\
    pandas\
    plyfile\
    pyyaml\
    scikit-learn\
    scipy\
    six\
    tensorboard\
    tensorboardX\
    tqdm\
    opencv-python\
    numpy\
    imageio\
    setuptools

RUN apt update && apt install -q -y --no-install-recommends \
    libsparsehash-dev \
    wget

# BEGIN cmake version 3.18
RUN wget https://cmake.org/files/v3.18/cmake-3.18.0-Linux-x86_64.sh -O /tmp/cmake-install.sh
RUN chmod u+x /tmp/cmake-install.sh
RUN mkdir /opt/cmake-3.18.0
RUN echo "y" | /tmp/cmake-install.sh --prefix=/opt/cmake-3.18.0
RUN rm /tmp/cmake-install.sh
RUN ln -s /opt/cmake-3.18.0/cmake-3.18.0-Linux-x86_64/bin/* /usr/local/bin
# END cmake version 3.18

# BEGIN Install Segmentator 
#WORKDIR /root
WORKDIR /workspace
RUN git clone https://github.com/Karbo123/segmentator.git \
    && cd segmentator/csrc \
    && mkdir build && cd build \
    && cmake .. \
        -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` \
        -DPYTHON_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())")  \
        -DPYTHON_LIBRARY=$(python -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))") \
        -DCMAKE_INSTALL_PREFIX=`python -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())'` \
    && make && make install
# END Install Segmentator

RUN pip install \
    natsort \
    configparse \
    configargparse

# Begin Clone ISBNet
RUN git clone https://github.com/VinAIResearch/ISBNet.git
# End Clone ISBNet

# BEGIN Install pointnet2
ENV MAX_JOBS=1
RUN cd /workspace/ISBNet/isbnet/pointnet2 \
    && python3 setup.py bdist_wheel \
    && cd ./dist \
    && pip3 install pointnet2-0.0.0-cp37-cp37m-linux_x86_64.whl
# END ISBNet Installation

# Begin Setup ISBNet
RUN cd /workspace/ISBNet \
    && python3 setup.py build_ext develop
# End Setup ISBNet

# Begin Set Symlinks for S3DIS dataset
RUN ln -s /root/workspace/dataset/s3dis/Stanford3dDataset_v1.2_Aligned_Version/ /workspace/ISBNet/dataset/s3dis/ \
    && ln -s /root/workspace/dataset/s3dis/learned_superpoint_graph_segmentations/ /workspace/ISBNet/dataset/s3dis/ \
    && ln -s /root/workspace/dataset/s3dis/preprocess/ /workspace/ISBNet/dataset/s3dis \
    && ln -s /root/workspace/dataset/s3dis/superpoints/ /workspace/ISBNet/dataset/s3dis/ \
    && ln -s /root/workspace/dataset/s3dis/out/ /workspace/ISBNet/dataset/s3dis/ \
    && ln -s /root/workspace/dataset/s3dis/head_s3dis_area5.pth /workspace/ISBNet/dataset/s3dis/
# End Set Symlinks for S3DIS dataset
