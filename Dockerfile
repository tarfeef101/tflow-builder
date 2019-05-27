# Build tflow cause they suck with version support
# Define base image, in this case debian stretch w/ python3

FROM python:3.6-slim-stretch AS builder
LABEL maintainer="tarfeef101"

# Set to non-interactive mode as container should not be entered
#ENV DEBIAN_FRONTEND noninteractive
SHELL ["/bin/bash", "-c"]

# Install required packages, etc
RUN apt -y update && \
    apt -y upgrade && \
    apt -y install curl wget git bzip2 unzip && \
    apt purge && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*
ENV LANG C.UTF-8

# Install pip packages
RUN pip install -U pip six numpy wheel mock
RUN pip install -U keras_applications==1.0.6 --no-deps
RUN pip install -U keras_preprocessing==1.0.5 --no-deps
RUN pip install -U google_pasta>=0.1.1 gast>=0.2.0 astor>=0.2.0 absl-py>=0.1.6 protobuf>=0.1.6 'tensorboard>=1.12.0,<1.13.0' 'tensorflow_estimator>=1.13.0rc0,<1.14.0rc0' termcolor>=1.1.0

# install cuda
RUN apt update && \
    apt -y install ca-certificates apt-transport-https gnupg1-curl && \
    rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub | apt-key add - && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
    echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list

ENV CUDA_VERSION 9.1.85

ENV CUDA_PKG_VERSION 9-1=$CUDA_VERSION-1
RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-cudart-$CUDA_PKG_VERSION && \
    ln -s cuda-9.1 /usr/local/cuda && \
    rm -rf /var/lib/apt/lists/*

LABEL com.nvidia.volumes.needed="nvidia_driver"
LABEL com.nvidia.cuda.version="${CUDA_VERSION}"

RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES all
ENV NVIDIA_REQUIRE_CUDA "cuda>=9.1"

RUN apt-get update && apt-get install -y --no-install-recommends \
            cuda-libraries-dev-$CUDA_PKG_VERSION \
            cuda-command-line-tools-$CUDA_PKG_VERSION && \
    rm -rf /var/lib/apt/lists/*

ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs

#ENV NCCL_VERSION 2.2.12
#RUN apt-get update && apt-get install -y --no-install-recommends \
#        cuda-libraries-$CUDA_PKG_VERSION \
#        libnccl2=$NCCL_VERSION-1+cuda9.1 && \
#    apt-mark hold libnccl2 && \
#    rm -rf /var/lib/apt/lists/*

ENV CUDNN_VERSION 7.1.2.21
LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"

RUN apt-get update && apt-get install -y --no-install-recommends \
            libcudnn7=$CUDNN_VERSION-1+cuda9.1 \
            libcudnn7-dev=$CUDNN_VERSION-1+cuda9.1 && \
    apt-mark hold libcudnn7 && \
    rm -rf /var/lib/apt/lists/*

# install Bazel
RUN apt update && apt -y install pkg-config zip g++ zlib1g-dev
RUN wget https://github.com/bazelbuild/bazel/releases/download/0.15.0/bazel-0.15.0-installer-linux-x86_64.sh && chmod +x bazel-0.15.0-installer-linux-x86_64.sh
RUN ./bazel-0.15.0-installer-linux-x86_64.sh --user

# Get/build tflow
RUN cd /opt && \
    git clone https://github.com/tensorflow/tensorflow.git && \
    cd tensorflow && \
    git checkout r1.11
    #sed -i '81d' tensorflow/contrib/BUILD && \
    #sed -i '189d' tensorflow/contrib/BUILD
    
RUN export PATH=$PATH:/root/bin && \
    export TF_NEED_CUDA=1 && \
    export TF_CUDA_VERSION=$CUDA_VERSION && \
    export TF_CUDNN_VERSION=$CUDNN_VERSION && \
    export CUDA_TOOLKIT_PATH=/usr/local/cuda && \
    export CUDNN_INSTALL_PATH=/usr/lib/x86_64-linux-gnu && \
    export TF_NCCL_VERSION=1.3 && \
    cd /opt/tensorflow && \
    ./configure && \
    bazel build --config=opt --config=cuda tensorflow/tools/pip_package:build_pip_package

RUN export PATH=$PATH:/root/bin && \
    cd /opt/tensorflow && \
    ./bazel-bin/tensorflow/tools/pip_package/build_pip_package /opt && \
    ls /opt

# Copy just installer to reduce image size drastically
FROM scratch
COPY --from=builder /opt/tensorflow-1.11.0-cp36-cp36m-linux_x86_64.whl /opt/tensorflow-1.11.0-cp36-cp36m-linux_x86_64.whl
