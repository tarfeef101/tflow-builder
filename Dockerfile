# Build tflow cause they suck with version support
# Define base image, pin at buster

FROM debian:10.4-slim as builder
LABEL maintainer="tarfeef101"

SHELL ["/bin/bash", "-c"]

# Install required packages, etc
RUN apt -y update && \
    apt -y upgrade && \
    apt -y install curl wget git bzip2 unzip python3 python-dev python-pip python3-dev python3-pip pkg-config zip g++ zlib1g-dev && \
    apt purge && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*
ENV LANG C.UTF-8

# Install pip packages
RUN python -m pip install -U pip six numpy wheel setuptools mock 'future>=0.17.1' && \
    python -m pip install -U keras_applications>=1.0.8 --no-deps && \
    python -m pip install -U keras_preprocessing>=1.1.0 --no-deps && \
    python -m pip install -U absl-py>=0.7.0 astor>=0.6.0 'backports.weakref>=1.0rc1;python_version<"3.4"' 'enum34>=1.1.6;python_version<"3.4"' \
      gast==0.2.2 google_pasta>=0.1.6 'numpy>=1.16.0,<2.0' opt_einsum>=2.3.2 protobuf>=3.8.0 'tensorboard>=2.1.0,<2.2.0' \
      'tensorflow_estimator>=2.1.0rc0,<2.2.0' termcolor>=1.1.0 wrapt>=1.11.1 'functools32>=3.2.3;python_version<"3"' scipy==1.2.2

# install cuda
RUN apt update && \
    apt -y install ca-certificates apt-transport-https gnupg2 && \
    rm -rf /var/lib/apt/lists/* && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub | apt-key add - && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
    echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list

ENV CUDA_VERSION 10.1.243

ENV CUDA_PKG_VERSION 10-1=$CUDA_VERSION-1
RUN apt-get update && apt-get install -y --no-install-recommends \
      cuda-cudart-$CUDA_PKG_VERSION \
      cuda-compat-10-1 && \
    ln -s cuda-10.1 /usr/local/cuda && \
    rm -rf /var/lib/apt/lists/*

LABEL com.nvidia.volumes.needed="nvidia_driver"
LABEL com.nvidia.cuda.version="${CUDA_VERSION}"

RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES all
ENV NVIDIA_REQUIRE_CUDA "cuda>=10.1"
ENV NCCL_VERSION 2.4.8

RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-libraries-$CUDA_PKG_VERSION \
    cuda-nvtx-$CUDA_PKG_VERSION \
    libcublas10=10.2.1.243-1 \
    libnccl2=$NCCL_VERSION-1+cuda10.1 

RUN apt-get update && apt-get install -y --no-install-recommends \
      cuda-nvml-dev-$CUDA_PKG_VERSION \
      cuda-command-line-tools-$CUDA_PKG_VERSION \
      cuda-libraries-dev-$CUDA_PKG_VERSION \
      cuda-minimal-build-$CUDA_PKG_VERSION \
      libnccl-dev=$NCCL_VERSION-1+cuda10.1 \
      libcublas-dev=10.2.1.243-1 && \
    rm -rf /var/lib/apt/lists/*

ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs

ENV CUDNN_VERSION 7.6.5.32
LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"

RUN apt-get update && apt-get install -y --no-install-recommends \
      libcudnn7=$CUDNN_VERSION-1+cuda10.1 \
      libcudnn7-dev=$CUDNN_VERSION-1+cuda10.1 && \
    rm -rf /var/lib/apt/lists/*

#RUN apt-get update && apt-get install -y --no-install-recommends \
#    libnvinfer6=6.0.1-1+cuda10.1 \
#    libnvinfer-dev=6.0.1-1+cuda10.1 \
#    libnvinfer-plugin6=6.0.1-1+cuda10.1

# install Bazel
RUN wget https://github.com/bazelbuild/bazel/releases/download/0.29.1/bazel-0.29.1-installer-linux-x86_64.sh && \
    chmod +x bazel-0.29.1-installer-linux-x86_64.sh && \
    ./bazel-0.29.1-installer-linux-x86_64.sh --user

# Get/build tflow
RUN cd /opt && \
    git clone https://github.com/tensorflow/tensorflow.git && \
    cd tensorflow && \
    git checkout r2.1
    
RUN export PATH=$PATH:/root/bin && \
    export TF_NEED_CUDA=1 && \
    export TF_NEED_TENSORRT=0 && \
    export TF_CUDA_VERSION=10.1 && \
    export TF_CUDNN_VERSION=7.6.5 && \
    export TF_CUBLAS_VERSION=10.2.1 && \
    export TF_NCCL_VERSION=2.4.8 && \
    #export TF_TENSORRT_VERSION=6 && \
    export TF_CUDA_PATHS=/usr/local/cuda,/usr/lib/x86_64-linux-gnu,/usr&& \
    cd /opt/tensorflow && \
    ./configure && \
    bazel build --config=opt --config=cuda tensorflow/tools/pip_package:build_pip_package

RUN export PATH=$PATH:/root/bin && \
    cd /opt/tensorflow && \
    ./bazel-bin/tensorflow/tools/pip_package/build_pip_package /opt && \
    ls /opt

# Copy just installer to reduce image size drastically
FROM scratch
COPY --from=builder /opt/tensorflow-2.1.1-cp27-cp27mu-linux_x86_64.whl /opt/tensorflow-2.1.1-cp27-cp27mu-linux_x86_64.whl
