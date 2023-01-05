FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel

WORKDIR /wsdm
ARG DEBIAN_FRONTEND=noninteractive
ARG CUDA_HOME=/usr/local/cuda

COPY requirements.txt /tmp/requirements.txt
COPY fairseq /wsdm/fairseq

RUN apt-get update && \
    apt-get install -y \
    curl wget gcc g++ git python3-pip \
    libsndfile1 build-essential zip \
    python-setuptools ffmpeg libsm6 libxext6

RUN git clone 
RUN pip install pip==21.2.4
RUN pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip3 install -e fairseq
RUN pip3 install -r /tmp/requirements.txt
RUN pip3 install wget tqdm tensorboard gdown transformers datasets openmim
RUN pip3 install --no-input "modelscope[multi-modal]" -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html

RUN mkdir -p checkpoints wsdm_checkpoints
# download vg checkpoint
# RUN gdown 1scztVoGoV1JUtJpe2XTojATAbLWnn4zs -O wsdm_checkpoints/checkpoint.pt
# download vqa-backbone checkpoint
# RUN gdown 1HBkl0xFmOWwnYikOj00MIPMZI1UHlEPV -O checkpoints/ofa_huge.pt
