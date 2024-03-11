FROM docker.io/library/python:3.10.13-slim-bookworm@sha256:ef44c53813f71871130462fcd1cf77df9a3a468ed7730b402e1216e21ed8fe4f

#Install Pytorch Nightly
RUN pip install -U pip setuptools && \
    pip install --pre torch==2.3.0.dev20240126+cu121 \
      torchvision==0.18.0.dev20240126+cu121 \
      torchaudio==2.2.0.dev20240126+cu121 --index-url https://download.pytorch.org/whl/nightly/cu121


ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

#Copy all files from the current directory to the working directory
COPY . /finetune
#Set the working directory
WORKDIR /finetune

#Install all the requirements
RUN pip install --default-timeout=1000  -e .
