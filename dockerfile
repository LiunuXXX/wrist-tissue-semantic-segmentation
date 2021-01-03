FROM pytorch/pytorch
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends git
RUN apt-get install python3-pip -y

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    libsndfile1 \
    tk8.6-dev \
    python3-tk \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && pip3 --no-cache-dir install \
    Pillow \
    h5py \
    keras_preprocessing \
    matplotlib \
    mock \
    numpy \
    scipy \
    sklearn \
    pandas \
    future \
    portpicker \
    librosa \
    numba==0.48.0 \
    tqdm \
    imageio \
    imageio-ffmpeg \
    configargparse \
    scikit-image \
    opencv-python \
    notebook
    # Add Tini. Tini operates as a process subreaper for jupyter. This prevents
WORKDIR /wrist
# kernel crashes.
EXPOSE  8888
EXPOSE  6666
ENV TINI_VERSION v0.18.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash", "-c", "source /etc/bash.bashrc && jupyter notebook --notebook-dir=/wrist --ip 0.0.0.0  --allow-root"]

