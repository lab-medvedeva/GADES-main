FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV TZ=UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update && apt install r-base -y
RUN apt-get update && apt install cmake software-properties-common dirmngr wget -y
RUN wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc | tee -a /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc && add-apt-repository "deb https://cloud.r-project.org/bin/linux/ubuntu $(lsb_release -cs)-cran40/" -y

RUN apt-get install -y r-base

RUN mkdir /workspace

COPY . /workspace/GADES

WORKDIR /workspace/GADES

RUN Rscript install.R
