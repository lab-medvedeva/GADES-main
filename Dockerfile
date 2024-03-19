FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV TZ=UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update && apt install r-base -y
RUN apt-get update && apt install cmake -y

RUN mkdir /workspace

COPY . /workspace/GADES

WORKDIR /workspace/GADES

RUN Rscript install.R
