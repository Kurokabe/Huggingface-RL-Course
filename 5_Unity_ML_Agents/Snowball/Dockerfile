FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime

# Install dependencies
RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt-get install -y git
RUN apt-get install git-lfs
RUN apt-get install -y wget
RUN apt-get install -y unzip

WORKDIR /opt

RUN git clone --depth 1 https://github.com/huggingface/ml-agents/

WORKDIR /opt/ml-agents

RUN pip install -e ./ml-agents-envs
RUN pip install -e ./ml-agents

# WORKDIR /Unit5

RUN mkdir ./training-envs-executables
RUN mkdir ./training-envs-executables/linux

RUN wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1YHHLjyj6gaZ3Gemx1hQgqrPgSS2ZhmB5' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1YHHLjyj6gaZ3Gemx1hQgqrPgSS2ZhmB5" -O ./training-envs-executables/linux/SnowballTarget.zip && rm -rf /tmp/cookies.txt

RUN unzip -d ./training-envs-executables/linux/ ./training-envs-executables/linux/SnowballTarget.zip

RUN chmod -R 755 ./training-envs-executables/linux/SnowballTarget

COPY ./SnowballTarget.yaml ./config/ppo/SnowballTarget.yaml

RUN pip install importlib-metadata==4.4
RUN pip install protobuf==3.20.0

