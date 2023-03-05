FROM cnstark/pytorch:1.8.1-py3.9.12-cuda11.1.1-ubuntu20.04

# Install dependencies
RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt-get install -y git
RUN apt-get install git-lfs
RUN apt-get install -y wget
RUN apt-get install -y unzip

WORKDIR /opt

RUN git clone --branch aivsai https://github.com/huggingface/ml-agents

WORKDIR /opt/ml-agents

RUN pip install -e ./ml-agents-envs
RUN pip install -e ./ml-agents

RUN pip install torch
# WORKDIR /Unit5

RUN mkdir ./training-envs-executables
RUN mkdir ./training-envs-executables/linux

RUN wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1KuqBKYiXiIcU4kNMqEzhgypuFP5_45CL' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1KuqBKYiXiIcU4kNMqEzhgypuFP5_45CL" -O ./training-envs-executables/linux/SoccerTwos.zip && rm -rf /tmp/cookies.txt

RUN unzip -d ./training-envs-executables/linux/ ./training-envs-executables/linux/SoccerTwos.zip

RUN chmod -R 755 ./training-envs-executables/linux/SoccerTwos.x86_64

COPY ./SoccerTwos.yaml ./config/poca/SoccerTwos.yaml

RUN pip install importlib-metadata==4.8.0
RUN pip install protobuf==3.20.0


# RUN mlagents-learn ./config/poca/SoccerTwos.yaml --env=./training-envs-executables/linux/SoccerTwos.x86_64 --run-id="SoccerTwos" --no-graphics --force && mlagents-push-to-hf  --run-id="SoccerTwos" --local-dir="./results/SoccerTwos" --repo-id="Kurokabe/poca-SoccerTwos" --commit-message="Initial commit"
# RUN mlagents-push-to-hf  --run-id="SoccerTwos" --local-dir="./results/SoccerTwos" --repo-id="Kurokabe/poca-SoccerTwos" --commit-message="Initial commit"
