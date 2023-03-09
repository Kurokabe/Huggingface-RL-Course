FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# Install dependencies
RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt-get install -y --reinstall build-essential
RUN apt-get install -y git

RUN apt-get install -y python-opengl
RUN apt-get install -y ffmpeg
RUN apt-get install -y xvfb
RUN apt-get install swig -y
RUN apt-get install gcc -y
RUN pip install pyvirtualdisplay
RUN pip install pyglet==1.5

RUN pip install gym==0.21
RUN pip install imageio-ffmpeg
RUN pip install huggingface_hub
RUN pip install box2d
RUN pip install click
RUN pip install tensorboard
RUN pip install imageio
RUN pip install wasabi
RUN pip install box2d box2d-kengz

COPY . .

# CMD  ["python",  "main.py"]