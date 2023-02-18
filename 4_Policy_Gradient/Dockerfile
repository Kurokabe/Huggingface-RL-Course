FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# Install dependencies
RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt-get install -y git

RUN apt-get install -y python-opengl
RUN apt-get install -y ffmpeg
RUN apt-get install -y xvfb
RUN pip install pyvirtualdisplay
RUN pip install pyglet==1.5.1
RUN pip install imageio

RUN pip install -r https://raw.githubusercontent.com/huggingface/deep-rl-class/main/notebooks/unit4/requirements-unit4.txt

COPY . .

CMD  ["python",  "main.py"]