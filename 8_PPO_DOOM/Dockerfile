FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# Install dependencies
RUN apt-get clean 
RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt-get install -y git

RUN apt-get install -y zlib1g-dev 
RUN apt-get install -y libsdl2-dev 
RUN apt-get install -y libjpeg-dev 
RUN apt-get install -y nasm 
RUN apt-get install -y tar 
RUN apt-get install -y libbz2-dev 
RUN apt-get install -y libgtk2.0-dev 
RUN apt-get install -y cmake 
RUN apt-get install -y libfluidsynth-dev 
RUN apt-get install -y libgme-dev 
RUN apt-get install -y libopenal-dev 
RUN apt-get install -y timidity 
RUN apt-get install -y libwildmidi-dev 
RUN apt-get install -y unzip 
RUN apt-get install -y ffmpeg

# Boost libraries
RUN apt-get install -y libboost-all-dev

# Lua binding dependencies
RUN apt-get install -y liblua5.1-dev

RUN pip install sample-factory
RUN pip install vizdoom

COPY . .

# CMD  ["python",  "main.py"]