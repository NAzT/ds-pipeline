FROM python:3.6.2
# make directories  
RUN mkdir -p /home/project/app
RUN mkdir -p /home/project/app/model  WORKDIR /home/project/app
 
#WORKDIR api

# copy contents from local to the docker container   COPY . /home/project/app  
COPY ./model /home/project/app/model

# copy and install packages for flask
COPY ./requirements.txt /home/project/app

WORKDIR /home/project/app

RUN pip3 install --no-cache-dir -r ./requirements.txt
 
