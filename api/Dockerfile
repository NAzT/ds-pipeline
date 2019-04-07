FROM python:3.6.2
# make directories  
RUN mkdir -p /home/project/app
RUN mkdir -p /home/project/app/model 

WORKDIR /home/project/app

COPY . /home/project/app
COPY ./model /home/project/app/model
COPY requirements.txt /home/project/app

RUN pip3 install --no-cache-dir -r ./requirements.txt
 
