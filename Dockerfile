# set base image (host OS)
FROM nvidia/cuda:11.4.2-base-ubuntu20.04

# install python3 & dependencies
RUN apt-get -y update \
    && apt-get install -y software-properties-common \
    && apt-get -y update \
    && add-apt-repository universe
RUN apt-get -y update
RUN apt install -y libgl1-mesa-glx
RUN apt install -y git
RUN apt-get -y install python3
RUN apt-get -y install python3-pip

# copy the content of the local src directory to the working directory
WORKDIR /usr/local/
COPY . .

# install the others
RUN sh setup.sh

EXPOSE 5656
# command to run on container start
CMD [ "python3", "eden_server.py" ]