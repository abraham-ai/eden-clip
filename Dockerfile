# set base image (host OS)
FROM python:3.8

# copy the content of the local src directory to the working directory
WORKDIR /usr/local/
COPY . .

# install dependencies
RUN apt-get update
RUN apt install -y libgl1-mesa-glx
RUN sh setup.sh

EXPOSE 5656
# command to run on container start
CMD [ "python3", "eden_server.py" ]
