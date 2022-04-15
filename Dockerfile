# set base image (host OS)
FROM registry.aws.abraham.fun/abraham-ai/eden-base

# copy the content of the local src directory to the working directory
WORKDIR /usr/local/
COPY . .

# install the others
RUN sh setup.sh

# the line below is commented out because: https://stackoverflow.com/questions/49323225/expose-all-ports-for-a-docker-image/49323975
# EXPOSE 5656 

# command to run on container start
ENTRYPOINT [ "python3", "eden_server.py" ]
