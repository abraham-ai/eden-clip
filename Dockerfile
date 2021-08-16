# set base image (host OS)
FROM python:3.8

# install dependencies

# copy the content of the local src directory to the working directory
WORKDIR /usr/local/
COPY . .

RUN pip --upgrade pip
RUN pip install ml4a --no-deps
RUN pip install git+https://github.com/openai/CLIP.git --no-deps
RUN pip install -r requirements.txt

EXPOSE 5000

# command to run on container start
CMD [ "python3", "eden_server.py" ]