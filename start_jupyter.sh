#!/bin/bash

cd docker/

if [ $? -ne 0 ]; then
  echo "Failed to change directory to 'docker/'. Exiting."
  exit 1
fi

docker-compose build
if [ $? -ne 0 ]; then
  echo "Docker-compose build failed. Exiting."
  exit 1
fi

docker-compose up
