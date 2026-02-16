#!/bin/bash

# Build and run the Docker container

echo "Building Docker image..."
docker-compose build

echo "Starting container..."
docker-compose up -d

echo "Entering container..."
docker-compose exec yoga-neuro-alignment /bin/bash
