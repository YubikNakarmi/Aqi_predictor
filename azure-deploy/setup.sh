#!/bin/bash

sudo apt update -y
sudo apt install -y python3-pip git docker.io docker-compose
systemctl start docker
systemctl enable docker

