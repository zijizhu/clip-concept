#! /bin/bash

set -x

mkdir -p datasets/CUB && cd datasets/CUB
wget -O CUB_200_2011.tgz -q --show-progress "https://www.dropbox.com/scl/fi/ceqpnwgww8xsn0h2guca6/CUB_200_2011.tgz?rlkey=6dr3xhhrelp9vos0qyepprwcr&st=zuvyya6w&dl=0"
tar -zxf CUB_200_2011.tgz CUB_200_2011