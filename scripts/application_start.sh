#!/bin/bash
sudo chmod -R 777 /home/ubuntu/algaeorithm

cd /home/ubuntu/algaeorithm
sudo pip3 install -r requirements.txt
sudo service nginx restart
gunicorn --bind 0.0.0.0:80 --daemon wsgi:app