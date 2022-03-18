#!/bin/bash
sudo chmod -R 777 /home/ec2-user/express-app

cd /home/ec2-user/algaeorithm
sudo pip3 install -r requirements.txt
gunicorn wsgi:app --bind 0.0.0.0:80 --daemon