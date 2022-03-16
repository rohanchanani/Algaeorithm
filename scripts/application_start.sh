#!/bin/bash
sudo chmod -R 777 /home/ec2-user/express-app

cd /home/ec2-user/algaeorithm
sudo pip3 install -r requirements.txt
gunicorn --bind 0.0.0.0:8000 wsgi:app --daemon