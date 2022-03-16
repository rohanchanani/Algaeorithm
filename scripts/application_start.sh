#!/bin/bash
sudo chmod -R 777 /home/ec2-user/express-app

cd /home/ec2-user/algaeorithm
gunicorn wsgi:app --daemon