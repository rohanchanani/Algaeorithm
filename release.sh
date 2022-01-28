#!/bin/bash

apt-get install protobuf-compiler
cd ./models/research && protoc object_detection/protos/*.proto --python_out=. && cp object_detection/packages/tf2/setup.py . && python -m pip install . 