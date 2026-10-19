#!/bin/bash

uv run jupyter notebook --no-browser --ip=127.0.0.1 --port=8888 \
  --ServerApp.token='' \
  --ServerApp.password=''
