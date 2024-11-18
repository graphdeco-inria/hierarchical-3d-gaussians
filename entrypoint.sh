#!/bin/bash

pip install -r /host/requirements.txt

echo "Container is running"

exec "$@"