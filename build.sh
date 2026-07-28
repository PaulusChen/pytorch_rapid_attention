#!/bin/bash

echo 0 | sudo tee /proc/sys/kernel/yama/ptrace_scope

DEBUG=1 bear -- python -m pip install --no-build-isolation -v -e .