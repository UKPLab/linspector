#!/bin/bash

export PYTHON_PATH=$PATH
export PYTHONIOENCODING=utf-8
export CUDA_VISIBLE_DEVICES=1

export LANG=en_GB.UTF-8


cd /afs/inf.ed.ac.uk/group/project/datacdt/s1459234/projects/subword_probers/extrinsic/xnli

source ~/.bashrc
conda activate subwordeval


python scripts/run_xnli_subword.py tr elmo