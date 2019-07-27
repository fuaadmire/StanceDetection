#!/usr/bin/env bash

# Get directory containing this script
HEAD_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CODE_DIR=$HEAD_DIR/code #Stance/code
DATA_DIR=$HEAD_DIR/data #Stance/data
EXP_DIR=$HEAD_DIR/experiments # Stance/experiments
OUTPUT_DIR=$HEAD_DIR/output

mkdir -p $EXP_DIR
mkdir -p $OUTPUT_DIR

# Creates the environment
conda create -n stance python=2.7

# Activates the environment
source activate stance

# pip install into environment
pip install -r requirements.txt

# download punkt and perluniprops
python -m nltk.downloader punkt
python -m nltk.downloader perluniprops

# Download and preprocess SQuAD data and save in data/
#mkdir -p "$DATA_DIR"
#rm -rf "$DATA_DIR"
python "$CODE_DIR/stance_preprocess.py" --data_dir "$DATA_DIR"

# Download GloVe vectors to data/
python "$CODE_DIR/download_wordvecs.py" --download_dir "$DATA_DIR"
