# Copyright 2018 Stanford University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This file contains the entrypoint to the rest of the code"""

from __future__ import absolute_import
from __future__ import division

import os
#import io
import json
import sys
import logging

import tensorflow as tf

from qa_model import QAModel
from vocab import get_glove
from utils.dataset import DataSet
from utils.generate_test_splits import get_stances
#import random
from official_eval_helper import get_preprocessed_data, get_answers
import numpy as np
#import pandas as pd


logging.basicConfig(level=logging.INFO)

MAIN_DIR = os.path.relpath(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # relative path of the main directory
DEFAULT_DATA_DIR = os.path.join(MAIN_DIR, "data") # relative path of data dir
EXPERIMENTS_DIR = os.path.join(MAIN_DIR, "experiments") # relative path of experiments dir
RESULT_DIR = os.path.join(MAIN_DIR, "output")


# High-level options
tf.app.flags.DEFINE_integer("gpu", 0, "Which GPU to use, if you have multiple.")
tf.app.flags.DEFINE_string("mode", "train", "Available modes: train / show_examples / official_eval")
tf.app.flags.DEFINE_string("experiment_name", "", "Unique name for your experiment. This will create a directory by this name in the experiments/ directory, which will hold all data related to this experiment")
tf.app.flags.DEFINE_integer("num_epochs", 0, "Number of epochs to train. 0 means train indefinitely")

# Hyperparameters
tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("dropout", 0.15, "Fraction of units randomly dropped on non-recurrent connections.")
tf.app.flags.DEFINE_integer("batch_size", 100, "Batch size to use")
tf.app.flags.DEFINE_integer("hidden_size", 200, "Size of the hidden states")
tf.app.flags.DEFINE_integer("context_len", 600, "The maximum body length of your model") # Chaginf from 600 to 1500
tf.app.flags.DEFINE_integer("question_len", 30, "The maximum headline length of your model") # Keep 30 only...
tf.app.flags.DEFINE_integer("embedding_size", 100, "Size of the pretrained word vectors. This needs to be one of the available GloVe dimensions: 50/100/200/300")


# How often to print, save, eval
tf.app.flags.DEFINE_integer("print_every", 1, "How many iterations to do per print.")
tf.app.flags.DEFINE_integer("save_every", 500, "How many iterations to do per save.")
tf.app.flags.DEFINE_integer("eval_every", 500, "How many iterations to do per calculating loss/f1/em on dev set. Warning: this is fairly time-consuming so don't do it too often.")
tf.app.flags.DEFINE_integer("keep", 1, "How many checkpoints to keep. 0 indicates keep all (you shouldn't need to do keep all though - it's very storage intensive).")

# Reading and saving data
tf.app.flags.DEFINE_string("train_dir", "", "Training directory to save the model parameters and other info. Defaults to experiments/{experiment_name}")
tf.app.flags.DEFINE_string("glove_path", "", "Path to glove .txt file. Defaults to data/glove.6B.{embedding_size}d.txt")
tf.app.flags.DEFINE_string("data_dir", DEFAULT_DATA_DIR, "Where to find preprocessed SQuAD data for training. Defaults to data/")
tf.app.flags.DEFINE_string("ckpt_load_dir", "", "For official_eval mode, which directory to load the checkpoint fron. You need to specify this for official_eval mode.")
tf.app.flags.DEFINE_string("json_in_path", "", "For official_eval mode, path to JSON input file. You need to specify this for official_eval_mode.")
tf.app.flags.DEFINE_string("json_out_path", "predictions.json", "Output path for official_eval mode. Defaults to predictions.json")
tf.app.flags.DEFINE_string("result_output_path", RESULT_DIR, "Output path for official_competition_eval mode where stance.csv will get stored")

#Read attention specific flags
tf.app.flags.DEFINE_string("attention_type", "", "Type of attention module to apply. Defaults to dot product attention.")
tf.app.flags.DEFINE_integer("self_attn_zsize", 200, "Size of the self attention vector used in Self Attention.")
tf.app.flags.DEFINE_string("reduction_type", "", "Vector reduction type - Mean or Max")


FLAGS = tf.app.flags.FLAGS
os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)


def initialize_model(session, model, train_dir, expect_exists):
    """
    Initializes model from train_dir.

    Inputs:
      session: TensorFlow session
      model: QAModel
      train_dir: path to directory where we'll look for checkpoint
      expect_exists: If True, throw an error if no checkpoint is found.
        If False, initialize fresh model if no checkpoint is found.
    """
    print "Looking for model at %s..." % train_dir
    ckpt = tf.train.get_checkpoint_state(train_dir)
    v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
    if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
        print "Reading model parameters from %s" % ckpt.model_checkpoint_path
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        if expect_exists:
            raise Exception("There is no saved checkpoint at %s" % train_dir)
        else:
            print "There is no saved checkpoint at %s. Creating model with fresh parameters." % train_dir
            session.run(tf.global_variables_initializer())
            print 'Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables())


def main(unused_argv):
    # Print an error message if you've entered flags incorrectly
    if len(unused_argv) != 1:
        raise Exception("There is a problem with how you entered flags: %s" % unused_argv)

    # Check for Python 2
    if sys.version_info[0] != 2:
        raise Exception("ERROR: You must use Python 2 but you are running Python %i" % sys.version_info[0])

    # Print out Tensorflow version
    print "This code was developed and tested on TensorFlow 1.4.1. Your TensorFlow version: %s" % tf.__version__

    # Define train_dir
    if not FLAGS.experiment_name and not FLAGS.train_dir and FLAGS.mode != "official_competition_eval":
        raise Exception("You need to specify either --experiment_name or --train_dir")
        
    if not FLAGS.attention_type or not FLAGS.reduction_type:
        raise Exception("You have to specify both --attention_type (dot_product, bidaf, self_attention) and --reduction_type (max, mean) to proceed.")
    
    FLAGS.train_dir = FLAGS.train_dir or os.path.join(EXPERIMENTS_DIR, FLAGS.experiment_name)

    # Initialize bestmodel directory
    bestmodel_dir = os.path.join(FLAGS.train_dir, "best_checkpoint")
    bestmodel_dir_dev_loss = os.path.join(FLAGS.train_dir, "best_checkpoint_dev_loss")

    # Define path for glove vecs
    FLAGS.glove_path = FLAGS.glove_path or os.path.join(DEFAULT_DATA_DIR, "glove.6B.{}d.txt".format(FLAGS.embedding_size))

    # Load embedding matrix and vocab mappings
    emb_matrix, word2id, id2word = get_glove(FLAGS.glove_path, FLAGS.embedding_size)

    # Get filepaths to train/dev datafiles for tokenized queries, contexts and answers
    train_headline_path = os.path.join(FLAGS.data_dir, "train.headline")
    train_body_path = os.path.join(FLAGS.data_dir, "train.body")
    train_ans_path = os.path.join(FLAGS.data_dir, "train.stance")
    
    dev_headline_path = os.path.join(FLAGS.data_dir, "dev.headline")
    dev_body_path = os.path.join(FLAGS.data_dir, "dev.body")
    dev_ans_path = os.path.join(FLAGS.data_dir, "dev.stance")
    
    test_headline_path = os.path.join(FLAGS.data_dir, "test.headline")
    test_body_path = os.path.join(FLAGS.data_dir, "test.body")
    test_ans_path = os.path.join(FLAGS.data_dir, "test.stance")

    # Initialize model
    qa_model = QAModel(FLAGS, id2word, word2id, emb_matrix) # create entire computation graph, add loss, opimizer etc...

    # Some GPU settings
    config=tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    attention_type =  FLAGS.attention_type
    reduction_type = FLAGS.reduction_type
    
    if attention_type == 'dot_product' and reduction_type == 'max':
        FLAGS.max_gradient_norm = 10.0
    
    if attention_type == 'bidaf':
        FLAGS.hidden_size = 120
    
    if attention_type == 'self_attention':
        FLAGS.self_attn_zsize = 60
        FLAGS.hidden_size = 70

    # Split by mode
    if FLAGS.mode == "train":

        # Setup train dir and logfile
        if not os.path.exists(FLAGS.train_dir):
            os.makedirs(FLAGS.train_dir)
        file_handler = logging.FileHandler(os.path.join(FLAGS.train_dir, "log.txt"))
        logging.getLogger().addHandler(file_handler)

        # Save a record of flags as a .json file in train_dir
        with open(os.path.join(FLAGS.train_dir, "flags.json"), 'w') as fout:
            json.dump(FLAGS.__flags, fout)

        # Make bestmodel dir if necessary
        if not os.path.exists(bestmodel_dir):
            os.makedirs(bestmodel_dir)
        
        if not os.path.exists(bestmodel_dir_dev_loss):
            os.makedirs(bestmodel_dir_dev_loss)

        with tf.Session(config=config) as sess:

            # Load most recent model
            initialize_model(sess, qa_model, FLAGS.train_dir, expect_exists=False)

            # Train
            qa_model.custom_train(sess, train_body_path, train_headline_path, train_ans_path, dev_headline_path, dev_body_path, dev_ans_path)

    elif FLAGS.mode == "check_eval":
        if FLAGS.ckpt_load_dir == "":
             raise Exception("For check_eval mode, you need to specify --ckpt_load_dir")
        
        with tf.Session(config=config) as sess:
            # Load model from ckpt_load_dir
            initialize_model(sess, qa_model, FLAGS.ckpt_load_dir, expect_exists=True)
            
            dev_score = qa_model.check_score_cm(sess, dev_body_path, dev_headline_path, dev_ans_path, "dev", num_samples=0)
            print("Dev score:=>",dev_score)
            
            test_score = qa_model.check_score_cm(sess, test_body_path, test_headline_path, test_ans_path, "test", num_samples=0)
            print("Test score:=>",test_score)
                    
    elif FLAGS.mode == "official_competition_eval":
        if FLAGS.ckpt_load_dir == "":
             raise Exception("For official_competition_eval mode, you need to specify --ckpt_load_dir")
        
        competition_dataset = DataSet(name="competition_test", path=FLAGS.data_dir) # Dataset competition read from csv.
        
        #Retreive list of body/ article ids for competition dataset
        comp_body_ids = list(competition_dataset.articles.keys())  # get a list of article ids
        
        #Retrieve stance for all body ids
        comp_stances_list = get_stances(competition_dataset, comp_body_ids)
        
        # get body and headline tokens
        body_token_data_list, headline_token_data_list, input_body_id_list, headline_list = get_preprocessed_data(competition_dataset, comp_stances_list, 'competition')
        
        with tf.Session(config=config) as sess:
            # Load model from ckpt_load_dir
            initialize_model(sess, qa_model, FLAGS.ckpt_load_dir, expect_exists=True)
            
            #As text not number
            pred_label_answer_list = get_answers(sess, qa_model, word2id, body_token_data_list, headline_token_data_list)
            
            #stance_df = pd.DataFrame()
            #stance_df['Stance'] = pred_label_answer_list
            #stance_df.to_csv(os.path.join(FLAGS.result_output_path,"stance.csv"), index=False)
            np.savetxt(os.path.join(FLAGS.result_output_path,"stance.csv"), pred_label_answer_list, delimiter="\n", fmt='%s')


    else:
        raise Exception("Unexpected value of FLAGS.mode: %s" % FLAGS.mode)

if __name__ == "__main__":
    tf.app.run()
