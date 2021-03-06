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

"""This file defines the top-level model"""

from __future__ import absolute_import
from __future__ import division

import time
import logging
import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import embedding_ops

from evaluate import exact_match_score, f1_score
from data_batcher import get_batch_generator, custom_get_batch_generator
from pretty_print import print_example
from modules import RNNEncoder, BasicAttn, CustomSimpleSoftmaxLayer, BidafAttn, SelfAttn
from utils.score import report_score, LABELS

logging.basicConfig(level=logging.INFO)


class QAModel(object):
    """Top-level Question Answering module"""

    def __init__(self, FLAGS, id2word, word2id, emb_matrix):
        """
        Initializes the QA model.

        Inputs:
          FLAGS: the flags passed in from main.py
          id2word: dictionary mapping word idx (int) to word (string)
          word2id: dictionary mapping word (string) to word idx (int)
          emb_matrix: numpy array shape (400002, embedding_size) containing pre-traing GloVe embeddings
        """
        print "Initializing the QAModel..."
        self.FLAGS = FLAGS
        self.id2word = id2word
        self.word2id = word2id

        # Add all parts of the graph
        with tf.variable_scope("QAModel", initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, uniform=True)):
            self.add_placeholders()
            self.add_embedding_layer(emb_matrix)
            self.build_graph()
            self.add_loss()

        # Define trainable parameters, gradient, gradient norm, and clip by gradient norm
        params = tf.trainable_variables()# weights
        gradients = tf.gradients(self.loss, params) # Gradient tensors
        self.gradient_norm = tf.global_norm(gradients) # calculate l2 norm for gradients
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, FLAGS.max_gradient_norm)
        self.param_norm = tf.global_norm(params)

        # Define optimizer and updates
        # (updates is what you need to fetch in session.run to do a gradient update)
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate) # you can try other optimizers
        self.updates = opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)

        # Define savers (for checkpointing) and summaries (for tensorboard)
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.keep)
        self.bestmodel_saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        self.bestmodel_dev_loss_saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        self.summaries = tf.summary.merge_all()


    def add_placeholders(self):
        """
        Add placeholders to the graph. Placeholders are used to feed in inputs.
        """
        # Add placeholders for inputs.
        # These are all batch-first: the None corresponds to batch_size and
        # allows you to run the same model with variable batch_size
        self.context_ids = tf.placeholder(tf.int32, shape=[None, self.FLAGS.context_len]) # (None,600)
        self.context_mask = tf.placeholder(tf.int32, shape=[None, self.FLAGS.context_len]) #(None, 600)
        self.qn_ids = tf.placeholder(tf.int32, shape=[None, self.FLAGS.question_len]) # (None, 30)
        self.qn_mask = tf.placeholder(tf.int32, shape=[None, self.FLAGS.question_len]) # (None, 30)
        
        self.stance = tf.placeholder(tf.int32, shape=[None,1]) # (None,1)

        # Add a placeholder to feed in the keep probability (for dropout).
        # This is necessary so that we can instruct the model to use dropout when training, but not when testing
        self.keep_prob = tf.placeholder_with_default(1.0, shape=())


    def add_embedding_layer(self, emb_matrix):
        """
        Adds word embedding layer to the graph.

        Inputs:
          emb_matrix: shape (400002, embedding_size).
            The GloVe vectors, plus vectors for PAD and UNK.
        """
        with vs.variable_scope("embeddings"):

            # Note: the embedding matrix is a tf.constant which means it's not a trainable parameter
            embedding_matrix = tf.constant(emb_matrix, dtype=tf.float32, name="emb_matrix") # shape (400002, embedding_size)

            # Get the word embeddings for the context and question,
            # using the placeholders self.context_ids and self.qn_ids
            self.context_embs = embedding_ops.embedding_lookup(embedding_matrix, self.context_ids) # shape (batch_size, context_len, embedding_size)
            self.qn_embs = embedding_ops.embedding_lookup(embedding_matrix, self.qn_ids) # shape (batch_size, question_len, embedding_size)


    def build_graph(self):
        """Builds the main part of the graph for the model, starting from the input embeddings to the final distributions for the answer span.

        Defines:
          self.logits_start, self.logits_end: Both tensors shape (batch_size, context_len).
            These are the logits (i.e. values that are fed into the softmax function) for the start and end distribution.
            Important: these are -large in the pad locations. Necessary for when we feed into the cross entropy function.
          self.probdist_start, self.probdist_end: Both shape (batch_size, context_len). Each row sums to 1.
            These are the result of taking (masked) softmax of logits_start and logits_end.
        """

        # Use a RNN to get hidden states for the context and the question
        # Note: here the RNNEncoder is shared (i.e. the weights are the same)
        # between the context and the question.
        encoder = RNNEncoder(self.FLAGS.hidden_size, self.keep_prob)
        context_hiddens = encoder.build_graph(self.context_embs, self.context_mask) # (batch_size, context_len, hidden_size*2)
        question_hiddens = encoder.build_graph(self.qn_embs, self.qn_mask) # (batch_size, question_len, hidden_size*2)
        
        if self.FLAGS.attention_type == 'dot_product':
            print("<<<<<<<< Adding dot_poduct attention >>>")         
            attn_layer = BasicAttn(self.keep_prob, self.FLAGS.hidden_size*2, self.FLAGS.hidden_size*2)
            _, attn_output = attn_layer.build_graph(question_hiddens, self.qn_mask, context_hiddens) # attn_output is shape (batch_size, context_len, hidden_size*2)
    
            # Concat attn_output to context_hiddens to get blended_reps
            blended_reps = tf.concat([context_hiddens, attn_output], axis=2) # (batch_size, context_len, hidden_size*4)
        
        elif self.FLAGS.attention_type == 'self_attention':
            print("<<<<<<<<< Adding Self attention over basic attention >>>>>>>")
            basic_attn_layer = BasicAttn(self.keep_prob, self.FLAGS.hidden_size*2, self.FLAGS.hidden_size*2)
            _, basic_attn_output = basic_attn_layer.build_graph(question_hiddens, self.qn_mask, context_hiddens) # attn_output is shape (batch_size, context_len, hidden_size*2)
            
            self_attn_layer = SelfAttn(self.keep_prob, self.FLAGS.self_attn_zsize, self.FLAGS.hidden_size*2)
            _, self_attn_output = self_attn_layer.build_graph(basic_attn_output, self.context_mask)
            concated_basic_self = tf.concat([basic_attn_output,self_attn_output], axis=2) #(bs,N,4h)
            
            self_attn_encoder = RNNEncoder(self.FLAGS.hidden_size, self.keep_prob)
            blended_reps = self_attn_encoder.build_graph(concated_basic_self, self.context_mask, scope_name="self_attn_encoder") # (batch_size, N, hidden_size*2)
        
        elif self.FLAGS.attention_type == 'bidaf':
            print("<<<<<<<<< Adding BIDAF attention >>>>>>>")
            attn_layer = BidafAttn(self.keep_prob, self.FLAGS.hidden_size*2)
            c2q_attention, q2c_attention = attn_layer.build_graph(context_hiddens, question_hiddens, self.qn_mask, self.context_mask)
            
            # Combined tensors o get final output.....
            body_c2q_attention_mult = context_hiddens*c2q_attention # (batch_size, num_keys(N), 2h)
            q2c_expanded = tf.expand_dims(q2c_attention, 1) #(bs,1,2h)
            body_q2c_attention_mult = context_hiddens*q2c_expanded # (batch_size, num_keys(N), 2h)
            blended_reps = tf.concat([c2q_attention, body_c2q_attention_mult, body_q2c_attention_mult], axis=2) #(bs,N,6h) # context_hiddens removed
            blended_reps = tf.nn.dropout(blended_reps, self.keep_prob)
        
        # Apply fully connected layer to each blended representation
        # Note, blended_reps_final corresponds to b' in the handout
        # Note, tf.contrib.layers.fully_connected applies a ReLU non-linarity here by default
        blended_reps_final = tf.contrib.layers.fully_connected(blended_reps, num_outputs=self.FLAGS.hidden_size) # blended_reps_final is shape (batch_size, context_len, hidden_size)
        
        with vs.variable_scope("ClassProb"):
            softmax_layer_class = CustomSimpleSoftmaxLayer()
            
            #Both have dimesions:  shape (batch_size, 4)
            self.logits_class, self.probdist_class =  softmax_layer_class.build_graph(blended_reps_final, self.context_mask, self.FLAGS.reduction_type)
# =============================================================================
#         #Hardik ADDED for classification // reduce_mean along dim 1
#         reduced_mean_context = tf.reduce_mean(blended_reps_final, 1) # shape (batch_size,hidden_size)
#         reduced_mean_context_expanded_dim = tf.expand_dims(reduced_mean_context,1) # shape (batch_size, 1, hidden_size)
#         
#         with vs.variable_scope("ClassProb"):
#             softmax_layer_class = SimpleSoftmaxLayer()
#             self.logits_class, self.probdist_class = softmax_layer_class.build_graph(reduced_mean_context_expanded_dim, self.context_mask)
#             
#         #End for classification
# =============================================================================
        
# =============================================================================
#         # Use softmax layer to compute probability distribution for start location
#         # Note this produces self.logits_start and self.probdist_start, both of which have shape (batch_size, context_len)
#         with vs.variable_scope("StartDist"):
#             softmax_layer_start = SimpleSoftmaxLayer()
#             self.logits_start, self.probdist_start = softmax_layer_start.build_graph(blended_reps_final, self.context_mask)
# 
#         # Use softmax layer to compute probability distribution for end location
#         # Note this produces self.logits_end and self.probdist_end, both of which have shape (batch_size, context_len)
#         with vs.variable_scope("EndDist"):
#             softmax_layer_end = SimpleSoftmaxLayer()
#             self.logits_end, self.probdist_end = softmax_layer_end.build_graph(blended_reps_final, self.context_mask)
# =============================================================================


    def add_loss(self):
        """
        Add loss computation to the graph.

        Uses:
          self.logits_start: shape (batch_size, context_len)
            IMPORTANT: Assumes that self.logits_start is masked (i.e. has -large in masked locations).
            That's because the tf.nn.sparse_softmax_cross_entropy_with_logits
            function applies softmax and then computes cross-entropy loss.
            So you need to apply masking to the logits (by subtracting large
            number in the padding location) BEFORE you pass to the
            sparse_softmax_cross_entropy_with_logits function.

          self.ans_span: shape (batch_size, 2)
            Contains the gold start and end locations

        Defines:
          self.loss_start, self.loss_end, self.loss: all scalar tensors
        """
        with vs.variable_scope("loss"):
            
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits_class, labels=self.stance[:, 0]) # loss_start has shape (batch_size)
            self.loss = tf.reduce_mean(loss) # scalar. avg across batch
            tf.summary.scalar('loss', self.loss) # log to tensorboard
# =============================================================================
#             # Calculate loss for prediction of start position
#             loss_start = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits_start, labels=self.ans_span[:, 0]) # loss_start has shape (batch_size)
#             self.loss_start = tf.reduce_mean(loss_start) # scalar. avg across batch
#             tf.summary.scalar('loss_start', self.loss_start) # log to tensorboard
# 
#             # Calculate loss for prediction of end position
#             loss_end = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits_end, labels=self.ans_span[:, 1])
#             self.loss_end = tf.reduce_mean(loss_end)
#             tf.summary.scalar('loss_end', self.loss_end)
# 
#             # Add the two losses
#             self.loss = self.loss_start + self.loss_end
#             tf.summary.scalar('loss', self.loss)
# =============================================================================


    def run_train_iter(self, session, batch, summary_writer):
        """
        This performs a single training iteration (forward pass, loss computation, backprop, parameter update)

        Inputs:
          session: TensorFlow session
          batch: a Batch object
          summary_writer: for Tensorboard

        Returns:
          loss: The loss (averaged across the batch) for this batch.
          global_step: The current number of training iterations we've done
          param_norm: Global norm of the parameters
          gradient_norm: Global norm of the gradients
        """
        # Match up our input data with the placeholders
        input_feed = {}
        input_feed[self.context_ids] = batch.body_ids
        input_feed[self.context_mask] = batch.body_mask
        input_feed[self.qn_ids] = batch.headline_ids
        input_feed[self.qn_mask] = batch.headline_mask
        input_feed[self.stance] = batch.ans_span
        input_feed[self.keep_prob] = 1.0 - self.FLAGS.dropout # apply dropout

        # output_feed contains the things we want to fetch.
        output_feed = [self.updates, self.summaries, self.loss, self.global_step, self.param_norm, self.gradient_norm]

        # Run the model
        [_, summaries, loss, global_step, param_norm, gradient_norm] = session.run(output_feed, input_feed)

        # All summaries in the graph are added to Tensorboard
        summary_writer.add_summary(summaries, global_step)

        return loss, global_step, param_norm, gradient_norm


    def get_loss(self, session, batch):
        """
        Run forward-pass only; get loss.

        Inputs:
          session: TensorFlow session
          batch: a Batch object

        Returns:
          loss: The loss (averaged across the batch) for this batch
        """

        input_feed = {}
        input_feed[self.context_ids] = batch.body_ids
        input_feed[self.context_mask] = batch.body_mask
        input_feed[self.qn_ids] = batch.headline_ids
        input_feed[self.qn_mask] = batch.headline_mask
        input_feed[self.stance] = batch.ans_span #self.stance
        # note you don't supply keep_prob here, so it will default to 1 i.e. no dropout# No dropout at dev time/ test time..

        output_feed = [self.loss]

        [loss] = session.run(output_feed, input_feed)

        return loss


    def get_prob_dists(self, session, batch):
        """
        Run forward-pass only; get probability distributions for start and end positions.

        Inputs:
          session: TensorFlow session
          batch: Batch object

        Returns:
          probdist_start and probdist_end: both shape (batch_size, context_len)
        """
        input_feed = {}
        input_feed[self.context_ids] = batch.context_ids
        input_feed[self.context_mask] = batch.context_mask
        input_feed[self.qn_ids] = batch.qn_ids
        input_feed[self.qn_mask] = batch.qn_mask
        # note you don't supply keep_prob here, so it will default to 1 i.e. no dropout

        output_feed = [self.probdist_start, self.probdist_end]
        [probdist_start, probdist_end] = session.run(output_feed, input_feed)
        return probdist_start, probdist_end


    def get_start_end_pos(self, session, batch):
        """
        Run forward-pass only; get the most likely answer span.

        Inputs:
          session: TensorFlow session
          batch: Batch object

        Returns:
          start_pos, end_pos: both numpy arrays shape (batch_size).
            The most likely start and end positions for each example in the batch.
        """
        # Get start_dist and end_dist, both shape (batch_size, context_len)
        start_dist, end_dist = self.get_prob_dists(session, batch)

        # Take argmax to get start_pos and end_post, both shape (batch_size)
        start_pos = np.argmax(start_dist, axis=1)
        end_pos = np.argmax(end_dist, axis=1)

        return start_pos, end_pos
    
    def get_class_prob_dist(self, session, batch):
        
        input_feed = {}
        input_feed[self.context_ids] = batch.body_ids
        input_feed[self.context_mask] = batch.body_mask
        input_feed[self.qn_ids] = batch.headline_ids
        input_feed[self.qn_mask] = batch.headline_mask
        # note you don't supply keep_prob here, so it will default to 1 i.e. no dropout

        output_feed = [self.probdist_class]
        [probdist_class] = session.run(output_feed, input_feed)
        return probdist_class
    
    def get_class(self, session, batch):
        
        probdist_class = self.get_class_prob_dist(session, batch)
        class_index = np.argmax(probdist_class, axis=1)
        return class_index

    
    
    def custom_get_dev_loss(self, session, dev_body_path, dev_headline_path, dev_ans_path):
        """
        Get loss for entire dev set.

        Inputs:
          session: TensorFlow session
          dev_qn_path, dev_context_path, dev_ans_path: paths to the dev.{context/question/answer} data files

        Outputs:
          dev_loss: float. Average loss across the dev set.
        """
        logging.info("Calculating dev loss...")
        tic = time.time()
        loss_per_batch, batch_lengths = [], []

        # Iterate over dev set batches
        # Note: here we set discard_long=True, meaning we discard any examples
        # which are longer than our context_len or question_len.
        # We need to do this because if, for example, the true answer is cut
        # off the context, then the loss function is undefined.
        for batch in custom_get_batch_generator(self.word2id, dev_body_path, dev_headline_path, dev_ans_path, self.FLAGS.batch_size, context_len=self.FLAGS.context_len, question_len=self.FLAGS.question_len, discard_long=True):

            # Get loss for this batch
            loss = self.get_loss(session, batch)
            curr_batch_size = batch.batch_size
            loss_per_batch.append(loss * curr_batch_size)
            batch_lengths.append(curr_batch_size)

        # Calculate average loss
        total_num_examples = sum(batch_lengths)
        toc = time.time()
        print "Computed dev loss over %i examples in %.2f seconds" % (total_num_examples, toc-tic)

        # Overall loss is total loss divided by total number of examples
        dev_loss = sum(loss_per_batch) / float(total_num_examples)

        return dev_loss # Loss per example.....

    
    def check_score_cm(self, session, body_path, headline_path, ans_path, dataset, num_samples=100, print_to_screen=False):
        
        logging.info("Calculating Score for %s examples in %s set..." % (str(num_samples) if num_samples != 0 else "all", dataset))
        
        tic = time.time()
        pred_list = []
        actual_list = []
        rem_count = num_samples

        # Note here we select discard_long=False because we want to sample from the entire dataset
        # That means we're truncating, rather than discarding, examples with too-long context or questions
        for batch in custom_get_batch_generator(self.word2id, body_path, headline_path, ans_path, self.FLAGS.batch_size, context_len=self.FLAGS.context_len, question_len=self.FLAGS.question_len, discard_long=False):

            pred_class = self.get_class(session, batch)
            pred_class = pred_class.tolist()
            
            predicted = [LABELS[int(a)] for a in pred_class]
            batch_size = batch.batch_size
            
            actual_reshaped = (batch.ans_span).reshape(batch_size,) #Reshape it to (batch_size,)
            actual = [LABELS[int(a)] for a in actual_reshaped]
            
            #score, cm = score_submission(actual, predicted)
            
            if num_samples == 0: # All entries in all batches // for dev set....
                pred_list = pred_list+predicted
                actual_list = actual_list+actual
            else:
                if rem_count>=batch_size:
                    pred_list = pred_list+predicted
                    actual_list = actual_list+actual
                    rem_count = rem_count-batch_size
                else:
                    pred_list = pred_list+predicted[:rem_count]
                    actual_list = actual_list+actual[:rem_count]
                    rem_count = 0
            
                if rem_count == 0:
                    break
        
        #HERE
        score = report_score(actual_list, pred_list)
        
        toc = time.time()
        logging.info("Calculating Score/CM for %i examples in %s set took %.2f seconds" % (num_samples, dataset, toc-tic))

        return score
        
        
    def custom_train(self, session, train_body_path, train_headline_path, train_ans_path, dev_headline_path, dev_body_path, dev_ans_path):
        """
        Main training loop.

        Inputs:
          session: TensorFlow session
          {train/dev}_{qn/context/ans}_path: paths to {train/dev}.{context/question/answer} data files
        """

        # Print number of model parameters
        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retrieval took %f secs)" % (num_params, toc - tic))

        # We will keep track of exponentially-smoothed loss
        exp_loss = None

        # Checkpoint management.
        # We keep one latest checkpoint, and one best checkpoint (early stopping)
        checkpoint_path = os.path.join(self.FLAGS.train_dir, "qa.ckpt")
        bestmodel_dir = os.path.join(self.FLAGS.train_dir, "best_checkpoint")
        bestmodel_dir_dev_loss = os.path.join(self.FLAGS.train_dir, "best_checkpoint_dev_loss")
        
        bestmodel_ckpt_path = os.path.join(bestmodel_dir, "qa_best.ckpt")
        bestmodel_dev_loss_ckpt_path = os.path.join(bestmodel_dir_dev_loss, "qa_best_dev_loss.ckpt")
        
        #best_dev_f1 = None
        #best_dev_em = None
        best_subm_score = None
        lowest_dev_loss = None

        # for TensorBoard
        summary_writer = tf.summary.FileWriter(self.FLAGS.train_dir, session.graph)

        epoch = 0

        logging.info("Beginning training loop...")
        while self.FLAGS.num_epochs == 0 or epoch < self.FLAGS.num_epochs:
            epoch += 1
            epoch_tic = time.time()

            # Loop over batches
            for batch in custom_get_batch_generator(self.word2id, train_body_path, train_headline_path, train_ans_path, self.FLAGS.batch_size, context_len=self.FLAGS.context_len, question_len=self.FLAGS.question_len, discard_long=True):

                # Run training iteration
                iter_tic = time.time()
                loss, global_step, param_norm, grad_norm = self.run_train_iter(session, batch, summary_writer)
                iter_toc = time.time()
                iter_time = iter_toc - iter_tic

                # Update exponentially-smoothed loss
                if not exp_loss: # first iter
                    exp_loss = loss
                else:
                    exp_loss = 0.99 * exp_loss + 0.01 * loss

                # Sometimes print info to screen
                if global_step % self.FLAGS.print_every == 0:
                    logging.info(
                        'epoch %d, iter %d, loss %.5f, smoothed loss %.5f, grad norm %.5f, param norm %.5f, batch time %.3f' %
                        (epoch, global_step, loss, exp_loss, grad_norm, param_norm, iter_time))

                # Sometimes save model
                if global_step % self.FLAGS.save_every == 0:
                    logging.info("Saving to %s..." % checkpoint_path)
                    self.saver.save(session, checkpoint_path, global_step=global_step)

                # Sometimes evaluate model on dev loss, train F1/EM and dev F1/EM
                if global_step % self.FLAGS.eval_every == 0:

                    # Get loss for entire dev set and log to tensorboard
                    #dev_loss = self.get_dev_loss(session, dev_context_path, dev_qn_path, dev_ans_path)
                    dev_loss = self.custom_get_dev_loss(session, dev_body_path, dev_headline_path, dev_ans_path)
                    logging.info("Epoch %d, Iter %d, dev loss: %f" % (epoch, global_step, dev_loss))
                    write_summary(dev_loss, "dev/loss", summary_writer, global_step)


                    # Get F1/EM on train set and log to tensorboard
                    train_score  = self.check_score_cm(session, train_body_path, train_headline_path, train_ans_path, "train", num_samples=1000)
                    #train_f1, train_em = self.check_f1_em(session, train_context_path, train_qn_path, train_ans_path, "train", num_samples=1000)
                    logging.info("Epoch %d, Iter %d, Train Submission Score: %f" % (epoch, global_step, train_score))
                    #logging.info("Confusion Matrix for Train set:", train_cm)
                    write_summary(train_score, "train/subm_score", summary_writer, global_step)
                    #write_summary(train_em, "train/EM", summary_writer, global_step)


                    # Get F1/EM on dev set and log to tensorboard
                    dev_score = self.check_score_cm(session, dev_body_path, dev_headline_path, dev_ans_path, "dev", num_samples=0)
                    logging.info("Epoch %d, Iter %d, Dev Submission Score: %f" % (epoch, global_step, dev_score))
                    #logging.info("Confusion Matrix for Dev set:", dev_cm)
                    write_summary(dev_score, "dev/subm_score", summary_writer, global_step)
                    #write_summary(dev_em, "dev/EM", summary_writer, global_step)

                    # Early stopping based on submission score.
                    if best_subm_score is None or dev_score > best_subm_score:
                        best_subm_score = dev_score
                        logging.info("Saving to %s..." % bestmodel_ckpt_path)
                        self.bestmodel_saver.save(session, bestmodel_ckpt_path, global_step=global_step)
                    
                    if lowest_dev_loss is None or dev_loss < lowest_dev_loss:
                        lowest_dev_loss = dev_loss
                        logging.info("Saving to %s..." % bestmodel_dev_loss_ckpt_path)
                        self.bestmodel_dev_loss_saver.save(session, bestmodel_dev_loss_ckpt_path, global_step=global_step)
                        
# =============================================================================
#                     # Early stopping based on dev EM. You could switch this to use F1 instead.
#                     if best_dev_em is None or dev_em > best_dev_em:
#                         best_dev_em = dev_em
#                         logging.info("Saving to %s..." % bestmodel_ckpt_path)
#                         self.bestmodel_saver.save(session, bestmodel_ckpt_path, global_step=global_step)
# =============================================================================


            epoch_toc = time.time()
            logging.info("End of epoch %i. Time for epoch: %f" % (epoch, epoch_toc-epoch_tic))

        sys.stdout.flush()



def write_summary(value, tag, summary_writer, global_step):
    """Write a single summary value to tensorboard"""
    summary = tf.Summary()
    summary.value.add(tag=tag, simple_value=value)
    summary_writer.add_summary(summary, global_step)
