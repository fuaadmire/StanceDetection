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

"""This file contains some basic model components"""

import tensorflow as tf
from tensorflow.python.ops.rnn_cell import DropoutWrapper
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import math_ops
#import numpy as np


class RNNEncoder(object):
    """
    General-purpose module to encode a sequence using a RNN.
    It feeds the input through a RNN and returns all the hidden states.

    Note: In lecture 8, we talked about how you might use a RNN as an "encoder"
    to get a single, fixed size vector representation of a sequence
    (e.g. by taking element-wise max of hidden states).
    Here, we're using the RNN as an "encoder" but we're not taking max;
    we're just returning all the hidden states. The terminology "encoder"
    still applies because we're getting a different "encoding" of each
    position in the sequence, and we'll use the encodings downstream in the model.

    This code uses a bidirectional GRU, but you could experiment with other types of RNN.
    """

    def __init__(self, hidden_size, keep_prob):
        """
        Inputs:
          hidden_size: int. Hidden size of the RNN
          keep_prob: Tensor containing a single scalar that is the keep probability (for dropout)
        """
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        self.rnn_cell_fw = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_fw = DropoutWrapper(self.rnn_cell_fw, input_keep_prob=self.keep_prob)
        self.rnn_cell_bw = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_bw = DropoutWrapper(self.rnn_cell_bw, input_keep_prob=self.keep_prob)

    def build_graph(self, inputs, masks, scope_name="RNNEncoder"):
        """
        Inputs:
          inputs: Tensor shape (batch_size, seq_len, input_size)
          masks: Tensor shape (batch_size, seq_len).
            Has 1s where there is real input, 0s where there's padding.
            This is used to make sure tf.nn.bidirectional_dynamic_rnn doesn't iterate through masked steps.

        Returns:
          out: Tensor shape (batch_size, seq_len, hidden_size*2).
            This is all hidden states (fw and bw hidden states are concatenated).
        """
        with vs.variable_scope(scope_name):
            input_lens = tf.reduce_sum(masks, reduction_indices=1) # shape (batch_size)

            # Note: fw_out and bw_out are the hidden states for every timestep.
            # Each is shape (batch_size, seq_len, hidden_size).
            (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw, self.rnn_cell_bw, inputs, input_lens, dtype=tf.float32)

            # Concatenate the forward and backward hidden states
            out = tf.concat([fw_out, bw_out], 2)

            # Apply dropout
            out = tf.nn.dropout(out, self.keep_prob)

            return out

class CustomSimpleSoftmaxLayer(object):
    def __init__(self):
        pass
    
    def build_graph(self, inputs, masks, reduction_type):
        """
        Applies one linear downprojection layer, then softmax.

        Inputs:
          inputs: Tensor shape (batch_size, seq_len, hidden_size)
          masks: Tensor shape (batch_size, seq_len)
            Has 1s where there is real input, 0s where there's padding.

        Outputs:
          logits: Tensor shape (batch_size, seq_len)
            logits is the result of the downprojection layer, but it has -1e30
            (i.e. very large negative number) in the padded locations
          prob_dist: Tensor shape (batch_size, seq_len)
            The result of taking softmax over logits.
            This should have 0 in the padded locations, and the rest should sum to 1.
        """
        with vs.variable_scope("CustomSimpleSoftmaxLayer"):
            inputs = tf.cast(inputs, 'float')
            masks_exp_dim = tf.cast(tf.expand_dims(masks,2),'float') # shape (batch_size, seq_len ,1)
            
            if reduction_type == 'max':
                print("Max reduction......")
                exp_mask = (1 - tf.cast(masks_exp_dim, 'float')) * (-1e30)
                # Now large -ve value for padded words...
                add_element_wise = inputs+exp_mask #(bs,N,h)
                max_column_tensor = tf.reduce_max(add_element_wise,1) #(bs,h)
                expanded_dim = tf.expand_dims(max_column_tensor,1) # shape (batch_size, 1, hidden_size)
            
            elif reduction_type == 'mean':
                print("Mean reduction....")
                input_multiply = math_ops.multiply(inputs, masks_exp_dim) # shape (batch_size, seq_len, hidden_size)
                reduced_mean_input = tf.reduce_mean(input_multiply, 1) # shape (batch_size,hidden_size)
                expanded_dim = tf.expand_dims(reduced_mean_input,1) # shape (batch_size, 1, hidden_size)


# =============================================================================
#             print("FLattening layers instead of mean")
#             shape = input_multiply.get_shape().as_list()
#             dim = np.prod(shape[1:])
#             expanded_dim = tf.reshape(input_multiply, [-1,1,dim])
# =============================================================================
            
            # Reduce to 4 dim
            logits = tf.contrib.layers.fully_connected(expanded_dim, num_outputs=4)# shape (batch_size, 1,4)
            logits = tf.squeeze(logits, axis=[1]) # shape (batch_size, 4)
            
            # Apply softmax on it
            prob_dist = tf.nn.softmax(logits, 1) # softmax along 1st dimension. # shape (batch_size, 4)
            return logits, prob_dist
    
    
class SimpleSoftmaxLayer(object):
    """
    Module to take set of hidden states, (e.g. one for each context location),
    and return probability distribution over those states.
    """

    def __init__(self):
        pass

    def build_graph(self, inputs, masks):
        """
        Applies one linear downprojection layer, then softmax.

        Inputs:
          inputs: Tensor shape (batch_size, seq_len, hidden_size)
          masks: Tensor shape (batch_size, seq_len)
            Has 1s where there is real input, 0s where there's padding.

        Outputs:
          logits: Tensor shape (batch_size, seq_len)
            logits is the result of the downprojection layer, but it has -1e30
            (i.e. very large negative number) in the padded locations
          prob_dist: Tensor shape (batch_size, seq_len)
            The result of taking softmax over logits.
            This should have 0 in the padded locations, and the rest should sum to 1.
        """
        with vs.variable_scope("SimpleSoftmaxLayer"):

            # Linear downprojection layer
            logits = tf.contrib.layers.fully_connected(inputs, num_outputs=1, activation_fn=None) # shape (batch_size, seq_len, 1)
            logits = tf.squeeze(logits, axis=[2]) # shape (batch_size, seq_len)

            # Take softmax over sequence
            masked_logits, prob_dist = masked_softmax(logits, masks, 1)

            return masked_logits, prob_dist


class BidafAttn(object):
    def __init__(self, keep_prob, vec_size):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          key_vec_size: size of the key vectors. int
          value_vec_size: size of the value vectors. int
        """
        self.keep_prob = keep_prob
        self.vec_size = vec_size #2*h
        self.Sv = tf.get_variable('Sv', shape = [self.vec_size*3,], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()) # Similarity vector (6h,)
        
    
    def build_graph(self, body_hiddens, headline_hiddens, headline_mask, body_mask):
        #body_hiddens: shape: (bs, N, 2h)
        #headline_hiddens: shape (bs, M,2h)
        #values_mask/ headline_mask: shape(bs, M)
        #keys_mask/ body_mask: shape (bs, N)
        
        with vs.variable_scope("BidafAttn"):
            body_hidden_expanded = tf.expand_dims(body_hiddens, 2) # (bs, N, 1, 2h)
            headline_hidden_expanded = tf.expand_dims(headline_hiddens, 1) #(bs, 1, M, 2h)
            
            shape_headline_hidden = headline_hiddens.get_shape().as_list()
            body_tiled = tf.tile(body_hidden_expanded, [1,1,shape_headline_hidden[1],1]) #(bs, N,M,2h)
            
            shape_body_hiddens = body_hiddens.get_shape().as_list()
            headline_tiled = tf.tile(headline_hidden_expanded, [1,shape_body_hiddens[1],1,1]) #(bs, N,M,2h)
            
            body_headline_mult = body_tiled*headline_tiled #(bs, N,M,2h)
            
            combined_matrix = tf.concat([body_tiled, headline_tiled,body_headline_mult], axis=3) #(bs, N,M,6h)
            
            # Calculating dot product with weight vector for evaluating similarity matrix...
            mult_wt_vector = self.Sv*combined_matrix #(bs, N,M,6h)
            sim_matrix = tf.reduce_sum(mult_wt_vector, axis=3) #(bs, N,M)
            
            #C2Q attention STARTS....
            attn_logits_mask = tf.expand_dims(headline_mask, 1) # shape (batch_size, 1, num_values)
            _, attn_dist = masked_softmax(sim_matrix, attn_logits_mask, 2) # shape (batch_size, num_keys(N), num_values(M)). take softmax over values
            
            c2q_attention = tf.matmul(attn_dist, headline_hiddens) # shape (batch_size, num_keys, 2h)

            # Apply dropout
            #c2q_attention = tf.nn.dropout(c2q_attention, self.keep_prob) # shape (batch_size, num_keys(N), 2h)
            #C2Q attention ENDS....
            
            #Q2C attention:
            #1. Calculate max along column
            max_column_tensor = tf.reduce_max(sim_matrix,2) #(bs, N)
            
            #2. Calculate softmax / attention distribution
            _, attn_dist_q2c = masked_softmax(max_column_tensor, body_mask, 1) #(bs, N)
            
            #3. Expand dimension
            attn_dist_q2c_expanded = tf.expand_dims(attn_dist_q2c,2) #(bs,N,1)
            
            #4. Calculate attended body vectors
            mult_attn_body = attn_dist_q2c_expanded*body_hiddens #(bs,N,2h)
            q2c_attention = tf.reduce_sum(mult_attn_body,axis=1) #(bs, 2h)
            #q2c_attention = tf.nn.dropout(q2c_attention, self.keep_prob) #(bs, 2h)
            
            #C2Q attention ENDS....
            
            return c2q_attention, q2c_attention
            
            
class SelfAttn(object):
    
    def __init__(self, keep_prob, z_len, l_len):
        self.keep_prob = keep_prob
        self.z_len = z_len
        self.l_len = l_len
        self.v = tf.get_variable("v", shape=[self.z_len,1], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        self.W1 = tf.get_variable("W1", shape=[self.z_len, self.l_len], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        self.W2 = tf.get_variable("W2", shape=[self.z_len, self.l_len], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        
    def build_graph(self, inputs, body_mask):
        #inputs/ basic_attn_output shape: (batch_size, context_len/N, hidden_size*2)
        #body_mask: (bs,N)
        inputs_t = tf.transpose(inputs, perm=[0, 2, 1]) #(bs, 2h,N)
        
        W1_input_mult = tf.transpose(tf.tensordot(self.W1, inputs_t, axes=[[1],[1]]),perm=[1,0,2]) #(bs,z_len,N)
        
        W2_input_mult = tf.transpose(tf.tensordot(self.W2, inputs_t, axes=[[1],[1]]),perm=[1,0,2]) #(bs,z_len,N)
        
        # Code fo addiion..
        W1_input_mult_expand = tf.expand_dims(W1_input_mult, 1) #(bs,1,z,N)
        shape_W1_input_mult_expand = W1_input_mult_expand.get_shape().as_list()
        W1_input_mult_expand_tiled = tf.tile(W1_input_mult_expand, [1,shape_W1_input_mult_expand[-1],1,1]) #(bs,N,z,N)
        W1_input_mult_to_add = tf.transpose(W1_input_mult_expand_tiled, perm=[0,3,2,1]) #(bs,N,z,N)
        
        W2_input_mult_expand = tf.expand_dims(W2_input_mult, 1) #(bs,1,z,N)
        shape_W2_input_mult_expand = W2_input_mult_expand.get_shape().as_list()
        W2_input_mult_to_add = tf.tile(W2_input_mult_expand, [1,shape_W2_input_mult_expand[-1],1,1]) #(bs,N,z,N)
        
        W1_W2_add = W1_input_mult_to_add+W2_input_mult_to_add #(bs,N,z,N)
        
        tanh_output = tf.tanh(W1_W2_add) #(bs,N,z,N)
        
        v_t = tf.transpose(self.v, perm=[1,0]) #(1,z)
        
        v_tan_matmul = tf.transpose(tf.tensordot(v_t, tanh_output, axes=[[1],[2]]), perm=[1,2,0,3])
        
        v_tan_matmul_squeeze = tf.squeeze(v_tan_matmul, axis=[2]) #(bs,N,N)
        
        body_mask_expand = tf.expand_dims(body_mask, 1) # shape (batch_size, 1, num_values/N)
        _, attn_dist = masked_softmax(v_tan_matmul_squeeze, body_mask_expand, 2) # shape (batch_size, N,N). take softmax over values
        
        self_attention = tf.matmul(attn_dist, inputs) #(bs,N,2h)
        
        return attn_dist, self_attention
        
        
        

class BasicAttn(object):
    """Module for basic attention.

    Note: in this module we use the terminology of "keys" and "values" (see lectures).
    In the terminology of "X attends to Y", "keys attend to values".

    In the baseline model, the keys are the context hidden states
    and the values are the question hidden states.

    We choose to use general terminology of keys and values in this module
    (rather than context and question) to avoid confusion if you reuse this
    module with other inputs.
    """

    def __init__(self, keep_prob, key_vec_size, value_vec_size):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          key_vec_size: size of the key vectors. int
          value_vec_size: size of the value vectors. int
        """
        self.keep_prob = keep_prob
        self.key_vec_size = key_vec_size
        self.value_vec_size = value_vec_size

    def build_graph(self, values, values_mask, keys):
        """
        Keys attend to values.
        For each key, return an attention distribution and an attention output vector.

        Inputs:
          values: Tensor shape (batch_size, num_values, value_vec_size).
          values_mask: Tensor shape (batch_size, num_values).
            1s where there's real input, 0s where there's padding
          keys: Tensor shape (batch_size, num_keys, value_vec_size)

        Outputs:
          attn_dist: Tensor shape (batch_size, num_keys, num_values).
            For each key, the distribution should sum to 1,
            and should be 0 in the value locations that correspond to padding.
          output: Tensor shape (batch_size, num_keys, hidden_size).
            This is the attention output; the weighted sum of the values
            (using the attention distribution as weights).
        """
        with vs.variable_scope("BasicAttn"):

            # Calculate attention distribution
            values_t = tf.transpose(values, perm=[0, 2, 1]) # (batch_size, value_vec_size, num_values)
            attn_logits = tf.matmul(keys, values_t) # shape (batch_size, num_keys, num_values)
            attn_logits_mask = tf.expand_dims(values_mask, 1) # shape (batch_size, 1, num_values)
            _, attn_dist = masked_softmax(attn_logits, attn_logits_mask, 2) # shape (batch_size, num_keys, num_values). take softmax over values

            # Use attention distribution to take weighted sum of values
            output = tf.matmul(attn_dist, values) # shape (batch_size, num_keys, value_vec_size)

            # Apply dropout
            output = tf.nn.dropout(output, self.keep_prob)

            return attn_dist, output


def masked_softmax(logits, mask, dim):
    """
    Takes masked softmax over given dimension of logits.

    Inputs:
      logits: Numpy array. We want to take softmax over dimension dim.
      mask: Numpy array of same shape as logits.
        Has 1s where there's real data in logits, 0 where there's padding
      dim: int. dimension over which to take softmax

    Returns:
      masked_logits: Numpy array same shape as logits.
        This is the same as logits, but with 1e30 subtracted
        (i.e. very large negative number) in the padding locations.
      prob_dist: Numpy array same shape as logits.
        The result of taking softmax over masked_logits in given dimension.
        Should be 0 in padding locations.
        Should sum to 1 over given dimension.
    """
    exp_mask = (1 - tf.cast(mask, 'float')) * (-1e30) # -large where there's padding, 0 elsewhere
    masked_logits = tf.add(logits, exp_mask) # where there's padding, set logits to -large
    prob_dist = tf.nn.softmax(masked_logits, dim)
    return masked_logits, prob_dist
