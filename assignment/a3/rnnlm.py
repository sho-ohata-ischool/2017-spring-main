import time

import tensorflow as tf
import numpy as np


def matmul3d(X, W):
    """Wrapper for tf.matmul to handle a 3D input tensor X.
    Will perform multiplication along the last dimension.

    Args:
      X: [m,n,k]
      W: [k,l]

    Returns:
      XW: [m,n,l]
    """
    Xr = tf.reshape(X, [-1, tf.shape(X)[2]])
    XWr = tf.matmul(Xr, W)
    newshape = [tf.shape(X)[0], tf.shape(X)[1], tf.shape(W)[1]]
    return tf.reshape(XWr, newshape)


def MakeFancyRNNCell(H, keep_prob, num_layers=1):
    """Make a fancy RNN cell.

    Use tf.nn.rnn_cell functions to construct an LSTM cell.
    Initialize forget_bias=0.0 for better training.

    Args:
      H: hidden state size
      keep_prob: dropout keep prob (same for input and output)
      num_layers: number of cell layers

    Returns:
      (tf.nn.rnn_cell.RNNCell) multi-layer LSTM cell with dropout
    """
    cell = tf.nn.rnn_cell.BasicLSTMCell(H, forget_bias=0.0)
    cell = tf.nn.rnn_cell.DropoutWrapper(
        cell, input_keep_prob=keep_prob, output_keep_prob=keep_prob)
    cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers)
    return cell


# Decorator-foo to avoid indentation hell.
# Decorating a function as:
# @with_self_graph
# def foo(self, ...):
#     # do tensorflow stuff
#
# Makes it behave as if it were written:
# def foo(self, ...):
#     with self.graph.as_default():
#         # do tensorflow stuff
#
# We hope this will save you some indentation, and make things a bit less
# error-prone.
def with_self_graph(function):
    def wrapper(self, *args, **kwargs):
        with self.graph.as_default():
            return function(self, *args, **kwargs)
    return wrapper


class RNNLM(object):
    def __init__(self, graph=None, *args, **kwargs):
        """Init function.

        This function just stores hyperparameters. You'll do all the real graph
        construction in the Build*Graph() functions below.

        Args:
          V: vocabulary size
          H: hidden state dimension
          num_layers: number of RNN layers (see tf.nn.rnn_cell.MultiRNNCell)
        """
        # Set TensorFlow graph. All TF code will work on this graph.
        self.graph = graph or tf.Graph()
        self.SetParams(*args, **kwargs)

    @with_self_graph
    def SetParams(self, V, H, softmax_ns=200, num_layers=1):
        # Model structure; these need to be fixed for a given model.
        self.V = V
        self.H = H
        self.num_layers = num_layers

        # Training hyperparameters; these can be changed with feed_dict,
        # and you may want to do so during training.
        with tf.name_scope("Training_Parameters"):
            self.softmax_ns = softmax_ns

            self.learning_rate_ = tf.placeholder_with_default(
                0.1, [], name="learning_rate")

            # For gradient clipping, if you use it.
            # Due to a bug in TensorFlow, this needs to be an ordinary python
            # constant instead of a tf.constant.
            self.max_grad_norm_ = 5.0

            self.use_dropout_ = tf.placeholder_with_default(
                False, [], name="use_dropout")

            # If use_dropout is fed as 'True', this will have value 0.5.
            self.dropout_keep_prob_ = tf.cond(
                self.use_dropout_,
                lambda: tf.constant(0.5),
                lambda: tf.constant(1.0),
                name="dropout_keep_prob")

            # Dummy for use later.
            self.no_op_ = tf.no_op()


    @with_self_graph
    def BuildCoreGraph(self):
        """Construct the core RNNLM graph, needed for any use of the model.

        This should include:
        - Placeholders for input tensors (input_w_, initial_h_, target_y_)
        - Variables for model parameters
        - Tensors representing various intermediate states
        - A Tensor for the final state (final_h_)
        - A Tensor for the output logits (logits_), i.e. the un-normalized argument
          of the softmax(...) function in the output layer.
        - A scalar loss function (loss_)

        Your loss function should be a *scalar* value that represents the
        _average_ loss across all examples in the batch (i.e. use tf.reduce_mean,
        not tf.reduce_sum).

        You shouldn't include training or sampling functions here; you'll do
        this in BuildTrainGraph and BuildSampleGraph below.

        We give you some starter definitions for input_w_ and target_y_, as
        well as a few other tensors that might help. We've also added dummy
        values for initial_h_, logits_, and loss_ - you should re-define these
        in your code as the appropriate tensors.

        See the in-line comments for more detail.
        """
        # Input ids, with dynamic shape depending on input.
        # Should be shape [batch_size, max_time] and contain integer word indices.
        self.input_w_ = tf.placeholder(tf.int32, [None, None], name="w")

        # Initial hidden state. You'll need to overwrite this with cell.zero_state
        # once you construct your RNN cell.
        self.initial_h_ = None

        # Final hidden state. You'll need to overwrite this with the output from
        # tf.nn.dynamic_rnn so that you can pass it in to the next batch (if
        # applicable).
        self.final_h_ = None

        # Output logits, which can be used by loss functions or for prediction.
        # Overwrite this with an actual Tensor of shape [batch_size, max_time]
        self.logits_ = None

        # Should be the same shape as inputs_w_
        self.target_y_ = tf.placeholder(tf.int32, [None, None], name="y")

        # Replace this with an actual loss function
        self.loss_ = None

        # Get dynamic shape info from inputs
        with tf.name_scope("batch_size"):
            self.batch_size_ = tf.shape(self.input_w_)[0]
        with tf.name_scope("max_time"):
            self.max_time_ = tf.shape(self.input_w_)[1]

        # Get sequence length from input_w_.
        # TL;DR: pass this to dynamic_rnn.
        # This will be a vector with elements ns[i] = len(input_w_[i])
        # You can override this in feed_dict if you want to have different-length
        # sequences in the same batch, although you shouldn't need to for this
        # assignment.
        self.ns_ = tf.tile([self.max_time_], [self.batch_size_, ], name="ns")

        #### YOUR CODE HERE ####

        # Construct embedding layer
        # Fill in:
        # - self.W_in_: the embedding matrix variable.
        # - self.x_: the result of looking up the input_w_-ords in the embedding variable.
        # Hint: see materials/week4



        # Construct RNN/LSTM cell and recurrent layer.
        # Hint: Constructing a RNN in TensorFlow involves two steps:
        #
        #       A.  Create a "template" LSTM cell.  MakeFancyRNNCell earlier in this
        #           file does this for you.  (Just pass self.dropout_keep_prob_,
        #           which we define for you, as the second parameter.
        #           Optionally, see Piazza or
        #           https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf
        #           if you want to learn more about dropout.)
        #
        #       B.  Repeat that cell the appropriate number of times
        #           (tf.nn.dynamic_rnn does this for you).
        #
        # Hint: There are a number of types of RNN cell.  LSTM is but one of them.
        #       what they all have in common is that they generate:
        #       1. a hidden state you can use for classification, etc.
        #          For a LSTM, both the "C"ell and the "H"idden layer are bundled
        #          together in (A).
        #          The "H" is duplicated as (B).
        #
        #       2. some state that must be forwarded to the next cell in the sequence and
        #
        #       tf.nn.dynamic_rnn returns:
        #         - The hidden layer (#1) from each cell through the sequence.
        #         - The final state output (i.e. #2) from the last cell in the sequence.
        #
        # Hint: The first cell of a LSTM (or any other RNN) needs some initial state.
        #       (subsequence cells get their state from the previous cell).  In TensorFlow,
        #       this is called the zero_state.  Each cell knows how to generate a zero-state
        #       of appropriate shape for itself and makes it available through the zero_state
        #       function.
        #
        # You want to fill in:
        # - self.cell_: the cell template to use
        # - self.initial_h_: The corresponding zero state for this cell
        # - self.o_: the output hidden layer for each step in the sequence.
        # - self.final_h_: The final state from the sequence that we'd want to pass
        #                  along to the next cell in the sequence, if there were one.





        # Softmax output layer, over vocabulary
        # Fill in:
        # - self.W_out_
        # - self.b_out_: the usual weights and bias variables of the final affine layer.
        # Hint: No need to do the actual softmax here, just compute the logits.
        # Hint: use the matmul3d() helper here to perform the affine layer over all
        #       steps in your sequence in a single function call.



        # Loss computation (true loss, for prediction)
        # Fill in:
        # - self.loss_: the *mean* of the loss of the examples in this batch.
        # Hint: Use tf.nn.sparse_softmax_cross_entropy_with_logits to compute the loss.
        #       Like assignment 1, be careful which parameters are logits and which are
        #       labels!



        #### END(YOUR CODE) ####

    @with_self_graph
    def BuildTrainGraph(self):
        """Construct the training ops.

        You should define:
        - train_loss_ : sampled softmax loss, for training
        - train_step_ : a training op that can be called once per batch

        Your loss function should be a *scalar* value that represents the
        _average_ loss across all examples in the batch (i.e. use tf.reduce_mean,
        not tf.reduce_sum).
        """
        # Replace this with an actual training op
        self.train_step_ = None

        # Replace this with an actual loss function
        self.train_loss_ = None

        #### YOUR CODE HERE ####

        # Define approximate loss function.
        # Hierarchical softmax is one way to speed up training.
        # Another approach, with a similar goal, is sampled_softmax_loss.
        # The mechanics (and paper) are described in the instructions.
        #
        # Fill in:
        # - self.train_loss_
        #
        # Hint: self.softmax_ns is already defined (see SetParams) as the number
        #       of sampled negative examples to use.
        # Hint: use printf and .get_shape() in here to make sure you understand the
        #       shape of all your variables.  Depending on how you implement the
        #       rest of the assignment, it's quite likely that you'll want to use
        #       a tf.transpose or tf.expand_dims.
        # Hint: use tf.reduce_mean, not tf.reduce_sum to turn the vector of per-example
        #       loss values into a single number.
            # Loss computation (sampled, for training)



        # Define optimizer and training op
        # Fill in: self.train_step_
        # Hint: use AdagradOptimizer.  This optimizer adapts the learning rate
        #       on a per-variable basis.  This aggressively moves word vectors for
        #       rarely seen words, but doesn't do much to the word vector for 'the'
        #       after the first little bit of training.
        #       (see http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf
        #        for more detail if you are interested)



        #### END(YOUR CODE) ####

    @with_self_graph
    def BuildSamplerGraph(self):
        """Construct the sampling ops.

        You should define pred_samples_ to be a Tensor of integer indices for
        sampled predictions for each batch element, at each timestep.

        Hint: use tf.multinomial, along with a couple of calls to tf.reshape
        """
        # Replace with a Tensor of shape [batch_size, max_time, 1]
        self.pred_samples_ = None

        #### YOUR CODE HERE ####



        #### END(YOUR CODE) ####


