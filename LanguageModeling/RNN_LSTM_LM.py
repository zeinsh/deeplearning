# modifications
# 1. add training/inference variable
# 2. define placeholder for state [NONE,2*stateLength]
# 3. if inference use state placeholder
# 4. modify beam search
# 5. try to use built-in beam search

import tensorflow as tf
import numpy as np
from hyper import HyperParameters
hp=HyperParameters()

class LSTMModel():
    def __declare_placeholders(self):
        """Specifies placeholders for the model."""

        # Placeholders for input and ground truth output.
        self.input_batch = tf.placeholder(dtype=tf.int32, shape=[None, None], name='input_batch') 
        self.ground_truth_tags = tf.placeholder(dtype=tf.int32, shape=[None, None], name='ground_truth_tags')
        self.initial_state_h= tf.placeholder(dtype=tf.float32, shape=[None, hp.n_hidden_rnn], name='initial_state_h')
        self.initial_state_c= tf.placeholder(dtype=tf.float32, shape=[None, hp.n_hidden_rnn], name='initial_state_c')

        # Placeholder for lengths of the sequences.
        self.lengths = tf.placeholder(dtype=tf.int32, shape=[None], name='lengths') 

        # Placeholder for a dropout keep probability. If we don't feed
        # a value for this placeholder, it will be equal to 1.0.
        self.dropout_ph = tf.placeholder_with_default(tf.cast(1.0, dtype=tf.float32), shape=[])

        # Placeholder for a learning rate (tf.float32).
        self.learning_rate_ph = tf.placeholder(dtype=tf.float32, shape=[], name='learning_rate_ph')

    def __build_layers(self, vocabulary_size, embedding_dim, n_hidden_rnn, n_tags):
        """Specifies bi-LSTM architecture and computes logits for inputs."""

        # Create embedding variable (tf.Variable) with dtype tf.float32
        initial_embedding_matrix = np.random.randn(vocabulary_size, embedding_dim) / np.sqrt(embedding_dim)
        embedding_matrix_variable = tf.Variable(initial_value=initial_embedding_matrix, dtype=tf.float32, name='embeddings_matrix')

        # Create RNN cells (for example, tf.nn.rnn_cell.BasicLSTMCell) with n_hidden_rnn number of units 
        # and dropout (tf.nn.rnn_cell.DropoutWrapper), initializing all *_keep_prob with dropout placeholder.

        forward_cell =  tf.nn.rnn_cell.DropoutWrapper(cell=tf.nn.rnn_cell.BasicLSTMCell(num_units=n_hidden_rnn),
                                                      input_keep_prob=self.dropout_ph, output_keep_prob=self.dropout_ph, state_keep_prob=self.dropout_ph, dtype=tf.float32)
        #backward_cell = tf.nn.rnn_cell.DropoutWrapper(cell=tf.nn.rnn_cell.BasicLSTMCell(num_units=n_hidden_rnn), input_keep_prob=self.dropout_ph, output_keep_prob=self.dropout_ph, state_keep_prob=self.dropout_ph, dtype=tf.float32)

        # Look up embeddings for self.input_batch (tf.nn.embedding_lookup).
        # Shape: [batch_size, sequence_len, embedding_dim].
        embeddings = tf.nn.embedding_lookup(embedding_matrix_variable, self.input_batch)

        # Pass them through Bidirectional Dynamic RNN (tf.nn.bidirectional_dynamic_rnn).
        # Shape: [batch_size, sequence_len, 2 * n_hidden_rnn]. 
        # Also don't forget to initialize sequence_length as self.lengths and dtype as tf.float32.
        rnn_output, self.states = tf.nn.dynamic_rnn(cell=forward_cell,
                                          initial_state=tf.nn.rnn_cell.LSTMStateTuple(self.initial_state_c, self.initial_state_h),
                                          sequence_length=self.lengths,
                                          dtype=tf.float32,
                                          inputs=embeddings)
        #rnn_output = tf.concat([rnn_output_fw, rnn_output_bw], axis=2)

        # Dense layer on top.
        # Shape: [batch_size, sequence_len, n_tags].   
        self.logits = tf.layers.dense(rnn_output, vocabulary_size, activation=None)

    def __compute_predictions(self):
        """Transforms logits to probabilities and finds the most probable tags."""

        # Create softmax (tf.nn.softmax) function
        self.softmax_output = tf.nn.softmax(logits=self.logits)

        # Use argmax (tf.argmax) to get the most probable tags
        # Don't forget to set axis=-1
        # otherwise argmax will be calculated in a wrong way
        self.predictions = tf.argmax(self.softmax_output, axis=-1)
        
    def __compute_loss(self, vocabulary_size, PAD_index):
        """Computes masked cross-entopy loss with logits."""

        # Create cross entropy function function (tf.nn.softmax_cross_entropy_with_logits)
        ground_truth_tags_one_hot = tf.one_hot(self.ground_truth_tags, vocabulary_size)
        #loss_tensor = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=ground_truth_tags_one_hot)
        mask = tf.cast(tf.not_equal(self.input_batch, PAD_index), dtype=tf.float32)
        loss_tensor = tf.contrib.seq2seq.sequence_loss(
            logits=self.logits,
            targets=self.ground_truth_tags,
            weights=mask)
        # Create loss function which doesn't operate with <PAD> tokens (tf.reduce_mean)
        # Be careful that the argument of tf.reduce_mean should be
        # multiplication of mask and loss_tensor.
        self.loss = tf.reduce_mean(loss_tensor)#np.multiply(mask, loss_tensor))
        
    def __perform_optimization(self):
        """Specifies the optimizer and train_op for the model."""

        # Create an optimizer (tf.train.AdamOptimizer)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)
        self.grads_and_vars = self.optimizer.compute_gradients(self.loss)

        # Gradient clipping (tf.clip_by_norm) for self.grads_and_vars
        # Pay attention that you need to apply this operation only for gradients 
        # because self.grads_and_vars contains also variables.
        # list comprehension might be useful in this case.
        clip_norm = tf.cast(1.0, dtype=tf.float32)  ##??
        self.grads_and_vars = [(tf.clip_by_norm(grad, clip_norm), var) for grad, var in self.grads_and_vars]

        self.train_op = self.optimizer.apply_gradients(self.grads_and_vars)
        
    def __init__(self, vocabulary_size, n_tags, embedding_dim, n_hidden_rnn, PAD_index):
        self.__declare_placeholders()
        self.__build_layers(vocabulary_size, embedding_dim, n_hidden_rnn, n_tags)
        self.__compute_predictions()
        self.__compute_loss(n_tags, PAD_index)
        self.__perform_optimization()
        
    def train_on_batch(self, session, x_batch, y_batch, lengths, learning_rate, dropout_keep_probability ,state_c=None ,state_h=None, zero_state=True):
        st_c, st_h=state_c, state_h
        if zero_state:
            st_h,st_c=np.random.random((lengths.shape[0],hp.n_hidden_rnn)),np.random.random((lengths.shape[0],hp.n_hidden_rnn))
            #st_h,st_c=np.zeros((lengths.shape[0],hp.n_hidden_rnn)),np.zeros((lengths.shape[0],hp.n_hidden_rnn))
        feed_dict = {self.input_batch: x_batch,
                     self.ground_truth_tags: y_batch,
                     self.initial_state_h: st_h,
                     self.initial_state_c: st_c,
                     self.learning_rate_ph: learning_rate,
                     self.dropout_ph: dropout_keep_probability,
                     self.lengths: lengths}

        _,loss,states=session.run([self.train_op, self.loss,self.states], feed_dict=feed_dict)
        return np.exp(loss),states

    def predict_for_batch(self, session, x_batch,init_c,init_h):
        lengths=np.array([1000000]*len(x_batch))
        feed_dict = {self.input_batch: x_batch,
                     self.initial_state_h: init_h,
                     self.initial_state_c: init_c,
                     self.lengths: lengths}
        k=3
        softmax, states = session.run([self.softmax_output ,self.states], feed_dict=feed_dict)
        return softmax, states
    
    def calculatePerplexity(self, session, x_batch, y_batch, lengths ):
        init_c, init_h=np.random.random((1,200)),np.zeros((1,200)) # initial states
        feed_dict = {self.input_batch: x_batch,
                     self.ground_truth_tags: y_batch,
                     self.initial_state_h: init_h,
                     self.initial_state_c: init_c,
                     self.lengths: lengths}
        loss, states = session.run([self.loss ,self.states], feed_dict=feed_dict)
        return loss
        #loss = session.run([self.loss], feed_dict=feed_dict)
        #perplexity=np.exp(loss)
        #return perplexity