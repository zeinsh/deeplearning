class HyperParameters:
    DATAPATH='./data/sport/'
    VOCPATH='./preprocessed'
    VOCFILE='bbc_sports.voc'
    MODEL_CHKPNT_PATH='./model/RNN_LSTM_V1_3.ckpt'

    MAXLENGTH=1000  # MAX Length of the document
    K=3 # beam search
    n_hidden_rnn=200
    embedding_dim=200
    
    batch_size = 1
    n_epochs = 50
    learning_rate = 0.01
    learning_rate_decay = 1#.41
    dropout_keep_probability = 0.6
