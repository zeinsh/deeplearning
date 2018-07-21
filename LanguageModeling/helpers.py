import numpy as np
from lib.dictionarymd import Dictionary
from hyper import HyperParameters
hp=HyperParameters()

def batches_generator(batch_size, docs,dictionary,
                      shuffle=False, allow_smaller_last_batch=True):
    """Generates padded batches of tokens and tags."""
    
    n_samples = len(docs)
    vecs=[dictionary.text2vec(doc)[:hp.MAXLENGTH] for doc in docs]
    if shuffle:
        order = np.random.permutation(n_samples)
    else:
        lengths=[len(s) for s in vecs]
        order = np.argsort(lengths)

    n_batches = n_samples // batch_size
    if allow_smaller_last_batch and n_samples % batch_size:
        n_batches += 1

    for k in range(n_batches):
        batch_start = k * batch_size
        batch_end = min((k + 1) * batch_size, n_samples)
        current_batch_size = batch_end - batch_start
        x_list = []
        y_list = []
        max_len_token = 0
        for idx in order[batch_start: batch_end]:
            x_list.append(vecs[idx][:-1])
            y_list.append(vecs[idx][1:])
            max_len_token = max(max_len_token, len(vecs[idx]))  #why?!
            
        # Fill in the data into numpy nd-arrays filled with padding indices.
        x = np.ones([current_batch_size, max_len_token], dtype=np.int32) * dictionary.word2idx['<PAD>']
        y = np.ones([current_batch_size, max_len_token], dtype=np.int32) * dictionary.word2idx['<PAD>']
        lengths = np.zeros(current_batch_size, dtype=np.int32)
        
        for n in range(current_batch_size):
            utt_len = len(x_list[n])
            x[n, :utt_len] = x_list[n]
            lengths[n] = utt_len
            y[n, :utt_len] = y_list[n]
        yield x, y, lengths

class Beam:
    def __init__(self,K):
        self.state_c=[]
        self.state_h=[]
        self.probabilities=[]
        self.lastOutput=[]
        self.history=[]
        self.K=K
    def addBeam(self,lastOutput,prob,state_c,state_h,history):
        self.lastOutput.append(lastOutput)
        self.probabilities.append(prob)
        self.state_c.append(state_c)
        self.state_h.append(state_h)
        current_hist=history.copy()
        current_hist.extend(lastOutput)
        self.history.append(current_hist)
    def getTopK(self):
        topK=np.argsort(self.probabilities)[-self.K:]
        lenTop=len(topK)
        state_c,state_h=np.zeros((lenTop,200)),np.zeros((lenTop,200))
        prob=[]
        lastOutput=[]
        history=[]
        for i,k in enumerate(topK):
            lastOutput.append(self.lastOutput[k])
            history.append(self.history[k])
            prob.append(self.probabilities[k])
            state_c[i,:]=self.state_c[k]
            state_h[i,:]=self.state_h[k]
        return lastOutput,prob,state_c,state_h,history

def beam_search(sess,model,num_generated,seed,dictionary):
    """ 
    Parameters
            num_generated: number of tokens to be generated.
            seed: initial sentence of the text.
            K: default value is 3;  #number of sequences to track
    return: topK candidates of the generated text
    """

    beam=Beam(hp.K)
    vec=dictionary.text2vec(seed)  # 1xL 
    state_c,state_h=np.zeros((1,200)),np.zeros((1,200)) # initial states
    p=1
    beam.addBeam(vec,p,state_c,state_h,[])

    for i in range(num_generated):
        lastOutputs,probs,state_c,state_h,history=beam.getTopK()

        softmax,states=model.predict_for_batch(sess,lastOutputs,state_c,state_h) # softmax [NONE,Len,10000]
                                                                                 # states ([NONE,200],[NONE,200])
        state_c,state_h=states

        beam=Beam(hp.K)
        topK=np.argsort(softmax[:,-1])[:,-hp.K:] #[None,K]
        for i in range(topK.shape[0]):
            for j in range(hp.K):
                cand=topK[i,j]
                if cand==1:continue
                vec=[cand]
                st_c=state_c[i,:]
                st_h=state_h[i,:]
                p=softmax[i,-1,cand]*probs[i]
                hist=history[i]
                beam.addBeam(vec,p,st_c,st_h,hist)
    lastOutputs,probs,state_c,state_h,history=beam.getTopK()
    return history

def getBestCandidate(sess,model,num_tokens,seed,dictionary):
    history=beam_search(sess,model,num_tokens,seed,dictionary)
    return dictionary.vec2text(history[-1])

if __name__ == "__main__":
    # load dictionary
    dictionary=Dictionary()
    dictionary.load_vocab(hp.VOCPATH,hp.VOCFILE)

    # test Beam
    beam=Beam(3)
    vec=dictionary.text2vec('Hi')
    prob=1
    state_c,state_h=np.zeros((1,200)),np.zeros((1,200))
    beam.addBeam(vec+[0],0.8,state_c,state_h,[])
    beam.addBeam(vec+[1],0.9,state_c+1,state_h+1,[])
    beam.addBeam(vec+[2],0.1,state_c+2,state_h+2,[])
    beam.addBeam(vec+[3],0.96,state_c+3,state_h+3,[])
    lastOutputs,probs,state_c,state_h,history=beam.getTopK()

    assert (lastOutputs[0][2]==0 and lastOutputs[1][2]==1 and lastOutputs[2][2]==3)
    assert probs==[0.8, 0.9, 0.96]
    assert state_c[0,0]==0 and state_c[1,0]==1 and state_c[2,0]==3 
    assert state_h[0,0]==0 and state_h[1,0]==1 and state_h[2,0]==3 
    print("Beam class test successfully passed")