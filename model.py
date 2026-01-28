import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):

  def __init__(self,d_model:int,vocab_size:int):
    super().__init__()
    self.d_model=d_model # dimensions of model
    self.vocab_size=vocab_size # number of vocabulary inputs that we have given
    self.embedding=nn.Embedding(vocab_size,d_model)# tells to create a look up table of matrix with size d_model and vacab_size
    

  def forward(self,x):
    return self.embedding(x)*math.sqrt(self.d_model)
  ## transformers work when the model size is neither too big nor too small 
  ## we multiply by √(d_model) to make the numbers just the right size.

class PositionalEncoding(nn.Module):

  def __init__(self,d_model:int,seq_len:int,dropout:float)->None:
    super().__init__()
    self.d_model=d_model
    self.seq_len=seq_len
    self.dropout=nn.Dropout(dropout) ## Dropout stops the model from depending too much on positions

    ## create a matrix of shape (seq_len,d_model)
    pe=torch.zeros(seq_len,d_model) ## this will hold pe[position][dimension] 
    ## example: pe[3][10] → value for position 3, dimension 10

    ## create a vector of shape (seq_len,1)
    position=torch.arange(0,seq_len,dtype=torch.float).unsqueeze(1) #(seq_len,1)
    ## This creates different frequencies for each embedding dimension.
    ## Small index → slow sine wave (global position)
    ## Large index → fast sine wave (local position)
    div_term=torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model))

    ## apply sin to even positions
    pe[:,0::2]=torch.sin(position*div_term)
    
    ## apply cosine to odd psoitions
    pe[:,1::2]=torch.cos(position*div_term)

    ## why sin + cos? Together they uniquely encode positions and help relative comparisons.

    pe=pe.unsqueeze(0) # (1,seq_len,d_model)

    self.register_buffer('pe',pe)
  
  def forward(self,x): ## forward pass is basically the runtime
    x=x+(self.pe[:,:x.shape[1],:]).requires_grad(False)
    return self.dropout(x)

## layer normalization
class layerNormalization:
  def __init__(self,d_model:int,eps:float=1e-6):
    super().__init__()
    self.d_model=d_model
    self.eps=eps
    self.gamma=nn.Parameter(torch.ones(d_model))
    self.beta=nn.Parameter(torch.zeros(d_model))
