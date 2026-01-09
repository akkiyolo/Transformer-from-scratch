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