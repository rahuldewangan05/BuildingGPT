import torch
import torch.nn as nn
from torch.nn import functional as F

## hyperparameters
batch_size=32 ## how many independent sequence send to model to process parallely
block_size=8  ## how many token are present in each sequence
max_iters=3000 
eval_interval=300
learning_rate=1e-2  
device='cuda' if torch.cuda.is_available() else 'cpu'
eval_iters=200
n_embed=32

## setting seed to get the same result every time
torch.manual_seed(1337)

## using input.txt file "shakespeare poem"
with open('Building GPT/input.txt','r',encoding='utf-8') as f:
    text=f.read()
    
## finding all the unique character present in input.txt and making list of that
chars=sorted(list(set(text)))
vocab_size=len(chars)

## dictnary containg {character:index}
stoi={ch:i for i,ch in enumerate(chars)}

## dictnary containg {index:character}
itos={i:ch for i,ch in enumerate(chars)}

## function to convert all character of string to integer and return list of integer
encode=lambda s: [stoi[c] for c in s]

## function to convert inters in list to their desired character and return a string
decode= lambda s: ''.join([itos[i] for i in s])

## train and test split
## since it is sequential data so we can't use traditio90% data for traditional train test split so take first 90% data as trainging and rest as testing
data=torch.tensor(encode(text),dtype=torch.long)
n=int(0.9*len(data))
train_data=data[:n]
test_data=data[n:]

## this function will return x(input sequence) and y(output sequence) being passed to model
def get_batch(split):
    ## generate a small batch of data of inputs x and target y
    data=train_data if split=='train' else test_data
    
    ## randomly generate 'batch size' no. of integer b/w 0 to len(data)-block_size
    ix=torch.randint(len(data)-block_size, (batch_size,))
    ## note :  ix is index (NOT ANY INTEGER REPRESENTATION OF ANY CHARACTER)
    ## now from each randomly generated integer, next 8(block_size) interger are taken 
    ## this list of 8 integer represnt 8 token in a sequence
    ## this means for each randomly generated integer there will be a sequence of 8 token starting from that randomly generated integer
    ## so we will have 32(batch_size) sequence each having 8(block_size) token
    
    ## making input sequence
    x=torch.stack([data[i:i+block_size] for i in ix])
    ## makin target sequence
    y=torch.stack([data[i+1:i+block_size+1] for i in ix])
    
    x,y=x.to(device) , y.to(device)  ## doubt 
    return x,y

@torch.no_grad()
def estimate_loss():
    ## it will store losses in train/test split
    out={}
    
    ## setting model in evalution mode : it is considered good practice to set model in desired mode in pytorch
    model.eval()
    
    ## using both training and testing data for calculation loss
    for split in ['train','test']:
        ## it will create a tensor(a kind of list) named losses or length 'eval_iters' 
        ## losses will store all the loses of each epoch(eval_iter) in each split(train/test)
        losses=torch.zeros(eval_iters)
        
        ## calculating loss in each epoch in a split(train/test)
        for k in range(eval_iters):
            ## getting x,y for each epoch
            X,Y=get_batch(split)
            ## calcuting logits and loss at each epoch
            logits,loss=model(X,Y)
            ## storing loss in losses list
            losses[k]=loss.item()
            
        ## storing loss of each split in out dictnary
        out[split]=losses.mean()
        ## calling model to train
        model.train()
    return out
    
## BigramLanguageModel
class BigramLanguageModel(nn.Module):
    ##m nn.module is a base class for all neural network model in pytorch
    def __init__(self):
        super().__init__()
        
        ## making embedding vector of dim 'n_embed' of integer reprenstation of character (As an alternative we may use one hot encoding but it will be inefficient)
        self.token_embedding_table=nn.Embedding(vocab_size,n_embed)
        
        ## positional embedding : this created another postional embedding layer to keep track of position of character in sequence
        self.position_embedding_table=nn.Embedding(block_size,n_embed)
        
        ## now till this point we will have input (embeddngs) to send to model
        
        ## now to convert this embedding to ouput we need linear layer
        ## linear layer is like Dense layer in tensorflow use to compute output from input so provided
        self.im_head=nn.Linear(n_embed,vocab_size)
        
    def forward(self,idx,targets=None):
        
        ## idx = input tensor of shape (B,T) , target = expected output or labels
        ## B-> batch size , T-> time step (block_size)
        
        #Let's assume our vocabulary contains 5 unique characters with assigned indices : {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4}
        # suppose we have a batch size of B=2 and block size of T=4 : Sequence 1: "abcd" → Encoded as [0, 1, 2, 3] ,Sequence 2: "bcde" → Encoded as [1, 2, 3, 4]
        ## so idx will look like : idx = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]]) 
        
        ## token embedding : convert the input tensor 'idx' to token embedding. convert every number to 32 dim vector embedding
        token_embedd=self.token_embedding_table(idx)
        ## since idx = (B,T) and due to vector embedding it becomes (B,T,n_embed)
        
        ## posotional embedding : assign 32 dim vector embedding to each position in a sequence of token
        pos_embed=self.position_embedding_table(torch.arange(T,device=device))  # (1,T,n_embed)
        
        ## adding position embedding (1,T,n_embed) and token_Embedding (B,T,n_embed) ----> x (B,T,n_embed)
        ## note : x will have knowlegde what the token is also remembering postion of token in sequence
        x= token_embedd+pos_embed # B,T,C
        
        ## im_head is a fuly connected layer like of Dense layer in tensorflow : it will convert input embedding (B,T,n_embed) to output ie, logits (B,T,C) ## here C is vocab_Size
        ## note: im_head is a linear layer used to map input embedding to output 'logits'
        logits=self.im_head(x) ## it gives B,T,vocab_Size
        
        if targets==None:
            loss=None
        else:
            ## extract batch_size (B), Time_step(T) and embedding_vector_size(C)
            B,T,C = logits.shape
            ## flatten logits to pass to model
            logits=logits.view(B*T,C)
            ## flattedn targets to 1d
            targets=targets.view(B*T)
            
            ## note : here we are flattening logits (B,T,C) ie, 3d to 2d (B*T,C)
            ##        and targets (B,T) ie, 2d to 1d(B*T)
            ##        because we are using cross entropy function for calculating loss
            ##        cross_entropy loss expects logits as a 2d tensor and target as 1d tensor
            
            ## calculating loss
            loss=F.cross_entropy(logits,targets)
        
        return logits,loss
    
    
    ## generate function takes starting tensor as input (it can be a just single character ex, H -> [0] ) and repeatedly generate next character for this input tensor and append it to idx
    def generate(self,idx,max_new_token):
        ## idx is (B,T) array of indices in the current context
        for _ in range(max_new_token):
            
            ## get the prediction
            logits,loss=self(idx)
            
            ## Gets the logits (prediction of next character) only for the last token in the sequence due to which shape of logits becomes (B,C) , here c->vocab_Size
            logits=logits[:,-1,:] # becomes (B,C)
            
            ## apply softmax to convert logits to probabilities
            probs=F.softmax(logits,dim=-1) # (B,C)
            
            # randomly picks a character based on probabilities.
            idx_next=torch.multinomial(probs,num_samples=1) # (B,1)
            
            ## append the new token to the running sequence ie, idx
            idx=torch.cat((idx,idx_next),dim=1) #(B,T+1)
            
        return idx
            
model=BigramLanguageModel()
m=model.to(device)
    
## create a pytorch optimiser
optimizer=torch.optim.AdamW(model.parameters(),lr=learning_rate)

for iter in range(max_iters):
    ## every once in a while evaluate the loss on train adn val sets
    if iter % eval_interval==0:
        losses=estimate_loss()
        print("step ",iter," :train loss ",losses['train']," ,test loss ",losses['test']," :")
    
    # sample a batch of data
    xb,yb=get_batch('train')
    
    #evaluate the loss by passing x_train (xb) and y_train (yb) to model
    logits,loss=model(xb,yb)
    
    # zero out the gradient
    optimizer.zero_grad(set_to_none=True)
    
    # compute gradient
    loss.backward()
    
    # update model parameter
    optimizer.step()
    
# generate form model
## now after minimising the loss if again try model to print next character for the character embedded as zero -> it gives better result
context=torch.zeros((1,1),dtype=torch.long,device=device)
print(decode(m.generate(context,max_new_token=500)[0].tolist()))
# print(context)


    