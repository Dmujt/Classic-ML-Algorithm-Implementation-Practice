#
# define the model training code
#

#%% Load the training dataset
PATH_TO_DATA = "C:/Users/dena1/Documents/shakespear-tiny.txt"

with open(PATH_TO_DATA, 'r', encoding='utf-8') as f:
    txt = f.read()
    
print("Dataset length:", len(txt))
print(txt[:200])

# Look at vocabulary of text (character-level vocab)
cchars = sorted(list(set(txt)))
print(''.join(cchars))
print(len(cchars))

#map the characters from strings to integers
stoi = { ch:i for i,ch in enumerate(cchars) }
itos = { i:ch for i,ch in enumerate(cchars) }

encode=  lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

encoded = encode("hii there")
print(encoded)
print(decode(encoded))

import torch
data = torch.tensor(encode(txt), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:100])

#%% split dataset into train/val
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

block_size = 8 #batching
train_data[:block_size + 1]

x = train_data[:block_size]
y = train_data[1:block_size + 1]
for t in range(block_size):
    context = x[:t + 1]
    target = y[t]
    print(f"when input is {context} the target: {target}")
    
#%% introduce batch dimension

torch.manual_seed(1337) #in random num generator to recreate results
batch_size = 4
block_size = 8 #max context length for preds

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

xb, yb = get_batch('train')
print('inputs:')
print(xb.shape)
print(xb)
print('targets:')
print(yb.shape)
print(yb)

for b in range(batch_size):
    for t in range(block_size):
        context = xb[b, :t+1]
        target = yb[b,t]
        print(f"when input is {context.tolist()} the target {target}")
        
#%% model set up initial 
import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx) # (B,T,C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

vocab_size =len(cchars)
m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)
print(logits.shape)
print(loss)

print(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))


#%% training set up
# create a PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

batch_size = 32
for steps in range(10000):
    
    xb, yb = get_batch('train')
    logits,loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
print(loss.item())

print(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))

#%% Getting into transformer

torch.manual_seed(42)
a = torch.tril(torch.ones(3, 3)) #triangular function that returns the lower triangle of matrix
a = a / torch.sum(a, 1, keepdim=True)
b = torch.randint(0,10,(3,2)).float()
c = a @ b
print('a=')
print(a)
print('--')
print('b=')
print(b)
print('--')
print('c=')
print(c)

torch.manual_seed(1337)
B,T,C = 4,8,2
x = torch.randn(B,T,C) #batch, time, channel
print(x.shape)

xbow = torch.zeros((B,T,C)) #bag of words for averaging all prev tokens
for b in range(B):
    for t in range(T):
        xprev = x[b,:t + 1] #everything up to and including the t token in batch
        xbow[b,t] = torch.mean(xprev, 0) #calculate average over 0th dimension
        
print(x[0]) #oth batch element


#%% version 2: using matrix multiply for a weighted aggregation
wei = torch.tril(torch.ones(T, T))
wei = wei / wei.sum(1, keepdim=True)
xbow2 = wei @ x # (B, T, T) @ (B, T, C) ----> (B, T, C)
torch.allclose(xbow, xbow2)

#%% version 3 : use softmax
tril = torch.tril(torch.ones(T, T))
wei = torch.zeros((T,T))
wei = wei.masked_fill(tril == 0, float('-inf')) #for all elements where tril is 0, make them -inf
wei = F.softmax(wei, dim=-1)
#taking softmax along every single row, results in exponentiating every single element and then sum, and after normalizing results in 1
xbow3 = wei @ x
print(xbow[0],xbow3[0])
torch.allclose(xbow, xbow3) #note: might not return True because of numerical precision differences in float


#%% Now using self-attention ;)
torch.manual_seed(1337)

B,T,C = 4, 8, 32
x = torch.randn(B,T,C)

head_size = 16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)
k = key(x) #B,T,16
q=  query(x) #B,T,16 
#all queries dot product with k to get affinities
wei = q @ k.transpose(-2, -1) #because of batch #B,T,16 @ B,16,T ---> B,T,T

tril = torch.tril(torch.ones(T,T))
wei = torch.zeros((T,T))
wei = wei.masked_fill(tril==0, float('-inf'))
wei = F.softmax(wei, dim=-1) #will aggregate high infinity items to the desired channel
v = value(x) #x is kind of private info to the token

out = wei @ v

out.shape