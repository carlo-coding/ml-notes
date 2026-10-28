from pathlib import Path
import torch

import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import time

torch.manual_seed(1337)

device = "cuda" if torch.cuda.is_available() else "cpu" 

batch_size = 32
block_size = 256
max_iters = 10_000
eval_interval = 500
eval_iters = 100
learning_rate = 4e-4
num_heads = 6
head_size = 64
dropout = 0.1
n_embd= num_heads * head_size

print("Device: ", device)

with (
    open(str(Path(__file__).parent)+"/data_train.txt", "r", encoding="utf-8") as f_train,
    open(str(Path(__file__).parent)+"/data_val.txt", "r", encoding="utf-8") as f_val,
):
    text_train = f_train.read()
    text_val = f_val.read()

print("Num of chars in dataset: ", len(text_train))
print("Num of chars in validation: ", len(text_val))

chars = sorted(list(set(text_train)))
vocab_size = len(chars)
print("Vocab: ", chars)
print("Vocab size: ", vocab_size)

# Encode and decode data
stoi = {c:i for i, c in enumerate(chars)}
itos = {i:c for i, c in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

print("Encoding and decoding working? ", decode(encode("Hello world !")) == "Hello world !")


train_data = torch.tensor(encode(text_train), dtype=torch.long)
val_data = torch.tensor(encode(text_val), dtype=torch.long)



# FUNCTION TO RETURN BATCH

def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])

    x, y = x.to(device), y.to(device)

    return x, y

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# Attention head

class Head(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.query = nn.Linear(input_dim, output_dim, bias=False)
        self.key = nn.Linear(input_dim, output_dim, bias=False)
        self.value = nn.Linear(input_dim, output_dim, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones((block_size, block_size))))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        _, T, _ = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        wei = q @ torch.transpose(k, -2, -1)
        wei = torch.masked_fill(wei, self.tril[:T, :T] == 0, float("-inf"))
        wei = wei * self.key.out_features ** -0.5
        wei = torch.softmax(wei, -1)
        # B, T, T
        wei = self.dropout(wei)

        return wei @ v

# Multihead attention

class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_input_dim, output_dim):
        if num_heads > output_dim:
            raise "MultiHeadAttention: output_dim must greater than num_heads" 
        if output_dim % num_heads != 0:
            raise "MultiHeadAttention: output_dim must be a multiple of num_heads"

        head_output_dim = output_dim // num_heads

        super().__init__()
        self.heads = nn.ModuleList([
            Head(head_input_dim, head_output_dim) for _ in range(num_heads)
        ])

        self.proj = nn.Linear(output_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = torch.cat([h(x) for h in self.heads], dim=-1)
        x = self.dropout(self.proj(x))
        return x


# Feed forward
class FeedForward(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class AttentionBlock(nn.Module):
    def __init__(self, num_heads, output_dim):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(
            num_heads = num_heads, 
            head_input_dim = output_dim, 
            output_dim = output_dim
        )
        self.feed_forward_net = FeedForward(
            output_dim
        )
        self.ln1 = nn.LayerNorm(output_dim)
        self.ln2 = nn.LayerNorm(output_dim)

    def forward(self, x):
        x = x + self.multi_head_attention(self.ln1(x))
        x = x + self.feed_forward_net(self.ln2(x))
        return x 




# Language MODEL

class LanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.attention_blocks = nn.Sequential(
            AttentionBlock(num_heads=num_heads, output_dim=n_embd),
            AttentionBlock(num_heads=num_heads, output_dim=n_embd),
            AttentionBlock(num_heads=num_heads, output_dim=n_embd),
            AttentionBlock(num_heads=num_heads, output_dim=n_embd),
        )
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets = None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, n_embd)
        x = tok_emb + pos_emb
        x = self.attention_blocks(x)        
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_chars):
        for _ in range(max_chars):
            context_block = idx[:, -block_size:]
            out, _ = self(context_block)
            probs = torch.softmax(out[:, -1, :], dim=1)
            new_idx = torch.multinomial(probs, num_samples=1)   
            idx = torch.concatenate((idx, new_idx), dim=1)
        return idx

model = LanguageModel().to(device)

# TRAIN MODEL
training_loop_text = "Model parameters: " + str(sum(p.numel() for p in model.parameters()))
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print("\n\nTraining model")
print(training_loop_text)

loss_arr = []

start_t = time.time()

for iter in range(max_iters):

    if (iter+1) % eval_interval == 0:
        now = time.time()
        losses = estimate_loss(model)
        minutes = (now - start_t) / 60
        step = f"\nStep: {iter+1}: train loss {losses["train"]:.4f}, val loss {losses["val"]:.4f}, minutes acc {minutes:.2f}"
        training_loop_text += step
        print(step, end="")


    X, Y = get_batch("train")

    out, loss = model(X, Y)

    loss_arr.append(loss.item())


    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    optimizer.step()


plt.plot(range(max_iters), loss_arr, color="red")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss during the training loop")
plt.savefig(str(Path(__file__).parent)+"./loss.png", dpi=300, bbox_inches="tight")

# TEST TRAINED MODEL
print("\n\nSample output for model")


prompt = "He wants a"
document_length = 10_000

generated_text = decode(
    map(
        int, 
        model.generate(
            torch.tensor([encode(prompt)]).to(device), document_length)[0]
        )
) 

with open(str(Path(__file__).parent)+"/generated.txt", "w", encoding="utf-8") as f:
    f.write(training_loop_text + "\n-----------\n" + generated_text)

print(generated_text[:500])