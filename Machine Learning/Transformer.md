# Transformer

## 架构





<img src="https://cdn.jsdelivr.net/gh/J-M-LIU/pic-bed@master//img/image-20241210114549589.png" alt="image-20241210114549589" style="zoom:40%;" />





## 示例代码

```python
import torch
import torch.nn as nn
import torch.optim as optim
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model) # 存储位置编码
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :] # 输入特征 + 位置信息

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def attention(self, q, k, v, mask=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        weights = torch.nn.functional.softmax(scores, dim=-1)
        return torch.matmul(weights, v), weights

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        q = self.q_linear(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        scores, _ = self.attention(q, k, v, mask)
        concat = scores.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)
        return self.out(concat)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.layer_norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.layer_norm2(x + self.dropout(ff_output))
        return x

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, dropout, max_len):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model) # 嵌入层将离散词汇索引映射到连续向量表示
        self.positional_encoding = PositionalEncoding(d_model, max_len) # 位置编码
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, n_layers, n_heads, d_ff, dropout, max_len):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab_size, d_model, n_layers, n_heads, d_ff, dropout, max_len)
        self.decoder = Encoder(tgt_vocab_size, d_model, n_layers, n_heads, d_ff, dropout, max_len)  # Simplified
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask, tgt_mask):
        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(tgt, tgt_mask)
        output = self.fc_out(dec_output)
        return output

# Training Process
def train_transformer():
    # Hyperparameters
    src_vocab_size = 10000
    tgt_vocab_size = 10000
    d_model = 512
    n_layers = 6
    n_heads = 8
    d_ff = 2048
    dropout = 0.1
    max_len = 100
    batch_size = 32
    learning_rate = 0.0001
    num_epochs = 10

    # Model, Loss, Optimizer
    model = Transformer(src_vocab_size, tgt_vocab_size, d_model, n_layers, n_heads, d_ff, dropout, max_len)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Dummy data
    src = torch.randint(1, src_vocab_size, (max_len, batch_size))
    tgt = torch.randint(1, tgt_vocab_size, (max_len, batch_size))
    src_mask = torch.ones((batch_size, 1, max_len))
    tgt_mask = torch.ones((batch_size, 1, max_len))

    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output = model(src, tgt, src_mask, tgt_mask)
        loss = criterion(output.view(-1, tgt_vocab_size), tgt.view(-1))
        loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

if __name__ == "__main__":
    train_transformer()

```





![](https://jalammar.github.io/images/t/transformer_decoding_2.gif)