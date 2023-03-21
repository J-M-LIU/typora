**判断模型和数据是否在gpu上**

```python
model = nn.LSTM(input_size=10, hidden_size=4, num_layers=1, batch_first=True)
print(next(model.parameters()).device)
data = torch.ones([2, 3])
print(data.device) 
```

**判断mps是否可用**

```python
print(torch.backends.mps.is_available())
```

**将数据和模型分配到指定位置**

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
同
if torch.cuda.is_available():
	device = torch.device("cuda")
else:
	device = torch.device("cpu")
data = data.to(device)
model = Model(...).to(device)
```

