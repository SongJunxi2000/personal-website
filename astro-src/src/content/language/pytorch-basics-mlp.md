---
title: "PyTorch Basics: Building a Two-Layer MLP"
date: "2026-03-22"
excerpt: "Q&A from Day 1 of learning NLP — understanding the MLP pattern that lives at the heart of every transformer."
project: language
readingTime: 4
---

*These are Q&As from my first day studying language modeling from scratch. The code is simple, but the pattern matters — it shows up everywhere.*

---

**Q: What does a basic two-layer MLP look like in PyTorch?**

A 2-layer MLP has two `nn.Linear` layers with a nonlinear activation between them:

```python
class SumModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 32)   # expand: 10 → 32
        self.layer2 = nn.Linear(32, 1)    # compress: 32 → 1

    def forward(self, x):
        z = F.relu(self.layer1(x))        # nonlinearity between layers
        y = self.layer2(z)
        return y
```

The model learns to compute `y = sum(x)` — trivial by design, but the architecture is the point.

---

**Q: Why does the activation function matter? Can't two linear layers do the job?**

For `y = sum(x)`, a single `nn.Linear(10, 1)` could actually learn it directly — weights converge to all-ones, bias to zero. We use two layers to practice the MLP pattern. But here's the critical thing: **without a nonlinearity, two linear layers collapse into one**.

Matrix multiplication is associative: `W2 @ (W1 @ x) = (W2 @ W1) @ x`. That's just one linear transformation with a different weight matrix. No matter how deep you stack linear layers, you can always simplify them to a single linear map. The activation function (ReLU, GELU, etc.) breaks this — it introduces structure that can't be collapsed away, letting the network learn richer intermediate representations.

---

**Q: What does the training loop look like?**

```
model = SumModel()
optimizer = Adam(model.parameters(), lr=1e-3)
loss_fn = MSELoss()

for each batch of random x (shape: 100×10):
    optimizer.zero_grad()           # clear old gradients
    target = x.sum(dim=1)          # true y = sum(x)
    pred = model(x)                # forward pass
    loss = loss_fn(pred, target)   # compute loss
    loss.backward()                 # compute new gradients
    optimizer.step()                # update weights
```

Five steps, always in the same order: **zero_grad → forward → loss → backward → step**.

`zero_grad` first because PyTorch accumulates gradients by default. Skip it and your gradients compound across batches.

---

**Q: Why does this toy example actually matter? What does it have to do with transformers?**

Every transformer FFN (feed-forward network) block is exactly this pattern — `Linear → activation → Linear` — wrapped in a residual connection. The hidden dimension (32 here, typically `4 × d_model` in real transformers) lets the network expand into a richer space before compressing back.

The specific task (`sum(x)`) is irrelevant. What you're learning is the loop, the shapes, the five-step rhythm. In transformers, attention handles token mixing across positions; the FFN block handles per-token transformation. That second part is exactly this — just much wider.

---

**Q: Any efficiency insight worth knowing from day one?**

When you work with the full corpus directly, processing is slow because you're iterating over every token on every pass. The key insight for BPE training (and for thinking about neural training generally) is to work with **frequency counts over unique patterns** rather than raw data.

For example: if the word `" the"` appears 5,000 times, you process it once and multiply by 5,000 — instead of processing it 5,000 times. This turns an O(corpus size) operation into an O(unique patterns) operation, which can be orders of magnitude faster.

The same intuition applies to mini-batching in PyTorch: you never process the whole dataset in one shot, you work with compressed representations of it.

---
