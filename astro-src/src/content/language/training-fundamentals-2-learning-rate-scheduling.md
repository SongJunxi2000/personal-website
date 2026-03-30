---
title: "Training Fundamentals (2/4) — Learning Rate Scheduling"
date: "2026-03-27"
excerpt: "Step decay, exponential decay, cosine annealing, and the warmup + cosine pattern that became the standard for transformer training."
project: language
readingTime: 6
---

The learning rate η is a single scalar that controls the step size of every parameter update. A learning rate **scheduler** is the rule that determines how η changes over the course of training. This post covers the major scheduling strategies and why warmup + cosine decay became the default for transformers.

---

**Q: Why not just use a constant learning rate?**

Early in training, parameters are far from any minimum — large steps are efficient. Late in training, parameters are near a good minimum — large steps overshoot and bounce. A fixed learning rate forces a compromise: too high and training is unstable late; too low and training is painfully slow early. Schedulers resolve this by adapting η over time.

---

**Q: What is step decay?**

Drop the learning rate by a fixed factor at predetermined epochs:

```
Epoch  1–30:   η = 0.1
Epoch 31–60:   η = 0.01    (÷10)
Epoch 61–90:   η = 0.001   (÷10)
```

Simple and effective for CNNs (the ResNet/VGG era). The downside is you have to hand-pick the milestones.

```python
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
```

---

**Q: What is exponential decay?**

Multiply the learning rate by a constant factor γ each epoch:

$$\eta_t = \eta_0 \times \gamma^t \quad \text{where } 0 < \gamma < 1$$

With η₀ = 0.1 and γ = 0.95: at epoch 10, η = 0.0598; at epoch 50, η = 0.0077. Smooth decay with no milestones to pick, but it decays relentlessly and can shrink too fast.

```python
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
```

---

**Q: What is cosine annealing?**

The learning rate follows a half-cosine curve:

$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})(1 + \cos(\pi \cdot t / T))$$

Where t = current step and T = total steps.

With η_max = 1e-3, η_min = 1e-5, T = 1000:

| Step | cos term | η |
|------|----------|---|
| 0 | cos(0) = 1 | 1e-3 |
| 250 | cos(π/4) ≈ 0.71 | ~8.5e-4 |
| 500 | cos(π/2) = 0 | ~5.05e-4 |
| 750 | cos(3π/4) ≈ −0.71 | ~1.5e-4 |
| 1000 | cos(π) = −1 | 1e-5 |

The key property: decays slowly at first, faster in the middle, slowly again near the end. This gives the optimizer time to explore early and settle gently late.

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=1000, eta_min=1e-5
)
```

---

**Q: What is warmup + cosine decay, and why is it the modern standard?**

This is the schedule used in GPT, LLaMA, and most modern LLM training. Two phases:

1. **Warmup** (steps 0 → warmup_steps): Linearly ramp η from 0 to η_max.
2. **Cosine decay** (steps warmup_steps → total_steps): Cosine anneal from η_max to η_min.

**Why warmup?** At initialization, weights are random. Adam's moment estimates (m, v) haven't calibrated yet — the first few gradient estimates are noisy. A large η on noisy estimates can cause irreversibly bad early updates. Warmup gives Adam time to build reliable statistics.

Implementation from scratch:

```python
import math

def get_lr(step, warmup_steps, total_steps, lr_max, lr_min):
    if step < warmup_steps:
        # Linear warmup: 0 → lr_max
        return lr_max * step / warmup_steps
    else:
        # Cosine decay: lr_max → lr_min
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))
```

Worked example (warmup_steps=200, total_steps=5000, η_max=3e-4, η_min=1e-5):

| Step | Phase | η |
|------|-------|---|
| 0 | Warmup | 0 |
| 100 | Warmup | 1.5e-4 |
| 200 | Peak | 3e-4 |
| 2600 | Cosine decay | ~1.53e-4 |
| 5000 | End | 1e-5 |

---

**Q: What does a full training loop look like with warmup + cosine + gradient clipping?**

```python
import math
import torch

model = MyModel()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
loss_fn = torch.nn.CrossEntropyLoss()

# Schedule config
warmup_steps = 200
total_steps = 5000
lr_max = 3e-4
lr_min = 1e-5
max_grad_norm = 1.0

def get_lr(step):
    if step < warmup_steps:
        return lr_max * step / warmup_steps
    else:
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))

global_step = 0
for epoch in range(num_epochs):
    for batch_x, batch_y in dataloader:

        # Set LR for this step
        lr = get_lr(global_step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Forward
        predictions = model(batch_x)
        loss = loss_fn(predictions, batch_y)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (before optimizer step)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

        # Update
        optimizer.step()

        global_step += 1
```

---

**Q: What is gradient clipping and why is it here?**

If the total gradient norm exceeds a threshold, rescale the entire gradient vector so its norm equals that threshold. Direction is preserved, magnitude is capped.

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

This prevents exploding gradients from causing catastrophically large updates, especially during early training. It must be called after `loss.backward()` and before `optimizer.step()`.

---

**Q: Does the scheduler affect weight decay in AdamW?**

Yes. In the standard AdamW formula, η multiplies both the gradient term and the weight decay term. When the scheduler reduces η, weight decay also weakens. Some implementations decouple them, but the default couples them through η.

---

**Q: Does the scheduler operate per step or per epoch?**

For transformers, **per step** is standard. Older CNN schedules (step decay) often operated per epoch. The distinction matters — an epoch can be thousands of steps, so per-epoch scheduling is much coarser.
