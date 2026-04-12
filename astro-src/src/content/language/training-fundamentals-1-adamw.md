---
title: "Training Fundamentals (1/4) — AdamW: Why Every LLM Uses It"
date: "2026-04-06"
excerpt: "From vanilla SGD to Adam to AdamW — how adaptive learning rates and decoupled weight decay became the default optimizer for transformer training."
project: language
readingTime: 7
---

Every major language model — GPT, LLaMA, Claude — uses some variant of AdamW as its optimizer. This post builds up to AdamW from first principles: what problem each piece solves, why the pieces fit together, and what "decoupled weight decay" actually means.

---

**Q: What's wrong with vanilla SGD?**

The simplest update rule is: subtract the gradient scaled by a learning rate.

$$\theta \leftarrow \theta - \eta \cdot g_t$$

The problem is that different parameters have very different gradient characteristics. Some get tiny gradients every step, others get wild, noisy gradients. A single learning rate can't serve them all — too big for noisy parameters, too small for quiet ones.

---

**Q: How does Adam fix this?**

Adam gives each parameter its own effective learning rate using two running statistics:

**First moment (momentum)** — smoothed direction:

$$m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t$$

This is an exponential moving average of the gradient with β₁ = 0.9 (~10 step memory). It smooths out noise: if gradients mostly point positive but occasionally flip negative, m_t stays positive. Important: with constant gradients, m_t converges to g_t — it does NOT accumulate beyond g_t. This is different from classical SGD momentum.

**Second moment** — smoothed scale:

$$v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2$$

An exponential moving average of the squared gradient with β₂ = 0.999 (~1000 step memory). It tracks how large gradients typically are for this parameter. Used to normalize the update: divide by √v_t so that parameters with large gradients take smaller steps, and parameters with small gradients take larger steps. The longer memory than m_t is intentional — gradient scale is a more stable property that shouldn't jump around.

---

**Q: What's bias correction?**

Both m_t and v_t are initialized at 0, so they're biased toward zero in early steps. If g_t = c constantly and β₁ = 0.9, then m₁ = 0.1c — way off from the true value c.

The fix:

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

At t = 1: m̂₁ = 0.1c / (1 − 0.9) = c — corrected back to the true gradient. As t → ∞, β₁ᵗ → 0 and the correction disappears (no longer needed).

---

**Q: What's the full Adam update?**

$$\theta \leftarrow \theta - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

Where ε ≈ 10⁻⁸ prevents division by zero.

---

**Q: What is weight decay and why do we need it?**

During training, weights can grow very large to overfit the training data. Weight decay discourages this — every step, shrink all weights a little toward zero:

$$\theta \leftarrow (1 - \delta) \cdot \theta$$

Big weights get pulled down harder in absolute terms. If a weight needs to be large, it must "earn" it by consistently getting gradients that push it back up.

---

**Q: What's wrong with combining weight decay and Adam naively?**

The naive approach adds weight decay to the gradient before feeding it through Adam:

$$g_t' = g_t + \lambda \cdot \theta_{t-1}$$

But then λ·θ gets divided by √v̂_t along with the real gradient. This means parameters with tiny gradients (small v_t) get **amplified** weight decay, while parameters with huge gradients (large v_t) get **suppressed** weight decay. That's wrong — weight decay is supposed to be uniform regardless of gradient history.

---

**Q: How does AdamW fix this?**

Apply weight decay **directly** to the parameter, not through Adam's adaptive scaling:

$$\theta_t = (1 - \eta\lambda) \cdot \theta_{t-1} - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

Two independent forces acting on each parameter:

- **Gradient update** (adaptive, through Adam): the second term
- **Weight decay** (simple, direct shrinkage): the first term

Weight decay is tied to η (the learning rate) because as η decreases toward end of training, both gradient updates and weight decay should wind down together. Otherwise weight decay would keep shrinking weights even after learning has stopped, undoing the training.

---

**Q: Can you show the full algorithm step by step?**

**Hyperparameters:** η (learning rate), β₁ = 0.9, β₂ = 0.999, ε = 10⁻⁸, λ (weight decay, e.g., 0.01)

**Initialize:** m₀ = 0, v₀ = 0, t = 0

**Each step:**

1. t ← t + 1
2. g_t ← ∇_θ L(θ_{t-1}) — compute gradient via backprop
3. m_t ← β₁ · m_{t-1} + (1 − β₁) · g_t — update first moment
4. v_t ← β₂ · v_{t-1} + (1 − β₂) · g_t² — update second moment
5. m̂_t ← m_t / (1 − β₁ᵗ) — bias correct first moment
6. v̂_t ← v_t / (1 − β₂ᵗ) — bias correct second moment
7. θ_t ← (1 − ηλ) · θ_{t-1} − η · m̂_t / (√v̂_t + ε) — update parameter

---

**Q: What does this look like in code?**

```python
import torch

class AdamW:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        # Initialize first and second moment for each parameter
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            g = p.grad

            # Update first moment (smoothed gradient)
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g

            # Update second moment (smoothed squared gradient)
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g ** 2

            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # AdamW update: decoupled weight decay + adaptive gradient step
            p.data = (1 - self.lr * self.weight_decay) * p.data \
                     - self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()
```

Usage:

```python
model = MyTransformer()
optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

for batch in dataloader:
    loss = model(batch)
    loss.backward()        # compute gradients
    optimizer.step()       # AdamW update
    optimizer.zero_grad()  # reset gradients for next step
```

---

**Q: Does m_t accumulate beyond g_t when gradients are constant? Like a ball rolling downhill gaining speed?**

No. With constant gradients g_t = c: m₁ = 0.1c, m₂ = 0.19c, m₃ = 0.271c, ... converges to c, never exceeds it. Adam's first moment is an exponential moving average, which is bounded by the values it averages. This is different from classical SGD momentum, where velocity CAN exceed the gradient. The real value of m_t is smoothing out noise, not acceleration.

---

**Q: So the weights are shrinking and also modified by the gradient — we want the shrinking pace to keep up with the gradient update pace?**

Exactly. When gradient updates take big steps, weight decay should pull proportionally. When gradient updates are barely moving near end of training, weight decay should barely pull too. They wind down together.
