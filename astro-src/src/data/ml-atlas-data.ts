/* ─── ML Atlas Data ─────────────────────────────────────────
   Extracted from Draft/ml-atlas.html
   All models, modules, categories, and graph structure.
   ──────────────────────────────────────────────────────────── */

export interface ModelCategory {
  id: string;
  label: string;
}

export interface ModuleCategory {
  id: string;
  label: string;
  desc: string;
}

export interface Module {
  id: string;
  name: string;
  cat: string;
  desc: string;
  where: string;
  why: string;
}

export interface Model {
  id: string;
  cat: string;
  name: string;
  year: string;
  author: string;
  desc: string;
  why: string;
  arch: string;
}

export interface GraphNode {
  label: string;
  modelIds?: string[];
  children?: GraphNode[];
}

export const MODEL_CATEGORIES: ModelCategory[] = [
  { id: "baseline",             label: "Baseline" },
  { id: "vision-cnn",           label: "Vision · CNN era" },
  { id: "vision-transformer",   label: "Vision · Transformer era" },
  { id: "language-pre",         label: "Language · pre-Transformer" },
  { id: "language-transformer", label: "Language · Transformer family" },
  { id: "generative",           label: "Generative" },
  { id: "rl",                   label: "Reinforcement Learning" },
];

export const MODULE_CATEGORIES: ModuleCategory[] = [
  { id: "activation",     label: "Activations",           desc: "Element-wise non-linearities. Without them, stacked Linears collapse into one." },
  { id: "linear",         label: "Linear / Projection",   desc: "Learned matrix transforms. The workhorse of almost every model." },
  { id: "normalization",  label: "Normalization",         desc: "Rescale activations for stable training." },
  { id: "regularization", label: "Regularization",        desc: "Prevent overfitting. Mostly dropout." },
  { id: "convolution",    label: "Convolutions",          desc: "Spatial filters. The foundation of classical vision." },
  { id: "pooling",        label: "Pooling / Sampling",    desc: "Downsample or upsample spatial feature maps." },
  { id: "attention",      label: "Recurrence & Attention", desc: "Modules for sequences — step-by-step (RNN) or all-at-once (attention)." },
  { id: "positional",     label: "Positional encodings",  desc: "Inject position information into otherwise position-invariant attention." },
  { id: "loss",           label: "Losses",                desc: "Technically nn.Modules too. How models are scored during training." },
  { id: "composite",      label: "Composite blocks",      desc: "Named patterns built from primitives. Not base PyTorch classes." },
];

export const CAT_HUE: Record<string, number> = {
  activation: 8, linear: 40, normalization: 188, regularization: 278,
  convolution: 140, pooling: 98, attention: 218, positional: 322, loss: 352,
  composite: 28,
};

export const CAT_SAT: Record<string, number> = { default: 45, composite: 28 };

export const MODULES: Module[] = [
  // ── Activations ──
  { id: "relu",      name: "nn.ReLU",      cat: "activation", desc: "max(0, x). Zero out negatives, pass positives.", where: "Default in CNNs, MLPs, original Transformer FFN.", why: "Fast, cheap, no vanishing gradient for positives." },
  { id: "gelu",      name: "nn.GELU",      cat: "activation", desc: "x · Φ(x). A smooth ReLU using the Gaussian CDF.", where: "BERT, GPT-2/3, ViT — Transformers pre-LLaMA.", why: "Smoother gradients; empirically better for Transformers." },
  { id: "silu",      name: "nn.SiLU",      cat: "activation", desc: "x · sigmoid(x). Also called Swish.", where: "LLaMA and derivatives (inside SwiGLU).", why: "Current favorite for modern LLMs; cheaper than GELU." },
  { id: "tanh",      name: "nn.Tanh",      cat: "activation", desc: "Squashes input to [−1, 1].", where: "LeNet, LSTM gates, AlphaZero value head.", why: "Zero-centered; useful for bounded outputs." },
  { id: "sigmoid",   name: "nn.Sigmoid",   cat: "activation", desc: "Squashes to (0, 1).", where: "Binary classification, LSTM / GAN gating.", why: "Reads as probability." },
  { id: "softmax",   name: "nn.Softmax",   cat: "activation", desc: "Turns a vector into a probability distribution.", where: "Classifier output, attention weights, policy heads.", why: "Often fused into CrossEntropyLoss." },
  { id: "leakyrelu", name: "nn.LeakyReLU", cat: "activation", desc: "Small non-zero gradient for negative inputs.", where: "GAN discriminators, some niche CNNs.", why: "Avoids 'dying ReLU'." },
  { id: "swiglu",    name: "SwiGLU",       cat: "activation", desc: "Gated FFN: (Linear_gate · SiLU) ⊙ Linear_up → Linear_down.", where: "LLaMA, Mistral, modern LLM FFN blocks.", why: "Outperforms plain Linear→SiLU→Linear." },

  // ── Linear / Projection ──
  { id: "linear",     name: "nn.Linear",    cat: "linear", desc: "y = xWᵀ + b. The fundamental learnable transform.", where: "Everywhere — classifier heads, attention Q/K/V, FFN.", why: "If you learn one module, learn this." },
  { id: "embedding",  name: "nn.Embedding", cat: "linear", desc: "Lookup table mapping integer IDs → dense vectors.", where: "Token embeddings in every language model.", why: "Turns discrete tokens into continuous vectors." },
  { id: "flatten",    name: "nn.Flatten",   cat: "linear", desc: "Reshapes multi-dim tensor to 1D per sample.", where: "Between CNN feature maps and final Linear head.", why: "Linear expects flat input." },
  { id: "patchembed", name: "PatchEmbed",   cat: "linear", desc: "Splits image into patches, flattens each, projects with Linear.", where: "Vision Transformer, multimodal models.", why: "Turns an image into a sequence of tokens." },
  { id: "identity",   name: "nn.Identity",  cat: "linear", desc: "Returns input unchanged.", where: "Placeholder when conditionally skipping a layer.", why: "Handy for swapping out heads." },

  // ── Normalization ──
  { id: "batchnorm2d", name: "nn.BatchNorm2d", cat: "normalization", desc: "Normalize each channel across the batch dimension.", where: "CNNs — ResNet, modern ConvNets.", why: "Stabilizes training; allows higher LR." },
  { id: "layernorm",   name: "nn.LayerNorm",   cat: "normalization", desc: "Normalize across feature dim, per sample.", where: "Every standard Transformer.", why: "Works at any batch size." },
  { id: "rmsnorm",     name: "nn.RMSNorm",     cat: "normalization", desc: "Simpler LayerNorm: divide by RMS only.", where: "LLaMA, Mistral, Qwen, DeepSeek, Grok.", why: "Fewer ops, similar quality." },
  { id: "groupnorm",   name: "nn.GroupNorm",   cat: "normalization", desc: "Normalize across subsets of channels.", where: "U-Nets in diffusion models.", why: "Works with small batch sizes." },

  // ── Regularization ──
  { id: "dropout", name: "nn.Dropout", cat: "regularization", desc: "Randomly zero fraction p of activations during training.", where: "AlexNet, VGG, Transformer outputs.", why: "Prevents neuron co-adaptation." },

  // ── Convolutions ──
  { id: "conv2d",          name: "nn.Conv2d",          cat: "convolution", desc: "Slides a small learnable filter over a 2D feature map.", where: "Every CNN — LeNet through ResNet.", why: "Exploits spatial locality efficiently." },
  { id: "convtranspose2d", name: "nn.ConvTranspose2d", cat: "convolution", desc: "Learnable upsampling — inverse spatial pattern of Conv2d.", where: "GAN generators, U-Net decoders.", why: "Lets the network learn how to upsample." },

  // ── Pooling / Sampling ──
  { id: "maxpool2d",         name: "nn.MaxPool2d",         cat: "pooling", desc: "Max over each window.", where: "LeNet, AlexNet, VGG, ResNet, U-Net.", why: "Cheap downsampling; mild translation invariance." },
  { id: "avgpool2d",         name: "nn.AvgPool2d",         cat: "pooling", desc: "Average over each window.", where: "LeNet-5, some ResNet variants.", why: "Smoother than max pooling." },
  { id: "adaptiveavgpool2d", name: "nn.AdaptiveAvgPool2d", cat: "pooling", desc: "Average-pools to a fixed output size regardless of input.", where: "ResNet, modern CNNs.", why: "Accepts arbitrary image sizes." },
  { id: "upsample",          name: "nn.Upsample",          cat: "pooling", desc: "Nearest or bilinear upsampling. Not learnable.", where: "U-Net decoder, diffusion U-Net.", why: "Cheap alternative to ConvTranspose2d." },

  // ── Recurrence & Attention ──
  { id: "lstm",               name: "nn.LSTM",               cat: "attention", desc: "Recurrent cell with gates + cell state.", where: "Seq2Seq (pre-2017), speech, time-series.", why: "Handles long sequences better than vanilla RNN." },
  { id: "gru",                name: "nn.GRU",                cat: "attention", desc: "Simpler LSTM variant (fewer gates).", where: "Same use cases as LSTM.", why: "Often competitive at lower cost." },
  { id: "rnn",                name: "nn.RNN",                cat: "attention", desc: "Simplest recurrent unit.", where: "Mostly educational now.", why: "Foundational concept." },
  { id: "multiheadattention", name: "nn.MultiheadAttention", cat: "attention", desc: "Scaled dot-product attention across multiple heads.", where: "Every Transformer.", why: "The core token-mixing op." },
  { id: "causalattention",    name: "Causal / Masked MHA",   cat: "attention", desc: "MHA with a mask blocking future positions.", where: "GPT decoders, causal language models.", why: "Enables autoregressive generation." },
  { id: "crossattention",     name: "CrossAttention",        cat: "attention", desc: "Q from one sequence, K/V from another.", where: "Encoder-decoder Transformers, Stable Diffusion.", why: "How decoder looks at encoder / image conditions on text." },
  { id: "gqa",                name: "GQA (Grouped Query)",   cat: "attention", desc: "Fewer K/V heads than Q heads, shared across groups.", where: "LLaMA 2+, Mistral.", why: "Saves KV-cache memory at inference." },

  // ── Positional encodings ──
  { id: "posenc-sinus",   name: "PositionalEncoding (sinusoidal)", cat: "positional", desc: "Fixed sine/cosine patterns added to embeddings.", where: "Original Transformer.", why: "Injects position without learned params." },
  { id: "posembed-learn", name: "Learned PositionalEmbedding",     cat: "positional", desc: "Embedding indexed by position ID.", where: "BERT, GPT-2, ViT.", why: "Learns position info from data." },
  { id: "rope",           name: "RoPE (Rotary Position)",          cat: "positional", desc: "Rotates Q/K vectors by position-dependent angle.", where: "LLaMA, Mistral, modern LLMs.", why: "Encodes relative position; extrapolates better." },
  { id: "relposbias",     name: "Relative Positional Bias",        cat: "positional", desc: "Learned bias on attention logits by relative distance.", where: "T5, Swin Transformer.", why: "Relative beats absolute for many tasks." },

  // ── Losses ──
  { id: "crossentropyloss",  name: "nn.CrossEntropyLoss",  cat: "loss", desc: "LogSoftmax + NLL fused for numerical stability.", where: "Multi-class classification, next-token prediction.", why: "Standard LM loss. Expects raw logits." },
  { id: "bcewithlogitsloss", name: "nn.BCEWithLogitsLoss", cat: "loss", desc: "Sigmoid + BCE, fused.", where: "Binary/multi-label classification, GAN discriminators.", why: "Numerically stable BCE on logits." },
  { id: "mseloss",           name: "nn.MSELoss",           cat: "loss", desc: "(pred − target)² averaged.", where: "Regression, diffusion noise prediction, autoencoder recon.", why: "Standard regression loss." },
  { id: "kldivloss",         name: "nn.KLDivLoss",         cat: "loss", desc: "KL divergence between two distributions.", where: "Distillation, VAE KL term, PPO/GRPO.", why: "Measures distribution distance." },
  { id: "contrastiveloss",   name: "Contrastive loss",     cat: "loss", desc: "Pairs close, non-pairs far.", where: "CLIP, SimCLR, embedding models.", why: "Aligns two modalities." },

  // ── Composite blocks ──
  { id: "residualblock",    name: "ResidualBlock",      cat: "composite", desc: "Conv → BN → ReLU → Conv → BN → (+input) → ReLU.", where: "ResNet, AlphaZero trunk.", why: "Skip connection lets gradients flow." },
  { id: "ffn",              name: "FFN / MLP block",    cat: "composite", desc: "Linear → activation → Linear (hidden dim 4× model dim).", where: "Every Transformer block, standalone MLPs.", why: "Per-token computation; complements attention." },
  { id: "transformerblock", name: "TransformerBlock",   cat: "composite", desc: "LN → Attention → residual → LN → FFN → residual.", where: "The repeated unit in every Transformer.", why: "Alternates token-mixing and channel-mixing." },
  { id: "unet-enc",         name: "U-Net Enc/Dec",      cat: "composite", desc: "Downsampling + upsampling paths with skip connections.", where: "Segmentation, diffusion denoising networks.", why: "Skips preserve spatial detail." },
  { id: "vae-enc",          name: "VAE Encoder",        cat: "composite", desc: "Outputs (μ, log σ²) over latents.", where: "VAEs, Stable Diffusion.", why: "Probabilistic encoding enables sampling." },
  { id: "vae-dec",          name: "VAE Decoder",        cat: "composite", desc: "Reconstructs data from a sampled latent z.", where: "VAEs, Stable Diffusion.", why: "Pair with encoder for generation." },
  { id: "gan-gen",          name: "GAN Generator",      cat: "composite", desc: "Noise → Linear/ConvTranspose2d stack → fake sample.", where: "GAN family.", why: "Learns to fool the discriminator." },
  { id: "gan-disc",         name: "GAN Discriminator",  cat: "composite", desc: "Sample → Conv2d stack → real/fake.", where: "GAN family.", why: "Adversary that forces the generator to improve." },
  { id: "policy-head",      name: "Policy head",        cat: "composite", desc: "Linear → Softmax over actions.", where: "AlphaZero, policy-gradient RL.", why: "Distribution over moves/actions." },
  { id: "value-head",       name: "Value head",         cat: "composite", desc: "Linear → Tanh scalar in [−1, 1].", where: "AlphaZero, actor-critic agents.", why: "Estimates state value." },
];

export const MODELS: Model[] = [
  { id: "mlp-mnist", cat: "baseline", name: "MLP (MNIST)", year: "—", author: "textbook",
    desc: "The 'hello world' of deep learning — classify 28×28 handwritten digits into 10 classes.",
    why: "A good first thing to build end-to-end.",
    arch:
`MLP
├── {flatten}                 # 28×28 → 784
├── {linear} (784 → 256)
├── {relu}
├── {linear} (256 → 128)
├── {relu}
└── {linear} (128 → 10)       # digit classes
Loss: {crossentropyloss}` },

  { id: "lenet5", cat: "vision-cnn", name: "LeNet-5", year: "1998", author: "Yann LeCun",
    desc: "The original successful CNN. Designed for reading zip codes and checks.",
    why: "Proof that CNNs work.",
    arch:
`LeNet-5
├── {conv2d} (1 → 6, 5×5)
├── {tanh} + {avgpool2d}
├── {conv2d} (6 → 16, 5×5)
├── {tanh} + {avgpool2d}
├── {flatten}
├── {linear} (400 → 120) + {tanh}
├── {linear} (120 → 84) + {tanh}
└── {linear} (84 → 10)` },

  { id: "alexnet", cat: "vision-cnn", name: "AlexNet", year: "2012", author: "Krizhevsky, Sutskever, Hinton",
    desc: "Won ImageNet 2012 by a massive margin, kicked off the deep-learning revolution.",
    why: "Introduced ReLU, Dropout, and GPU training to the mainstream.",
    arch:
`AlexNet
├── {conv2d} (3 → 96, 11×11, stride 4) + {relu} + {maxpool2d}
├── {conv2d} (96 → 256, 5×5) + {relu} + {maxpool2d}
├── {conv2d} (256 → 384, 3×3) + {relu}
├── {conv2d} (384 → 384, 3×3) + {relu}
├── {conv2d} (384 → 256, 3×3) + {relu} + {maxpool2d}
├── {flatten}
├── {linear} (9216 → 4096) + {relu} + {dropout}
├── {linear} (4096 → 4096) + {relu} + {dropout}
└── {linear} (4096 → 1000)` },

  { id: "vgg16", cat: "vision-cnn", name: "VGG-16", year: "2014", author: "Oxford VGG",
    desc: "Showed that 'just go deeper with small 3×3 filters' works.",
    why: "Elegant uniform design; still used as a feature extractor.",
    arch:
`VGG-16
├── [ {conv2d}(3×3) + {relu} ] × 2  → {maxpool2d}
├── [ {conv2d}(3×3) + {relu} ] × 2  → {maxpool2d}
├── [ {conv2d}(3×3) + {relu} ] × 3  → {maxpool2d}
├── [ {conv2d}(3×3) + {relu} ] × 3  → {maxpool2d}
├── [ {conv2d}(3×3) + {relu} ] × 3  → {maxpool2d}
├── {flatten}
├── [ {linear} + {relu} + {dropout} ] × 2
└── {linear} (→ 1000)` },

  { id: "resnet", cat: "vision-cnn", name: "ResNet-50", year: "2015", author: "Kaiming He et al.",
    desc: "The single most influential vision architecture. Introduced residual connections.",
    why: "Residual connections are now in everything — including Transformers.",
    arch:
`ResNet-50
├── {conv2d} (7×7, stride 2) + {batchnorm2d} + {relu} + {maxpool2d}
├── {residualblock} × 3        (channels: 64)
├── {residualblock} × 4        (channels: 128)
├── {residualblock} × 6        (channels: 256)
├── {residualblock} × 3        (channels: 512)
├── {adaptiveavgpool2d}
└── {linear} (→ 1000)

{residualblock:ResidualBlock}
├── {conv2d} + {batchnorm2d} + {relu}
├── {conv2d} + {batchnorm2d} + {relu}
├── {conv2d} + {batchnorm2d}
└── out = {relu}(input + above)` },

  { id: "unet", cat: "vision-cnn", name: "U-Net", year: "2015", author: "Ronneberger et al.",
    desc: "For image segmentation. Also the backbone of modern diffusion models.",
    why: "The denoising network inside Stable Diffusion.",
    arch:
`U-Net
├── Encoder (downsampling)
│   └── [ {conv2d} + {relu} + {conv2d} + {relu} + {maxpool2d} ] × 4
├── Bottleneck
└── Decoder (upsampling)
    └── [ {upsample} + concat(skip) + {conv2d} + {relu} + {conv2d} + {relu} ] × 4` },

  { id: "vit", cat: "vision-transformer", name: "ViT (Vision Transformer)", year: "2020", author: "Google",
    desc: "Replaced convolutions with Transformers for images.",
    why: "Proved Transformers work on images with enough data.",
    arch:
`ViT
├── {patchembed}                # 16×16 patches → Linear
├── prepend [CLS] token + {posembed-learn}
├── {transformerblock} × N
│   ├── {layernorm}
│   ├── {multiheadattention}    # bidirectional (no mask)
│   ├── {layernorm}
│   └── FFN: {linear} → {gelu} → {linear}
└── {linear} (from [CLS] → num_classes)` },

  { id: "word2vec", cat: "language-pre", name: "Word2Vec", year: "2013", author: "Mikolov et al.",
    desc: "Clever training around an Embedding layer.",
    why: "Showed word vectors encode semantics.",
    arch:
`Word2Vec (skip-gram)
├── {embedding} (vocab → d)
└── {linear} (d → vocab)` },

  { id: "seq2seq", cat: "language-pre", name: "LSTM Seq2Seq", year: "~2014-2017", author: "Sutskever et al. / Google",
    desc: "Pre-Transformer sequence model with LSTM encoder + decoder.",
    why: "Ran Google Translate for years.",
    arch:
`Seq2Seq
├── Encoder: {embedding} → {lstm} × N
└── Decoder: {embedding} → {lstm} × N → {linear} → vocab` },

  { id: "transformer-og", cat: "language-transformer", name: "Transformer (original)", year: "2017", author: "Vaswani et al.",
    desc: "The original encoder-decoder Transformer, designed for translation.",
    why: "The paper that started everything.",
    arch:
`Transformer
├── Encoder
│   ├── {embedding} + {posenc-sinus}
│   └── EncoderBlock × 6
│       ├── {layernorm}
│       ├── {multiheadattention}
│       ├── {layernorm}
│       └── FFN: {linear} → {relu} → {linear}
└── Decoder
    ├── {embedding} + {posenc-sinus}
    └── DecoderBlock × 6
        ├── {layernorm}
        ├── {causalattention}
        ├── {layernorm}
        ├── {crossattention}
        ├── {layernorm}
        └── FFN: {linear} → {relu} → {linear}` },

  { id: "bert", cat: "language-transformer", name: "BERT", year: "2018", author: "Google",
    desc: "Encoder-only Transformer with bidirectional attention.",
    why: "Dominated NLP benchmarks 2018-2020; still the go-to for classification and search.",
    arch:
`BERT
├── {embedding} (token + segment) + {posembed-learn}
├── {transformerblock} × 12
│   ├── {layernorm}
│   ├── {multiheadattention}
│   ├── {dropout}
│   ├── {layernorm}
│   └── FFN: {linear} → {gelu} → {linear}
└── Task heads: {linear}
Loss: {crossentropyloss}` },

  { id: "gpt", cat: "language-transformer", name: "GPT", year: "2018–", author: "OpenAI",
    desc: "Decoder-only Transformer with causal attention, trained to predict the next token.",
    why: "The dominant LLM paradigm.",
    arch:
`GPT
├── {embedding} + {posembed-learn}
├── {transformerblock} × N
│   ├── {layernorm}
│   ├── {causalattention}
│   │   ├── {linear} (Q projection)
│   │   ├── {linear} (K projection)
│   │   ├── {linear} (V projection)
│   │   └── {linear} (output projection)
│   ├── {dropout}
│   ├── {layernorm}
│   └── {ffn}
│       ├── {linear} (d → 4d)
│       ├── {gelu}
│       └── {linear} (4d → d)
├── {layernorm}
└── {linear} (LM head → vocab)
Loss: {crossentropyloss}` },

  { id: "t5", cat: "language-transformer", name: "T5", year: "2019", author: "Google",
    desc: "Encoder-decoder Transformer framing every NLP task as 'text in → text out.'",
    why: "Unified text-to-text framework.",
    arch:
`T5
├── Encoder: {embedding} + {relposbias} → EncoderBlock × N
└── Decoder: {embedding} → DecoderBlock × N
    ├── {causalattention}
    ├── {crossattention}
    └── FFN: {linear} → {relu} → {linear}
    → {linear} → vocab` },

  { id: "llama", cat: "language-transformer", name: "LLaMA / modern LLMs", year: "2023–", author: "Meta / open-weights",
    desc: "Same GPT skeleton, modernized: RMSNorm, SwiGLU, RoPE, GQA.",
    why: "The open-weights standard.",
    arch:
`LLaMA Block
├── {rmsnorm}
├── MHA with {rope} + {gqa}
│   ├── {linear} (Q)
│   ├── {linear} (K)
│   ├── {linear} (V)
│   └── {linear} (output)
├── {rmsnorm}
└── {swiglu} FFN
    ├── {linear} (gate)
    ├── {silu}
    ├── {linear} (up)
    └── {linear} (down)
Loss: {crossentropyloss}` },

  { id: "gan", cat: "generative", name: "GAN", year: "2014", author: "Ian Goodfellow",
    desc: "Two networks in adversarial competition.",
    why: "First high-quality image generation.",
    arch:
`GAN
├── {gan-gen}
│   └── noise → {linear} / {convtranspose2d} stack → fake image
└── {gan-disc}
    └── image → {conv2d} stack → {leakyrelu} → {linear} → real/fake
Loss: {bcewithlogitsloss}` },

  { id: "vae", cat: "generative", name: "VAE", year: "2013", author: "Kingma & Welling",
    desc: "Probabilistic autoencoder with a Gaussian latent space.",
    why: "Component of Stable Diffusion.",
    arch:
`VAE
├── {vae-enc} → (μ, log σ²) → sample z
└── {vae-dec} → reconstruction
Loss: {mseloss} + {kldivloss}` },

  { id: "stablediffusion", cat: "generative", name: "Stable Diffusion", year: "2022", author: "Stability AI / CompVis",
    desc: "Latent diffusion guided by CLIP text embeddings.",
    why: "Open-weights state-of-the-art image generation.",
    arch:
`Stable Diffusion
├── {vae-enc}: image → latent
├── {unet-enc} with {crossattention} to CLIP text
│   ├── {conv2d} + {groupnorm} + {silu}
│   └── ResBlocks + attention at each resolution
└── {vae-dec}: latent → image
Loss: {mseloss}` },

  { id: "clip", cat: "generative", name: "CLIP", year: "2021", author: "OpenAI",
    desc: "Two encoders aligned via contrastive learning.",
    why: "Zero-shot classification; text understander in Stable Diffusion.",
    arch:
`CLIP
├── Image encoder → {linear} projection → embedding
└── Text encoder  → {linear} projection → embedding
Loss: {contrastiveloss}` },

  { id: "alphazero", cat: "rl", name: "AlphaZero", year: "2017", author: "DeepMind",
    desc: "Self-play RL mastering Go, chess, shogi with one architecture.",
    why: "First superhuman Go without human data.",
    arch:
`AlphaZero
├── ResNet trunk
│   └── {residualblock} × N
├── {policy-head}: {linear} → {softmax}
└── {value-head}:   {linear} → {tanh}
+ Monte Carlo Tree Search` },
];

export const GRAPH: GraphNode = {
  label: "Models",
  children: [
    { label: "Baseline", modelIds: ["mlp-mnist"] },
    { label: "Vision", children: [
      { label: "CNN era",         modelIds: ["lenet5", "alexnet", "vgg16", "resnet", "unet"] },
      { label: "Transformer era", modelIds: ["vit"] },
    ]},
    { label: "Language", children: [
      { label: "pre-Transformer",    modelIds: ["word2vec", "seq2seq"] },
      { label: "Transformer family", modelIds: ["transformer-og", "bert", "gpt", "t5", "llama"] },
    ]},
    { label: "Generative", modelIds: ["gan", "vae", "stablediffusion", "clip"] },
    { label: "RL",         modelIds: ["alphazero"] },
  ]
};
