# Token-Shuffle-MLLMs

<div align="center">
  
![Token-Shuffle Architecture](https://github.com/user-attachments/assets/4006d117-adcb-4ae0-b6c9-018a3da74e3d)

**Efficient High-Resolution Image Generation via Token Shuffling in Multimodal LLMs**

[![arXiv](https://img.shields.io/badge/arXiv-2504.17789-b31b1b.svg)](https://arxiv.org/abs/2504.17789)

</div>

## 📑 Table of Contents

- [Overview](#overview)
- [Technical Approach](#technical-approach)
  - [Visual Tokenization](#visual-tokenization)
  - [Token-Shuffle Mechanism](#token-shuffle-mechanism)
- [How It Works (In-Depth)](#how-it-works-in-depth)
  - [Step-by-Step Walkthrough](#step-by-step-walkthrough)
  - [Mathematical Formulation](#mathematical-formulation)
  - [Dry Run Example](#dry-run-example)
- [Implementation Details](#implementation-details)

## 🔍 Overview

Token-Shuffle presents a breakthrough approach for efficient high-resolution image generation using autoregressive (AR) Multimodal Large Language Models (MLLMs). By addressing the fundamental challenge of processing lengthy visual token sequences, we enable MLLMs to generate stunning 2048×2048 images with significantly reduced computational overhead.

Our approach leverages the inherent redundancy in visual token representations to compress sequences during transformer processing, then decompress them afterward—all while preserving image quality and detail. This technique enables a lightweight 2.7B model to outperform much larger diffusion and AR models on standard benchmarks.


## 🔬 Technical Approach

### Visual Tokenization

MLLMs process images by converting them into "visual tokens"—discrete or continuous representations of image patches:

#### Discrete vs. Continuous Visual Tokens

| **Discrete Tokens** | **Continuous Tokens** |
|---------------------|------------------------|
| Fixed integers from a finite vocabulary | High-dimensional vectors |
| Compatible with standard LLM architectures | Require architectural modifications |
| Used in models like LlamaGen | Offer potentially better quality |
| Example: Cat's eye = token "1234" | Example: Cat's eye = [0.1, -0.3, 0.5, ...] |

### Token-Shuffle Mechanism

Our core innovation addresses the dimensional redundancy problem in visual tokens through a two-stage process:

1. **Token-Shuffle (Merge)**:
   - Combine spatially adjacent tokens (e.g., 2×2 window) into a single token along channel dimension
   - Apply dimensional reduction via MLP to maintain transformer compatibility
   - Reduce token count quadratically (4× fewer for 2×2 window)

2. **Token-Unshuffle (Split)**:
   - After transformer processing, expand dimensions via MLP
   - Split merged tokens back to original count and arrangement
   - Preserve spatial relationships for reconstruction


## 🧩 How It Works (In-Depth)

### Step-by-Step Walkthrough

#### Step 1: Patch Extraction (Input Image → Patches)
Starting with a sample 4×4 image, split into 2×2 patches and flatten:

```
Original Image:
[[ 1,  2,  3,  4],
 [ 5,  6,  7,  8],
 [ 9, 10, 11, 12],
 [13, 14, 15, 16]]

Patches (2×2):
1. Patch 1: [1, 2, 5, 6]  
2. Patch 2: [3, 4, 7, 8]  
3. Patch 3: [9, 10, 13, 14]  
4. Patch 4: [11, 12, 15, 16]  
```
*(Each patch becomes a flattened 4-dimensional vector)*

#### Step 2: Token-Shuffle (Merge Patches)
With a shuffle window size of 2, merge adjacent patches:

```
Merged Token A: [1, 2, 5, 6, 3, 4, 7, 8]  (Patch 1 + Patch 2)  
Merged Token B: [9, 10, 13, 14, 11, 12, 15, 16]  (Patch 3 + Patch 4)  
```

#### Step 3: Transformer Processing
The transformer processes these merged tokens, requiring only half the computational resources.

#### Step 4: Token-Unshuffle (Split Back)
Split transformed tokens back to their original arrangement:

```
Patch 1': [a1, a2, a3, a4] → Reshaped to 2×2  
Patch 2': [a5, a6, a7, a8] → Reshaped to 2×2  
Patch 3': [b1, b2, b3, b4] → Reshaped to 2×2  
Patch 4': [b5, b6, b7, b8] → Reshaped to 2×2  
```

### Mathematical Formulation

#### Token Reduction Factor
$$\text{Token Count After Shuffle} = \frac{\text{Original Token Count}}{s^2}$$

#### Computational Savings
Self-attention FLOPs drop by approximately $s^4$ (since FLOPs ∝ $n^2$, and $n → n/s^2$)

#### Dimension Transformations
- **Shuffle**: $[s^2 × d] → \text{MLP} → [d]$
- **Unshuffle**: $[d] → \text{MLP} → [s^2 × d]$

### Dry Run Example

For a 1024×1024 image with 16×16 pixel patches:
- Original token count: 4096 (64×64 grid)
- Embedding dimension (d): 3072
- Shuffle window (s): 2

**Token-Shuffle Process**:
1. Group every 2×2=4 tokens → 4096÷4 = 1024 tokens
2. Each merged token: 4×3072 = 12288 dimensions → MLP → 3072 dimensions

**Transformer Processing**:
- Input: 1024×3072 (instead of 4096×3072)
- Computational savings: ~16× fewer FLOPs

**Token-Unshuffle Process**:
1. Expand 1024×3072 → 1024×12288
2. Split into original 4096×3072 arrangement

## 💻 Implementation Details

### Key Components

- **Visual Tokenizer**: VQGAN-based image encoding/decoding
- **Token-Shuffle Module**: Custom MLP layers for merging/unmerging operations
- **Backbone**: Modified LLaMA architecture with visual token support

### Architecture Specifics

```
                ┌─────────────┐
                │ Input Image │
                └──────┬──────┘
                       │
                       ▼
                ┌─────────────┐
                │   VQGAN     │
                │  Tokenizer  │
                └──────┬──────┘
                       │ 4096 tokens
                       ▼
            ┌──────────────────┐
            │   Token-Shuffle  │
            │    (2×2 merge)   │
            └────────┬─────────┘
                     │ 1024 tokens
                     ▼
            ┌──────────────────┐
            │    Transformer   │
            │      Layers      │
            └────────┬─────────┘
                     │ 1024 tokens
                     ▼
            ┌──────────────────┐
            │  Token-Unshuffle │
            │  (2×2 unmarge)   │
            └────────┬─────────┘
                     │ 4096 tokens
                     ▼
            ┌──────────────────┐
            │   VQGAN Decoder  │
            └────────┬─────────┘
                     │
                     ▼
            ┌──────────────────┐
            │   Output Image   │
            └──────────────────┘
```

## 👤 Author

For any questions or issues, please open an issue on GitHub: [@Siddharth Mishra](https://github.com/Sid3503)

---

<p align="center">
  Made with ❤️ and lots of ☕
</p>
