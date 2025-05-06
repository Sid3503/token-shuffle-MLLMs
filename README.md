# token-shuffle-MLLMs

![Image](https://github.com/user-attachments/assets/4006d117-adcb-4ae0-b6c9-018a3da74e3d)

## Simplified Explanation of Key Concepts in the Token-Shuffle Paper

### 1. **Visual Tokens in MLLMs (Multimodal Large Language Models)**
   - **What are visual tokens?**  
     Just like words are broken into tokens for text processing, images are divided into smaller parts called "visual tokens." These tokens are created using a visual tokenizer (like VQGAN), which converts image patches into discrete or continuous numerical representations.
     - **Example**: Imagine dividing a photo of a cat into 16x16 grids. Each grid is encoded into a token (e.g., "cat_head," "paw," etc.).

   - **Discrete vs. Continuous Visual Tokens**  
     - **Discrete Tokens**: Each token is a fixed integer (like a word in a dictionary). Used in models like LlamaGen because they’re compatible with LLMs.  
       *Example*: A cat’s eye might always be token "1234."  
     - **Continuous Tokens**: Tokens are flexible, high-dimensional vectors (like CLIP embeddings). They offer better quality but require complex changes to LLMs.  
       *Example*: The cat’s eye could be represented as [0.1, -0.3, 0.5, ...].

### 2. **Dimensionality Redundancy in Visual Vocabulary**
   - **Problem**: When visual tokens (e.g., 256-dim) are added to an LLM’s vocabulary (e.g., 4096-dim), most of the extra dimensions are unused ("redundant").  
     *Analogy*: Storing a small PNG image in a giant 4K JPEG—most pixels are empty.  
   - **Solution (Token-Shuffle)**:  
     - **Merge Tokens**: Combine nearby tokens (e.g., 2x2 patches) into one token along the channel dimension, reducing token count.  
       *Example*: Instead of processing 4 tokens (cat_head, paw, tail, ear), merge them into 1 "cat_parts" token.  
     - **Unmerge Later**: After processing, split the merged token back into the original 4 tokens.  
     - **Why It Works**: Visual tokens have similar patterns (e.g., fur texture repeats), so merging doesn’t lose much info.

### 3. **Token-Shuffle Operations**
   - **Token-Shuffle (Merge)**:  
     - Combine local tokens (e.g., 2x2 window) into one token.  
     - *Example*: Merging 4 tokens (16x16 patches each) into 1 token (still 16x16 but with 4x channels).  
   - **Token-Unshuffle (Split)**:  
     - Reverse the merge after Transformer processing.  
   - **Efficiency**: Reduces token count by 75% for a 2x2 window, speeding up training/inference.

### 4. **High-Resolution Challenges**
   - **Issue**: More pixels = more tokens (e.g., 1024x1024 image → 4096 tokens). LLMs struggle with long sequences.  
   - **Token-Shuffle Fix**:  
     - For 2048x2048 images, shuffle reduces 16K tokens → 4K tokens (manageable for LLMs).  
     - *Trade-off*: Larger shuffle windows (e.g., 4x4) save more compute but may blur details.

### 5. **Classifier-Free Guidance (CFG) Scheduler**
   - **Problem**: Standard CFG can distort early tokens in AR models.  
   - **Solution**: Gradually increase CFG strength during generation.  
     *Example*: Start with low guidance for rough shapes (early tokens), then ramp up for details (later tokens).

### 6. **Why This Matters**
   - **AR vs. Diffusion Models**:  
     - Diffusion models (e.g., Stable Diffusion) dominate image generation but are slow.  
     - Token-Shuffle makes AR models (like LLMs) competitive by enabling high-res generation (2048x2048) efficiently.  
   - **Results**: Their 2.7B model beats larger AR/diffusion models in benchmarks (e.g., 0.77 vs. 0.62 for LDM in GenAI-Bench).

---

## Walkthrough

### **Step 1: Patch Extraction (Input Image → Patches)**
You start with a **4x4 image**, split into **2x2 patches**, and flatten them:  
```
Original Image:
[[ 1,  2,  3,  4],
 [ 5,  6,  7,  8],
 [ 9, 10, 11, 12],
 [13, 14, 15, 16]]

Patches (2x2):
1. Patch 1: [1, 2, 5, 6]  
2. Patch 2: [3, 4, 7, 8]  
3. Patch 3: [9, 10, 13, 14]  
4. Patch 4: [11, 12, 15, 16]  
```
*(Each patch is a flattened 4-dim vector.)*

---

### **Step 2: Token-Shuffle (Merge Patches)**
**Goal**: Reduce the number of tokens by merging spatially nearby patches.  

- Suppose we use a **shuffle window size = 2**, meaning we merge **2 adjacent patches** along the channel dimension.  
- We take **Patch 1 & Patch 2** and concatenate them into a single **8-dim vector** (instead of two 4-dim vectors).  
- Similarly, merge **Patch 3 & Patch 4**.  

**Result after Token-Shuffle**:  
```
1. Merged Token A: [1, 2, 5, 6, 3, 4, 7, 8]  (Patch 1 + Patch 2)  
2. Merged Token B: [9, 10, 13, 14, 11, 12, 15, 16]  (Patch 3 + Patch 4)  
```
Now, instead of **4 tokens**, we have **2 tokens**, reducing computation in the Transformer.

---

### **Step 3: Transformer Processing**
- The merged tokens (`A` and `B`) are fed into the Transformer (e.g., LLaMA).  
- The Transformer treats them like normal tokens but processes **half as many tokens**, saving computation.  
- The Transformer outputs modified versions of `A` and `B` (let’s call them `A'` and `B'`).  

---

### **Step 4: Token-Unshuffle (Split Back into Original Patches)**
**Goal**: Recover the original structure by splitting the merged tokens.  

- Take `A' = [a1, a2, a3, a4, a5, a6, a7, a8]` and split it into two 4-dim vectors:  
  - `Patch 1' = [a1, a2, a3, a4]`  
  - `Patch 2' = [a5, a6, a7, a8]`  
- Similarly, split `B'` into `Patch 3'` and `Patch 4'`.  

**Final Output Patches**:  
```
1. Patch 1': [a1, a2, a3, a4] → Reshaped to 2x2  
2. Patch 2': [a5, a6, a7, a8] → Reshaped to 2x2  
3. Patch 3': [b1, b2, b3, b4] → Reshaped to 2x2  
4. Patch 4': [b5, b6, b7, b8] → Reshaped to 2x2  
```
These are then reassembled into the output image.

---

### **Why This Works**
1. **Efficiency**: Instead of processing 4 tokens, the Transformer processes 2, reducing compute by **~50%** (or ~75% for 4x4 windows).  
2. **Preserves Local Information**: Nearby patches (e.g., `Patch 1` and `Patch 2`) are related (e.g., same fur texture), so merging them doesn’t lose much detail.  
3. **Reversible**: The unshuffle step perfectly reconstructs the original structure.  


### **Real-World Analogy**
- **Token-Shuffle** = Zipping two JPEGs into one file for faster emailing.  
- **Token-Unshuffle** = Unzipping them back into two images after delivery.  
- The recipient (Transformer) only deals with the zipped file, saving bandwidth (compute).  

---

## Dry Run


### **1. Input/Output Dimensions**
- **Original Image Resolution**:  
  - Example: `1024x1024` image.  
- **Patch Size (Tokenizer)**:  
  - VQGAN (or similar) divides the image into patches (e.g., `16x16` pixels).  
  - For `1024x1024`, this gives `(1024/16) × (1024/16) = 64 × 64 = 4096 tokens`.  
- **Token Dimension (`d`)**:  
  - Each token is a vector of size `d` (e.g., `d = 3072` for LLaMA).  

---

### **2. Token-Shuffle Operation (Merge Tokens)**
**Goal**: Reduce the number of tokens by merging spatially local tokens.  

#### **Key Parameters**:
- **Shuffle Window Size (`s`)**:  
  - Number of adjacent tokens to merge (e.g., `s = 2` merges `2×2 = 4` tokens).  
- **Compression Factor**:  
  - Merging `s²` tokens reduces token count by `s²` (e.g., `s=2` → `4×` fewer tokens).  

#### **Steps**:
1. **Group Tokens into Local Windows**:  
   - For `s=2`, group every `2×2` tokens into a block (e.g., 4 tokens → 1 merged token).  
   - Example:  
     ```
     Original tokens (4): [T1, T2, T3, T4]  
     Merged token: [T1 ⊕ T2 ⊕ T3 ⊕ T4]  
     ```  
     *(⊕ denotes concatenation along the channel dimension.)*  

2. **Linear Projection (MLP) to Match Dimensions**:  
   - The merged token has `s² × d` values (e.g., `4 × 3072 = 12288`).  
   - An **MLP layer** compresses this back to `d` dimensions (e.g., `12288 → 3072`):  
     ```
     Merged_Projected = MLP([T1; T2; T3; T4])  
     ```  
     *(This ensures the Transformer input dimension stays `d`.)*  

3. **Residual MLP Blocks (Optional)**:  
   - Additional MLPs may refine the merged features before passing to the Transformer.  

#### **Effect on Token Count**:
- Original tokens: `4096` (for `1024x1024`).  
- After shuffling (`s=2`): `4096 / 4 = 1024 tokens`.  
- Computation in Transformer reduces **quadratically** (since self-attention is `O(n²)`).  

---

### **3. Token-Unshuffle (Split Tokens Back)**
**Goal**: Recover the original token count after Transformer processing.  

#### **Steps**:
1. **Expand Dimensions**:  
   - The Transformer outputs `1024` tokens of size `d=3072`.  
   - An **MLP** expands each token to `s² × d` (e.g., `3072 → 12288`).  

2. **Split into Original Tokens**:  
   - Reshape the expanded token into `s²` separate tokens (e.g., `12288 → [3072, 3072, 3072, 3072]`).  
   - Example:  
     ```
     Unshuffled_Tokens = Split(MLP(Transformer_Output))  
     ```  

3. **Residual MLP Blocks (Optional)**:  
   - Additional MLPs may refine the split tokens.  

#### **Final Output**:
- The `4096` tokens are reconstructed and fed to the VQGAN decoder to generate the image.  

---

### **4. Key Formulas**
1. **Token Reduction Factor**:  
   \[
   \text{Token Count After Shuffle} = \frac{\text{Original Token Count}}{s^2}
   \]  
   *(Example: `s=2` → `4096 → 1024` tokens.)*  

2. **Dimension Compression/Expansion**:  
   - **Shuffle**: `[s² × d] → MLP → [d]`  
   - **Unshuffle**: `[d] → MLP → [s² × d]`  

3. **Computational Savings**:  
   - Self-attention FLOPs drop by `~s⁴` (since `FLOPs ∝ n²`, and `n → n/s²`).  

---

### **5. Example with Numbers**
Let’s say:  
- Image: `1024x1024` → `4096` tokens (`64x64` grid).  
- `d = 3072` (LLaMA dimension).  
- `s = 2` (merge `2×2` tokens).  

**Token-Shuffle**:  
1. Merge every `4` tokens → `4096 → 1024` tokens.  
2. Each merged token: `4 × 3072 = 12288` → projected to `3072`.  

**Transformer Processes**:  
- Input: `1024 × 3072` (instead of `4096 × 3072`).  

**Token-Unshuffle**:  
1. Expand `1024 × 3072` → `1024 × 12288`.  
2. Split into `4096 × 3072`.  

---

### **Why This Works**
- **Dimensional Redundancy**: Nearby patches share similar info (e.g., sky pixels), so merging them doesn’t lose much.  
- **Efficiency**: Reduces Transformer compute drastically (e.g., `16×` fewer FLOPs for `s=2`).  
- **Reversible**: Unshuffle perfectly reconstructs the original tokens.  

This enables **high-res image generation** (e.g., `2048x2048`) without overwhelming the Transformer.  
