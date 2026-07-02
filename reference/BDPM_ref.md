## Introduction and Motivation

Traditional Denoising Diffusion Probabilistic Models (DDPMs) assume continuous data inputs and rely on Gaussian perturbations paired with Mean Squared Error (MSE) loss objectives. While effective for real-valued spaces, this framework creates a fundamental mathematical mismatch when applied to inherently discrete or binary data (such as 8-bit RGB digital images). Quantizing continuous intermediate values during Gaussian diffusion introduces undesirable artifacts, causes information loss, and yields weak convergence. Furthermore, optimizing an MSE loss trains the denoiser to predict added Gaussian noise rather than the exact discrete pixel states.

To bridge this gap, the **Binary Diffusion Probabilistic Model (BDPM)** natively processes discrete data in the binary domain. BDPM maps inputs into a binary representation space, corrupts them using precise bit-flipping via exclusive-or (**XOR**) noise operations, and optimizes a differentiable **Binary Cross-Entropy (BCE)** loss. This architecture provides fine-grained noise control, guarantees lossless binary representation for pixel-level precision, accelerates training convergence, and significantly decreases inference costs by requiring only a few sampling steps.

---

## Transform-Domain Binary Representations

BDPM operates on two distinct types of transform-domain binary embeddings depending on the target application:

### 1. Multiple Bit-Plane Representation (MBPR)

Used for image-to-image translation tasks (super-resolution, inpainting, blind restoration) where per-pixel precision is required.

* **Mechanism:** A deterministic, bijective, and fully invertible transformation $\mathcal{T}$ decomposes an 8-bit RGB image $I_{0}$ into $n$ binary bit-planes.


* **Formula:**

$$I_{0}=\sum_{k=0}^{n-1}x_{0}(k)\cdot2^{k}$$



where $x_{0}(k)\in\{0,1\}^{H\times W}$ represents the $k$-th binary coefficient plane across the image, and $n$ is the bit-depth ($n=8$ for standard images).


* **Property:** Most Significant Bits (MSBs, $k=0$) retain massive inter-pixel correlation and macro-structure, while Least Significant Bits (LSBs, $k=n-1$) are highly stochastic and noise-like.



### 2. Learnable Binary Representation (LBR)

Used for large-scale class-conditional image generation (e.g., ImageNet-1k) where generic perceptual and semantic quality matters more than precise pixel matching.

* **Mechanism:** A parametric quantized autoencoder (specifically, MAGVIT-v2 with a patch size of $16\times16$) maps the image into a lossy, highly compressed binary latent space vector $x_{0}$.



---

## Core Architecture and Mathematical Framework

```
  TRAINING PROCESS:
  [Clean Image I_0] ---> Transform T ---> [Clean Latent x_0] ---\
                                                                 XOR (⊕) ---> [Noisy Latent x_t] ---> [Denoiser Network] ---> Loss (BCE)
  [Random Noise]   ---> Mapper M_t ----> [Binary Noise z_t] ----/

```

### The Forward Diffusion Process (XOR Noise)

Instead of adding Gaussian noise, bits are flipped deterministically based on a mapper $\mathcal{M}_t$. Let $x_{0}(k)$ be the clean binary channel plane and $z_{t}(k)\in\{0,1\}^{h\times w}$ be the random binary noise plane generated at timestep $t$. The noisy tensor $x_{t}(k)$ is derived via:

$$x_{t}(k)=x_{0}(k)\oplus z_{t}(k)$$

where $\oplus$ represents the bitwise XOR operation. The noise level determines the fraction of flipped bits in $z_{t}(k)$, governed by a flip probability range of $[0, 0.5]$.

### Noise Scheduling

To regulate bit-flip probabilities across total diffusion timesteps $T = 1000$, a **quadratic noise schedule** $\beta_{t}$ is applied:

$$\beta_{t}=\left(\sqrt{\beta_{\text{start}}}+\frac{t}{T}\left(\sqrt{\beta_{\text{end}}}-\sqrt{\beta_{\text{start}}}\right)\right)^{2}$$

* $\beta_{\text{start}} = 10^{-5}$ (default minimum noise)
* $\beta_{\text{end}} = 0.5$ (default maximum noise)

### Objective Function (Optimization)

The denoiser network $g_{\theta}^{B}(x_{t},t,y_{e})$ is configured to output twice the number of input channels. The first half of the output channels predicts the clean tensor $\hat{x}_{0}$, and the second half predicts the added noise tensor $\hat{z}_{t}$. The model is optimized using Binary Cross-Entropy (BCE) loss across a batch of $M$ samples:

$$\mathcal{L}(\theta)=\frac{1}{M}\sum_{m=1}^{M}\left[\mathcal{L}_{x}\left(\hat{x}_{0}^{(m)},x_{0}^{(m)}\right)+\mathcal{L}_{z}\left(\hat{z}_{t}^{(m)},z_{t}^{(m)}\right)\right]$$

where $\mathcal{L}_{x}$ and $\mathcal{L}_{z}$ are the standard BCE losses computed per bit-plane $k$ and pixel coordinates $(i,j)$.

### Loss Weighting Strategies

* **For MBPR (Image-to-Image Tasks):** A linear bit-plane weighting balance is enforced for $\mathcal{L}_{x}$ to prioritize structural macro-fidelity. The MSB ($k=0$) weight is set to $1$, the LSB ($k=n-1$) weight is set to $0.1$, and intermediate planes are linearly interpolated between $1$ and $0.1$. Noise prediction weights ($\mathcal{L}_{z}$) stay constant at $1$ across all planes.
* **For LBR (Generation Tasks):** Constant loss weights ($1.0$) are utilized across all latent channels.

---

## Experimental Setup and Implementation Details

### Model Architectures

1. **Image-to-Image Denoiser (35.8M Parameters):**
* **Base:** A lightweight U-Net architecture containing four convolutional downsampling blocks.


* **Attention Mechanisms:** Self-attention layers are placed exclusively within the deepest bottleneck block. Linear attention layers are utilized in the remaining three blocks to control compute overhead.
* **Conditioning:** Timestep $t$ is applied via sinusoidal embeddings acting as additive biases in every block. Task images ($I_y$) are transformed via $\mathcal{T}$ and directly concatenated as additional input channels alongside $x_t$.




2. **Class-Conditional Denoiser (DiT):**
* Uses a Diffusion Transformer structure conditioned on one-hot class labels via embedding layers that inject additive biases into every block.


* **DiT-S Variant:** 32.9M parameters.


* **DiT-B Variant:** 130M parameters.





### Detailed Hyperparameter Matrix

| Hyperparameter | Image-to-Image Translation (U-Net) | Class-Conditional (DiT-S / DiT-B) |
| --- | --- | --- |
| **Optimizer** | AdamW | AdamW |
| **Learning Rate** | $1 \times 10^{-4}$ | $4 \times 10^{-4}$ |
| **Weight Decay** | $1 \times 10^{-6}$ | $1 \times 10^{-2}$ (Excludes biases & pos-embeddings) |
| **Training Steps** | 500,000 | 1,000,000 (Cosine schedule + 50k warmup steps) |
| **EMA Update Freq / Decay** | Every 10 steps / $\alpha_{\text{ema}} = 0.995$ | Every 10 steps / $\alpha_{\text{ema}} = 0.995$ |
| **Total Diffusion Steps ($T$)** | 1,000 steps | 1,000 steps |
| **Total Batch Size** | 64 (for $256^2$) | 32 (for $$512^$) | 2,048 |
| **Precision Type** | bfloat16 + FlashAttention | bfloat16 + FlashAttention |
| **Inference Guidance Scale** | N/A | DiT-S: 11.25 | DiT-B: 8.75 (Classifier-Free Guidance) |

---

## Reproducible Task Configurations

### 1. Super-Resolution (4x Scaling)

* **Input Setup:** Low-resolution inputs ($64 \times 64$) are upscale-stretched back to $256 \times 256$ via baseline bilinear interpolation.
* **Conditioning Method:** This bilinearly upsampled image is split into its binary bit-planes via $\mathcal{T}$ and directly concatenated to $x_t$.
* **Augmentations:** Random cropping ($80\%\text{--}100\%$ scale) and horizontal flipping.
* **Sampling Step Target:** Exactly 30 steps.

### 2. Image Inpainting

* **Input Setup:** Destroys $10\%\text{--}30\%$ of pixel areas randomly using standard structural masks.
* **Conditioning Method:** Masked region pixels are filled entirely with random binary noise $\{0,1\}$. The corrupted image $I_m$ is mapped to bit-planes and combined alongside the explicit binary mask channel $M$, forming the total condition variable $I_y = [M, I_m]$.
* **Augmentations:** Random cropping ($80\%\text{--}100\%$) and horizontal flipping.
* **Sampling Step Target:** Exactly 100 steps.

### 3. Blind Image Restoration

* **Input Setup:** Pretrained strictly on a synthetic degradation pipeline generated from FFHQ images. Random combinations of perturbations are layered according to the following strict operational rules:
* *Gaussian Blur:* $21 \times 21$ kernel size; isotropic or anisotropic; $\sigma_y \in [0.1, 7]$; rotation angle $[-\pi, \pi]$ (**Probability: 100%**).
* *Downsampling:* Scale factor selection $\in [1, 4]$ (**Probability: 100%**).
* *Additive Gaussian Noise:* $\sigma \in [0, 15]/255$ (**Probability: 100%**).
* *JPEG Compression:* Quality factor interval $\in [50, 100]$ (**Probability: 100%**).
* *Color Shift:* Shift per channel $\in [-20/255, 20/255]$ (**Probability: 30%**).
* *Color Jitter:* Brightness $[0.5, 1.5]$, Contrast $[0.5, 1.5]$, Saturation $[0, 1.5]$, Hue $[-0.1, 0.1]$ (**Probability: 30%**).
* *Grayscale Conversion:* (**Probability: 1%**).


* **Conditioning Method:** Bit-planes of the degraded image are fed as additional input channels.
* **Sampling Step Target:** Exactly 40 steps.

### 4. Class-Conditional Image Generation (ImageNet-1k)

* **Input Setup:** Cached binary embeddings are precomputed directly through the MAGVIT-v2 encoder on raw ImageNet data without applying any data augmentations.
* **Sampling Step Target:** Exactly 7 steps using Classifier-Free Guidance.

---

## Complete Inference Algorithm

The generative step reverses the process by decoding predicted parameters directly through a sigmoid threshold function before applying bitwise operations:

### Algorithm 1: BDPM Inference Execution

```python
def get_bdpm_sample(conditioning_image, class_label=None, steps=[T, ..., 0], threshold=0.5):
    # 1. Initialize space with uniform random binary values 
    x_t = generate_random_binary_tensor() 
    
    # 2. Extract context embeddings
    if is_image_to_image_task:
        y_e = encoder_y(conditioning_image) # Concatenated bit-planes
    else:
        y_e = embedding_layer(class_label)   # Class conditioning bias
        
    # 3. Iterative Reverse Sampling Loop
    for t in steps:
        # Predict both clean image planes and noise components simultaneously
        predicted_x0_logits, predicted_zt_logits = denoiser_network(x_t, t, y_e)
        
        # Mapper Q: Apply sigmoid activation and map to strict binary values via threshold
        x0_hat = (sigmoid(predicted_x0_logits) >= threshold).astype(binary)
        
        # Generate random binary noise plane appropriate for step t
        z_t = get_binary_noise(t)
        
        # XOR bit-flip update step to compute the next latent state
        x_t = x0_hat ^ z_t
        
    # 4. Invert binary space back to RGB image array
    return inverse_transform_T(x_t)

```