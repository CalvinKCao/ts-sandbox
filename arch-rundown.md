
here's the prompt i used to write all this code which should explain the overall architecture of the diffusiontsf model 

**Role:** You are a Senior Deep Learning Engineer specialized in Generative AI and Time-Series Forecasting.

**Objective:** Implement a supervised Time-Series Forecasting model using **2D Diffusion**. The model should map 1D time-series data into a 2D image space (similar to the ViTime paper), use a Diffusion process to generate the "future" 2D representation, and then decode it back to 1D. The goal is to preserve high-frequency geometric "textures" (jagged edges, "W/V" shapes) that are usually lost in MSE-based regression.

---

### Phase 1: Data Preprocessing & 2D Mapping

1. **Normalization:** Implement a standardizer that scales input windows using local mean and standard deviation.
    
2. **2D Encoding (The "Stripe" Method):** * Map numerical values to a grid of height H=128 and width W=L (sequence length).
    
    - For each time step t, calculate the pixel index: yt​=clip(σxt​−μ​⋅MSH​+2H​,0,H−1), where MS (Maximum Scale) is 3.5.
        
    - **Representation:** Create a binary image where only the pixel at (t,yt​) is 1.
        
3. Apply a **1D Vertical Gaussian blur** (kernel size: Height=31, Width=1) or a highly anisotropic 2D blur (e.g., 31x1). This must strictly blur **only along the value axis** to create a probability density without smearing the temporal geometric patterns (W-shapes, sharp edges) across time steps."

---

### Phase 2: Architecture - Conditional 2D U-Net

1. **Base:** Implement a 2D U-Net with Residual blocks and Cross-Attention.
    
2. **Conditioning:** * The model must be conditioned on the **Historical Context**.
    
    - Convert the _past_ sequence into its 2D blurred representation.
        
    - Pass the _past_ image as a feature map through a "Conditioning Encoder" (simple CNN) and concatenate or inject it into the U-Net via Cross-Attention at each downsampling/upsampling step.
        
3. **Diffusion Framework:** * Use the **DDPM (Denoising Diffusion Probabilistic Models)** framework.
    
    - Set T=1000 diffusion steps with a linear or cosine noise schedule.
        
    - Input to the U-Net: (Noisy Future Image + Time Embedding + Past Context Encoding).
        
    - Output: Predicted noise ϵθ​.
        

---

### Phase 3: Training and Inference

1. **Loss Function:** Use L2​ loss between the added noise and the predicted noise in the 2D space. Note: The loss is computed on the _pixels_, not the numerical values.
    
2. **Reverse Process (Sampling):** * Start with pure Gaussian noise of shape (H,Future_Length).
    
    - Iteratively denoise using the trained U-Net conditioned on the "Past Image."
        
3. **Decoding (2D to 1D):**
    
    - After T steps, you will have a 2D probability map.
        
    - **Method:** For each column (time step), calculate the **Expectation**: xt′​=∑i=0H−1​P(i)⋅Value(i), where P(i) is the normalized intensity of the pixel at height i. This ensures a differentiable and smooth recovery of the 1D value.
        

---

### Phase 4: Implementation Requirements

- **Framework:** PyTorch.
    
- **Library:** Use `diffusers` for the noise scheduler logic if possible, but customize the U-Net for non-square, time-series-shaped images.
    
- **Performance Metric:** In addition to MSE/MAE, implement a **Shape-Preservation Metric**: Compare the distribution of the first-order derivatives (gradients) of the prediction vs. the ground truth to ensure high-frequency textures are captured.
- Before you are finished, run a test backward and forward pass on a tiny toy dataset to ensure everything works smoothly.
- before starting, examine @models/ViTime-main/Yang et al. - 2025 - ViTime Foundation Model for Time Series Forecasting Powered by Vision Intelligence.txt and take some notes on the similar implementation. this is not the exact thing i am trying to reimplement but the whole image representation thing is inspired by it so it may give you a better idea of what im looking for. again, this prompt overrides anything that paper says though. after examining the paper, check in with me before starting - i.e. do you have any questions before starting?
