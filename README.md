# MetalNN

MetalNN is a neural network framework built in Metal Shading Language and Metal C++, designed for dynamic, flexible, and scalable model architectures. It allows rapid iteration, modular design, and is particularly optimized for macOS GPU hardware.

---

## Current Features ✅

- **Dynamic Network Configuration:** 
  - YAML-based model configurations allowing rapid iteration and adjustment of neural architectures.

- **Supported Layers:**
  - Dense Layer
  - Dropout Layer
  - Batch Normalization Layer
  
- **Interactive Commands via Keyboard (real-time training):**
  - Press `L` to run training (forward + backward pass).
  - Press `F` to run inference forward pass.
  - Press `S` to save parameters.
  - Press `O` to load parameters from a binary file.

- **Logging:**  
  Logs and debug outputs directly displayed within the app window.

<img width="639" alt="Screenshot 2025-03-12 at 5 03 59 AM" src="https://github.com/user-attachments/assets/b8e696fc-de86-457e-ae73-fb98c345df8b" />

- **Visualization:**  
  Training logs outputted to `multilayer-kernels.log` and `multilayer-kernels-debug.log` for external analysis and debugging.

---

## 🚧 **Planned Layers (TODO)**

Here's a prioritized, explicit list of layers currently planned for development, clearly ordered by dependency:

### Stage 1: Transformer Core Layers
- [x] **LayerNorm**: Critical normalization for transformer-based models.
- [x] **ResidualConnection**: Enables skip connections for stable transformer training.
- [ ] **PositionalEncodingLayer**: Adds positional context explicitly needed by transformers.
- [ ] **EmbeddingLayer**: Maps tokens into continuous embedding spaces.
- [ ] **FeedForward** (Explicit Dense-based feed-forward layers): Clearly broken down into:
  - [ ] Dense (Expansion layer with GELU)
  - [ ] Dropout
  - [ ] Dense (Projection layer)
- [ ] **MultiHeadAttentionLayer**: Implements attention mechanism for Transformers.
- [ ] **TransformerBlock**: Composite block encapsulating attention, FFN, residuals, and normalization.

### Stage 2: CNN Layers (Multi-modal support)
- [ ] **ConvolutionLayer (CNN)**: For image-based feature extraction.
- [ ] **Dropout**: For CNN regularization.
- [ ] **PoolingLayer**: For spatial downsampling.
- [ ] **FlattenLayer**: Bridges CNN to Dense layers.

### Stage 3: Multimodal & Advanced Layers
- [ ] **EmbeddingLayer** (multi-modal embeddings, e.g., text, images, audio).
- [ ] **CrossAttentionLayer** (Multimodal Fusion): Allows explicit interaction between modalities.

---

## Quickstart 🛠️

- Clone the repository.
- Modify `model-config.yml` for your chosen architecture.
- Run the project in Xcode, then use keyboard commands:
  - **Training:** Press `L`
  - **Inference:** Press `F`
  - **Save model:** Press `S`
  - **Load model:** Press `O`

### Visualizing Results:

Logs are generated for easy plotting and visualization in Octave/MATLAB:

- Training results: `multilayer_nn_training.m`

**Sample results (single frame):**

- Regression output:
<img width="579" alt="Screenshot 2025-03-02 at 3 50 40 PM" src="https://github.com/user-attachments/assets/8616c562-ceb4-4ea0-a454-7b8a6fd61904" />

- Classification output:
<img width="559" alt="Screenshot 2025-03-11 at 8 22 16 PM" src="https://github.com/user-attachments/assets/93455b84-bfca-40cf-834c-1be2d6e0a9a1" />

---

## Medium-term Roadmap 🎯

- [ ] Automated diagnostics and gradient checks.
- [ ] Validate new Transformer layers individually.
- [ ] Begin implementing distributed training across multiple machines using gRPC.
- [ ] Implement comprehensive unit tests and diagnostics for layer validation.

## Long-term Roadmap 🌟

- [ ] Build a complete multi-modal Transformer model (text, images, audio).
- [ ] Provide community-accessible distributed training (similar to SETI@home).
- [ ] Provide pre-trained Transformer models and deployment tooling for macOS/iOS devices.
- [ ] Integration with Swift for easier app deployment and usage.

---

## Contributing 👩‍💻👨‍💻
Contributions and ideas are always welcome! Open an issue or a pull request to discuss your enhancements or new features.

---

## About ✨

- **Author:** James Couch
- **Description:** MetalNN—A dynamic, scalable neural network framework built in Metal for macOS. Focused explicitly on Transformer-based architectures and modular distributed computing.

Enjoy exploring MetalNN, and happy training! 🚀

