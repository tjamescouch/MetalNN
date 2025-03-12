# MetalNN: Dynamic, GPU-Accelerated Neural Network Framework in Metal 🚀

This project implements a dynamic, configurable Neural Network framework. Currently supports simple dense layers for regression and classification accelerated by GPU computing via Apple's Metal Shading Language. RNN and other layer types are under construction. It's built to flexibly support complex neural architectures and streamline iterative development.

---

## Current Features ✅

- **Dynamic Network Configuration:** 
  - YAML-based model configuration allowing rapid iteration and easy adjustment of neural architectures.
  
- **Supported Layers:**
  - RNN layers (`tanh` activation).
  - Fully connected Dense layers (`linear`, `relu`, `softmax` activations).
  - Dropout layers for regularization.

- **Interactive Training & Inference:**
  - Press `L` to run training (learning) iterations.
  - Press `F` to run forward-pass inference.
  - Press `S` to save model parameters to a binary file.
  - Press `O` to load parameters from a binary file.

- **Logging & Visualization:**
  - Generates a MATLAB-compatible file (`multilayer_nn_training.m`) for easy result visualization in Octave/MATLAB.

---

## Quickstart 🛠️

- Clone the repository.
- Adjust the network architecture in `model-config.yaml`.
- Run the project in Xcode and control training/inference via keyboard commands:
  - **Training:** Press `L`
  - **Forward Pass:** Press `F`

Visualization of results is possible by executing the output MATLAB script in Octave or MATLAB.

- Example regression output:
<img width="579" alt="Screenshot 2025-03-02 at 3 50 40 PM" src="https://github.com/user-attachments/assets/8616c562-ceb4-4ea0-a454-7b8a6fd61904" />

- Example classification output:
<img width="559" alt="Screenshot 2025-03-11 at 8 22 16 PM" src="https://github.com/user-attachments/assets/93455b84-bfca-40cf-834c-1be2d6e0a9a1" />


---

## Roadmap 🗺️

### Short-term Goals:
- [x] Refine weight initialization and training parameters.
- [x] Resolve amplitude scaling issues.
- [x] Implement batch normalization layers to improve training stability.
- [ ] Add automated tests and diagnostics for gradient checks and debugging.
- [x] Implement dynamically selectable activation functions.
- [x] Implement dropout regularization.
- [x] Create sample model configuration with good results
- [x] Allow for saving and restoring parameters
- [ ] Fix convergence issues.
    - [x] single-dense-layer.yml
    - [x] multi-dense-layer.yml
    - [x] simple-ocr.yml
    - [x] ocr.yml
    - [x] advanced-ocr.yml
    - [ ] rnn.ym
- [ ] Validate layer implementations
    - [x] Dense Layer
    - [x] Dropout Layer
    - [ ] Batch Normalization Layer
    - [ ] RNN Layer
- [x] Implement classification using MNIST dataset

### Medium-term Goals:
- [ ] Expand support for additional layer types (e.g., GRU, LSTM, CNNs, Attention layers).
- [x] Implement ADAM optimizer algorithm.
- [ ] Implement distributed training capabilities, enabling local and distributed workloads (similar to SETI@Home model).
- [ ] Enhance GPU memory management and computation scheduling for better performance.

### Long-term Goals:
- [ ] Build a Transfomer based language model
- [ ] Develop a distributed training framework for community-driven computation sharing.
- [ ] Provide pre-trained models and tools for easy deployment on macOS/iOS devices.
- [ ] Integrate with Swift-based frameworks for easy end-user application development.

---

## Contributing 👩‍💻👨‍💻
Contributions and ideas are always welcome! Open an issue or a pull request to discuss your enhancements or new features.

---

## Author 🖊️

- **James Couch**
- Passionate about Machine Learning, GPU computing, and distributed systems.

Enjoy exploring MetalNN, and happy training! 🚀
