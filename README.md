# MetalNN: Dynamic, GPU-Accelerated RNN using BPTT in Metal 🚀

This project implements a dynamic, configurable Recurrent Neural Network (RNN) using **Backpropagation Through Time (BPTT)**, accelerated by GPU computing via Apple's Metal Shading Language. It's built to flexibly support complex neural architectures and streamline iterative development.

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

---

## Roadmap 🗺️

### Short-term Goals:
- [ ] Refine weight initialization and training parameters to resolve amplitude scaling issues.
- [ ] Implement batch normalization layers to improve training stability.
- [ ] Add automated tests and diagnostics for gradient checks and debugging.
- [x] Implement dynamically selectable activation functions.

### Medium-term Goals:
- [ ] Expand support for additional layer types (e.g., GRU, LSTM, Attention layers).
- [ ] Impement ADAM optimizer algorithm
- [ ] Implement distributed training capabilities, enabling local and distributed workloads (similar to SETI@Home model).
- [ ] Enhance GPU memory management and computation scheduling for better performance.

### Long-term Goals:
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
