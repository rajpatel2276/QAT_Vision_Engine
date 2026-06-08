# Extreme Edge AI Compiler & QAT Engine

## Overview
This repository contains a Quantization-Aware Training (QAT) engine and compiler built from first principles using PyTorch. It bypasses high-level post-training quantization APIs to directly simulate 8-bit integer (INT8) hardware constraints during the training loop. This allows Convolutional Neural Networks to adapt to extreme precision loss, enabling high-accuracy deployment on edge microprocessors (sub-2W hardware envelopes) with restricted memory bandwidth.

## The Physical Constraint & Engineering Solution
Standard neural networks rely on 32-bit floating-point (FP32) arithmetic, causing fatal memory and bandwidth bottlenecks on bare-metal silicon. Shrinking a trained model to INT8 post-training causes non-linear accuracy degradation. 

This engine solves this by actively sabotaging the mathematical precision *during* the forward pass of the training loop, forcing the optimizer to learn to survive the constraints of physical hardware.

### 1. The Fake Quantization Node
Both layer activations and static weights are intercepted and mapped to a rigid 256-bin grid before matrix multiplication occurs.
* **Scale ($S$):** Dynamically calculates the discrete step size: $S = \frac{X_{max} - X_{min}}{q_{max} - q_{min}}$
* **Zero-Point ($Z$):** Aligns the integer grid to physical zero: $Z = q_{min} - \text{round}(\frac{X_{min}}{S})$
* **Sabotage & Restore:** Tensors are quantized to INT8 and immediately dequantized back to FP32, injecting precision-loss noise into the data stream.

### 2. Bypassing Gradient Death (The STE)
Rounding functions create a step-graph with a derivative of zero, which instantly kills backpropagation and freezes the network. This architecture implements a custom C++ level **Straight-Through Estimator (STE)** via `torch.autograd.Function`.
* **Forward Pass:** Executes `torch.round()` to enforce INT8 constraints.
* **Backward Pass:** Bypasses the non-differentiable rounding node, passing gradients directly to the optimizer, allowing the model to adapt mathematically to the quantization noise.

## Repository Architecture

\`\`\`text
qat_vision_engine/
├── qat_core/
│   ├── quantizer.py      # Core math: FakeQuantize limits and STE Autograd bypass
│   ├── layers.py         # Custom QATConv2d layers injecting the saboteur nodes
│   └── models.py         # The lightweight, edge-targeted QATNet architecture
├── train.py              # Production training loop on CIFAR-10 data
└── export_onnx.py        # Bare-metal compiler
\`\`\`

## Bare-Metal Compilation (ONNX)
The final stage of this pipeline strips away the PyTorch framework entirely. The `export_onnx.py` script traces the simulated INT8 mathematical operations and freezes them into a static Open Neural Network Exchange (`.onnx`) graph. 

This provides a zero-dependency deployment asset that can be executed directly via C++ on target embedded hardware.

## Execution
**1. Train the Engine:**
\`\`\`bash
python3 train.py
\`\`\`

**2. Compile to Bare-Metal:**
\`\`\`bash
python3 export_onnx.py
\`\`\`