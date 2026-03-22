# ⚡ Ternary-VLA: 1.58-Bit Vision-Language-Action Engine for Bare-Metal Edge Silicon

> **Running a 2.3-Billion parameter reasoning model and a 150+ FPS continuous vision tracker entirely on edge mobile silicon—without a GPU.**

[cite_start]Standard AI architectures are hitting a physical memory and thermal wall on edge devices[cite: 7, 130]. [cite_start]Standard 16-bit floating-point (FP16) matrix multiplications cook batteries and bottleneck memory bandwidth[cite: 131]. 

[cite_start]This project completely bypasses the floating-point tax[cite: 137]. [cite_start]By compressing a massive neural architecture into a **1.58-bit ternary state**, every weight in this model is strictly bound to just three values: **{-1, 0, 1}**[cite: 10, 156]. 

The result? [cite_start]Matrix multiplications are replaced by highly parallelized integer addition and subtraction [cite: 163][cite_start], allowing GPT-level logical reasoning and high-speed spatial perception to run natively on Apple M-Series chips with negligible thermal output[cite: 11, 100].

---

## 🚀 The 1.58-Bit Paradigm (Performance Benchmarks)

This repository unifies a high-speed perception engine with a MatMul-free reasoning brain, benchmarked directly on an Apple Silicon M2 processor:

* [cite_start]**Weight Precision:** Strictly Ternary (-1, 0, 1)[cite: 10, 156].
* [cite_start]**Massive Memory Compression:** * **Vision Engine:** Memory footprint reduced by >85%, shrinking the entire parametric model to **under 800 KB**[cite: 10, 23]. It executes entirely within the 16 MB L2 Cache, eliminating main RAM fetch latency[cite: 22, 23].
    * [cite_start]**Reasoning Brain (VLA):** A 2.3-Billion parameter network compressed into a strict **600 MB** memory footprint[cite: 132, 133].
* [cite_start]**Extreme Speed:** Achieves a sustained **150+ FPS** for continuous spatial tracking with <6ms latency per frame[cite: 11, 80].
* [cite_start]**Battery & Thermal Efficiency:** Eliminating floating-point multipliers and transcendental functions reduces system-wide energy consumption by approximately **70%**[cite: 223].

---

## 🧠 System Architecture

[cite_start]Running a full 2-Billion parameter network for every frame of a 60 FPS video feed is physically impossible on mobile constraints[cite: 140]. [cite_start]This system solves this using a **Multi-Exit Architecture**[cite: 133, 141].

### 1. The Reflex Exit: Bare-Metal Vision Engine (Layer 4)
[cite_start]For ultra-fast, 60 FPS object detection and spatial segmentation, the system utilizes a lightweight detection head[cite: 145, 148].
* [cite_start]**Hardware Intrinsics:** Bypasses standard ONNX FP32 cast behavior[cite: 70]. [cite_start]Re-written in C++ using ARM NEON hardware intrinsics (`<arm_neon.h>`), computing 16 parallel multiplications in a single clock cycle using the `vdotq_s32` instruction[cite: 76, 78, 80].
* **Temporal Decoupling:** Injects an Extended Kalman Filter (EKF) post-inference. [cite_start]This decouples spatial detection from temporal physics, computing predictive trajectory paths in under 1 millisecond[cite: 11, 84, 116].

### 2. The Reasoning Exit: 2.3B MatMul-Free VLA (Layer 24)
[cite_start]If the prompt requires deep logical complexity, the signal bypasses the early exit and routes to the 24-layer reasoning backbone[cite: 149, 151].
* **MatMul-Free Linear Attention (MLGRU):** Eliminates the quadratic time and memory complexity of standard Self-Attention. [cite_start]Processes sequences linearly using element-wise Hadamard products, directly mapping to hardware accumulators[cite: 173, 174, 175, 187].
* [cite_start]**Integer-Only Normalization (L1-ShiftNorm):** Standard `RMSNorm` introduces severe latency due to square roots and floating-point division[cite: 193]. [cite_start]This architecture utilizes Manhattan length (L1 norm) and bitwise right-shifts (`x >> k`), executing normalization in a single clock cycle on ARM NEON integer units[cite: 198, 205, 207].
* [cite_start]**Base-2 Softmax:** Converts the standard $e^x$ exponential function to a Base-2 bit-shift and Look-Up Table (LUT) to avoid inefficient Arithmetic Logic Unit (ALU) processing[cite: 210, 212, 217].

---

## ⚙️ How to Run (Zero-Dependency Compilation)

This engine is built for bare-metal performance. It does not require bloated external libraries and relies strictly on native C++ and ARM NEON intrinsics.

*(Instructions for CMake build, Core MLX wrappers, and running the `powermetrics` scripts will be added here).*
