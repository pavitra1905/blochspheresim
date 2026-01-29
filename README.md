# Bloch Sphere Simulator

A Python-based interactive visualization tool for understanding quantum computing through the Bloch sphere representation of single-qubit states and quantum gates.

## Overview

This project provides multiple ways to visualize and interact with quantum qubit states on the Bloch sphere:

- **Interactive slider control** — adjust quantum state parameters in real-time
- **Gate animations** — watch quantum gates transform qubit states with smooth 3D animations
- **Queue system** — chain multiple gates together for sequential application
- **Static visualization** — simple plots for quick state representation
- **GIF export** — save gate transformations as animated GIFs

## Files

### Core Modules

- **gates.py** — Quantum gate definitions and utilities
  - Single-qubit gates: X, Y, Z, Hadamard (H)
  - Gate application functions
  - Bloch coordinate conversion
  - SU(2) to SO(3) rotation matrix conversion

- **bloch_sphere.py** — Static Bloch sphere visualization
  - Displays initial and final states after a gate operation
  - Good for comparing quantum state transformations

- **bloch_interactive.py** — Interactive slider-based exploration
  - Control qubit state with θ (theta) and φ (phi) sliders
  - Real-time Bloch sphere visualization
  - Great for learning the relationship between parameters and state

- **bloch_final.py** — Full-featured interactive simulator
  - Multiple quantum gates (X, Y, Z, H)
  - Gate queueing system
  - Smooth gate animations
  - Real-time state information display (amplitudes, probabilities, Bloch coordinates)
  - Animation speed control
  - Revert button for undoing operations

- **bloch_anim.py** — Animation demonstration and GIF export
  - Demonstrates Hadamard gate transformation
  - Exports animation as bloch_rotation.gif
  - Shows rotation trail on Bloch sphere

## Requirements

```
numpy
matplotlib
```

Install with:
```bash
pip install numpy matplotlib
```

## Usage

### 1. Interactive Simulator (Recommended)

```bash
python bloch_final.py
```

**Controls:**
- **θ slider** — Polar angle (0 to π)
- **φ slider** — Azimuthal angle (0 to 2π)
- **Speed slider** — Animation speed (5–200 ms per frame)
- **X, Y, Z, H buttons** — Queue quantum gates
- **Play** — Execute queued gates sequentially
- **Clear** — Empty the gate queue
- **Revert** — Undo to the previous state

**Display Information:**
- Current quantum state |ψ⟩ (amplitudes α and β)
- Measurement probabilities P(0) and P(1)
- Bloch sphere coordinates (x, y, z)
- Queued gate operations

### 2. Simple Interactive Mode

```bash
python bloch_interactive.py
```

Adjust θ and φ sliders to explore different qubit states.

### 3. Static Visualization

```bash
python bloch_sphere.py
```

Shows a qubit state before and after applying a Hadamard gate.

### 4. Generate Animation

```bash
python bloch_anim.py
```

Creates bloch_rotation.gif showing a Hadamard gate transformation with rotation trail.

## Quantum Physics Background

### Bloch Sphere

The Bloch sphere is a geometric representation of single-qubit quantum states:
- **North pole** — |0⟩ state
- **South pole** — |1⟩ state
- **Equator** — Superposition states
- **Surface** — Pure quantum states

A qubit state is represented as:
$$|\psi\rangle = \cos\left(\frac{\theta}{2}\right)|0\rangle + e^{i\phi}\sin\left(\frac{\theta}{2}\right)|1\rangle$$

### Quantum Gates

The simulator includes four fundamental single-qubit gates:

- **X gate** — Bit flip (Pauli-X)
- **Y gate** — Pauli-Y rotation
- **Z gate** — Phase flip (Pauli-Z)
- **H gate** — Hadamard superposition

Each gate is represented as a 2×2 unitary matrix that transforms the quantum state.

## Features

- **3D Bloch sphere visualization** with wireframe
- **Real-time state updates** via sliders
- **Smooth gate animations** using rotation matrices
- **Physics-accurate transformations** (SU(2) → SO(3) mapping)
- **Gate queueing system** for sequential operations
- **Detailed state information panel**
- **Adjustable animation speed**
- **Undo/revert functionality**

## How It Works

1. **State Representation** — Qubit states are stored as 2-component complex vectors
2. **Gate Application** — Gates are applied via matrix multiplication
3. **Bloch Conversion** — State vectors are converted to 3D Bloch sphere coordinates
4. **Animation** — Rotations are interpolated smoothly using Rodrigues' rotation formula
5. **Visualization** — Matplotlib 3D axes display the sphere and state vectors

## Learning Outcomes

This simulator helps understand:
- How quantum states map to the Bloch sphere
- How quantum gates transform states geometrically
- The relationship between complex amplitudes and probabilities
- Quantum superposition and measurement
- Sequential gate operations and their effects

## License

Feel free to use and modify for educational purposes.

