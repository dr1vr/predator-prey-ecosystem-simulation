# Predator-Prey Ecosystem Simulation - Alpha Version

A simplified 2D simulation of a predator-prey ecosystem with basic genetics, AI-based decision making, and environmental factors.

## Features

- Real-time visualization with day/night cycle
- Simple physics-based movement and interactions
- Plants that grow and reproduce
- Herbivores that eat plants
- Carnivores that hunt herbivores
- Population statistics and graphing
- Time control (pause/resume, speed up/slow down)

## Requirements

- Python 3.6+
- Pygame
- NumPy

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Running the Simulation

Run the simulation directly with Python:

```bash
python simulation.py
```

## Controls

- **Space**: Pause/Resume simulation
- **Up Arrow**: Increase simulation speed
- **Down Arrow**: Decrease simulation speed
- **G**: Toggle population graph
- **ESC**: Quit the simulation

## Understanding the Simulation

This alpha version demonstrates a simplified ecosystem with three main types of entities:

1. **Plants** (green): Grow over time and can reproduce. They are the foundation of the food chain.
2. **Herbivores** (blue): Eat plants and reproduce when they have enough energy.
3. **Carnivores** (red): Hunt herbivores and reproduce when they have enough energy.

The simulation includes a day/night cycle that affects visibility. Water bodies are rendered as blue areas.

## Notes

This is an alpha version with simplified mechanics to demonstrate the concept. The full version will include more complex genetics, advanced AI behavior, and additional environmental factors. 