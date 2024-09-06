# Simulating the Vicsek Model in 2D with Python

The **Vicsek model** is a mathematical model used to describe **collective motion**, such as the flocking behavior of birds. It was proposed in 1995 in this [article](https://arxiv.org/abs/cond-mat/0611743v1).

In this model, each particle (e.g., bird) moves with a constant velocity `v`, while the orientation of the velocity is the dynamic variable. At each time step, the direction of a particle's velocity is updated to be the mean orientation of its neighbors within a certain radius `R`, combined with some random noise.

In this simulation, the motion occurs within a 2D square of size `L`. The number of particles is `N`, and the noise is uniformly distributed between `0` and `2π` with strength `σ`. Initially, birds are randomly positioned with random orientations, and the motion evolves using the **Euler method**.

## 📦 Installation Instructions

To run the program, you need to have **Python 3.11.3** installed. Additionally, the project uses **Jupyter** for running the simulation.

To install the necessary dependencies, use the following command:

```bash
pip install numpy pandas matplotlib tqdm
```

## 🚀 Usage Instructions

This project generates:

- Simulated trajectories saved as CSV files in the `simulationdata/` folder.
- Animations in GIF format saved in the `simulationgifs/` folder.

To simulate the movement:

1. Open the `example.ipynb` file.
2. Run the notebook cell by cell.
3. Follow the instructions and modify the parameters directly in the notebook.

If you want to generate an animation from a pre-saved movement file:

1. Open the `example_from_file.ipynb` notebook.
2. Follow the instructions there.

The main functions responsible for generating the trajectories and animations can be found in the `functions.py` file.

## 🤝 Contributions and Contact

If you'd like to contribute to this project, feel free to contact me at:  
📧 **sidney1395271@gmail.com**

## 🔮 Future Work

This simulation can be further optimized, especially for large numbers of birds. It may benefit from **GPU parallelization** for better performance. Additionally, the animation quality can be enhanced to create more visually appealing results.

