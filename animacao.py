import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Assuming you have arrays X and Y for the object's trajectory
X = np.array([1, 2, 3, 4, 5])
Y = np.array([2, 4, 1, 3, 5])

fig, ax = plt.subplots()
line, = ax.plot([], [], 'bo-')  # Blue line with circles for each point
ax.set_xlim(min(X) - 1, max(X) + 1)
ax.set_ylim(min(Y) - 1, max(Y) + 1)

def update(frame):
    line.set_data(X[:frame + 1], Y[:frame + 1])
    return line,

# Create animation
animation = FuncAnimation(fig, update, frames=len(X), interval=1000, repeat=False)

plt.show()
