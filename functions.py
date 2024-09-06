def generate_vicsek_simulation(N, L, velocity, T, R, sigma):

    import numpy as np
    import pandas as pd
    from tqdm import tqdm

    # Helper function to calculate the square of the distance
    def distance_squared(x, y):
        return x**2 + y**2

    # Helper function to compute the angle given x and y coordinates
    def calculate_angle(x, y):
        return np.arctan2(y, x)

    # Convert polar coordinates (r, theta) to cartesian (x, y)
    def polar_to_cartesian(r, theta):
        x = np.array([r * np.cos(theta), r * np.sin(theta)])
        return x

    # Update angle based on neighbors' velocities within radius R
    def evolve_angle(R, t, bird_idx, X, N, sigma):
        X_relative = X[:, 0, t] - X[bird_idx, 0, t]
        Y_relative = X[:, 1, t] - X[bird_idx, 1, t]
        count = 0
        vx_avg = 0
        vy_avg = 0
        for i in range(N):
            distance = distance_squared(X_relative[i], Y_relative[i])
            if distance < R:
                count += 1
                vx_avg += X[i, 2, t]
                vy_avg += X[i, 3, t]
        vx_avg = vx_avg / count
        vy_avg = vy_avg / count
        new_theta = calculate_angle(vx_avg, vy_avg) + np.random.uniform(-sigma / 2, sigma / 2)
        return new_theta

    # Update angles for all birds at time step t
    def evolve_all_angles(R, t, X, N, sigma):
        angles = np.zeros(N)
        for bird_idx in range(N):
            angles[bird_idx] = evolve_angle(R, t, bird_idx, X, N, sigma)
        return angles

    # Main simulation using Euler integration
    def euler_integration(N, L, velocity, T, R, sigma):
        X = np.zeros((N, 4, T))  # Array to hold position and velocity for each bird over time
        S = np.zeros((N, 4))     # Initial positions and velocities

        # Initialize random positions and angles
        x0 = np.random.uniform(0, L, N)
        y0 = np.random.uniform(0, L, N)
        S[:, 0] = x0
        S[:, 1] = y0    

        initial_angles = np.random.uniform(0, 2 * np.pi, N)
        initial_velocities = polar_to_cartesian(velocity, initial_angles)
        S[:, 2], S[:, 3] = initial_velocities
        X[:, :, 0] = S
        
        # Run the simulation for T time steps
        for i in tqdm(range(1, T)):
            new_velocities = np.array([polar_to_cartesian(velocity, angle) 
                                       for angle in evolve_all_angles(R, i-1, X, N, sigma)])

            # Update velocities for each bird
            X[:, 2, i] = new_velocities[:, 0]
            X[:, 3, i] = new_velocities[:, 1]

            # Update positions using Euler method
            X[:, 0, i] = X[:, 0, i-1] + X[:, 2, i-1]
            X[:, 1, i] = X[:, 1, i-1] + X[:, 3, i-1]

            # Apply periodic boundary conditions
            for j in range(N):
                if X[j, 0, i] > L:
                    X[j, 0, i] -= L
                if X[j, 1, i] > L:
                    X[j, 1, i] -= L
                if X[j, 0, i] < 0:
                    X[j, 0, i] += L
                if X[j, 1, i] < 0:
                    X[j, 1, i] += L
            
        return X
    
    # Run the Euler integration for the simulation
    simulation_data = euler_integration(N, L, velocity, T, R, sigma)
    
    # Store results in a dictionary
    data = {}

    for i in range(N):
        data[f'Bird_{i},x'] = simulation_data[i, 0, :]
        data[f'Bird_{i},y'] = simulation_data[i, 1, :]
        data[f'Bird_{i},vx'] = simulation_data[i, 2, :]
        data[f'Bird_{i},vy'] = simulation_data[i, 3, :]

    # Convert the dictionary to a DataFrame and save it as a CSV
    data_df = pd.DataFrame(data)
    data_df.to_csv(f'simulationdata/vicseksimulation,N={N},L={L},v={velocity},T={T},R={R},sigma={sigma}')
    
    return data_df



def animate_trajectory(trajectory,L,v,R,sigma):

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.animation import PillowWriter

    N=int(trajectory.shape[1]/4)
    T=trajectory.shape[0]

    X=np.zeros([N,T])
    Y=np.zeros([N,T])

    for i in range(N):
        x=trajectory[f'Bird_{i},x']
        y=trajectory[f'Bird_{i},y']
        for j in range(T):
            X[i,j]=x[j]
            Y[i,j]=y[j]

    def update(frame):
        plt.cla()  # Clear the current plot
        ax.set_xlim(0, L)
        ax.set_ylim(0, L )

        # Plot the particle's trajectory up to the current frame
        for i in range(N):
            if frame<50:
                plt.scatter(X[i,:frame], Y[i,:frame],s=1.0)
        
                # Plot the current position of the particle
                plt.plot(X[i,frame], Y[i,frame],'o')
            else:
                plt.scatter(X[i,frame-50:frame], Y[i,frame-50:frame],s=1.0)
        
                # Plot the current position of the particle
                plt.plot(X[i,frame], Y[i,frame],'o')

        # Set plot properties
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Bird Trajectories')

        #ax.plot(x, y, 'ro')  # Plot the position as a red dot
        #ax.quiver(x, y, vx * arrow_scale, vy * arrow_scale, angles='xy', scale_units='xy', scale=1, color='blue')

    # Create animation
    fig, ax = plt.subplots()
    animation = FuncAnimation(fig, update, frames=len(range(T)), interval=1)
    writer = PillowWriter(fps=30)
    animation.save(f'simulationgifs/animation,N={N},L={L},v={v},T={T},R={R},sigma={sigma}.gif', writer=writer)
    #plt.show()
    #plt.show()