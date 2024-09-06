def generate_vicsek_simulation(N,L,v,T,R,sigma):

    import numpy as np
    import pandas as pd
    from tqdm import tqdm

    def r(x,y):
        return x**2+y**2

    def angulo(x,y):
        return np.arctan2(y,x)

    def polares(r,theta):
        x=np.array([r*np.cos(theta),r*np.sin(theta)])
        return x

    def anguloevo(R, t, p, X, N, sigma):
        Xrel = X[:,0,t]-X[p,0,t]
        Yrel= X[:,1,t]-X[p,1,t]
        j=0
        vxmed=0
        vymed=0
        for i in range(N):
            distancia = r(Xrel[i],Yrel[i])
            if distancia < R:
                j+=1
                vxmed += X[i,2,t]
                vymed += X[i,3,t]
        vxmed=vxmed/j
        vymed=vymed/j
        thetanovo = angulo(vxmed,vymed) + np.random.uniform(-sigma/2,sigma/2)
        return thetanovo

    def anguloevot(R, t, X, N, sigma):
        ang=np.zeros(N)
        for p in range(N):
            ang[p]= anguloevo(R, t, p, X, N, sigma)
        return ang


    def euler(N,L,v,T,R,sigma):
        X = np.zeros((N,4,T))
        S = np.zeros((N,4)) #posicoes e velocidades iniciais

        x0 = np.random.uniform(0, L, N) #posicoes iniciais
        y0 = np.random.uniform(0, L, N)
        S[:,0] = x0
        S[:,1] = y0    

        theta0=np.random.uniform(0,2*np.pi,N)
        v0=polares(v,theta0)
        S[:,2],S[:,3]=v0
        X[:,:,0]=S
        
        for i in tqdm(range(1,T)):
            vnovo = np.array([polares(v,x) for x in anguloevot(R, i-1, X, N, sigma)])
            #vnovo= polares(v, anguloevot(R, i-1, X, N, sigma))
            for z in range(1):
                X[:,2,i]=vnovo[:,0]
                X[:,3,i]=vnovo[:,1]
            #X[:,2,i]=[vnovo[x,0] for x in range(3)]
            #X[:,3,i]=vnovo[:,1]
            #X[:,2:3,i]=vnovo
            X[:,0,i]=X[:,0,i-1]+X[:,2,i-1]
            X[:,1,i]=X[:,1,i-1]+X[:,3,i-1]
            for j in range(N): #condicao de contorno periodica
                if X[j,0,i]>L:
                    X[j,0,i]-=L
                if X[j,1,i]>L:
                    X[j,1,i]-=L
                if X[j,0,i]<0:
                    X[j,0,i]+=L
                if X[j,1,i]<0:
                    X[j,1,i]+=L
            
            
        #print(X)
        return X
    
    u=euler(N,L,v,T,R,sigma)
    
    dados={}

    for i in range(N):
        dados[f'Bird_{i},x']=u[i,0,:]
        dados[f'Bird_{i},y']=u[i,1,:]
        dados[f'Bird_{i},vx']=u[i,2,:]
        dados[f'Bird_{i},vy']=u[i,3,:]

    dados=pd.DataFrame(dados)
    dados.to_csv(f'simulationdata/vicseksimulation,N={N},L={L},v={v},T={T},R={R},sigma={sigma}')
    return dados


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
        plt.title('Particle Trajectory')

        #ax.plot(x, y, 'ro')  # Plot the position as a red dot
        #ax.quiver(x, y, vx * arrow_scale, vy * arrow_scale, angles='xy', scale_units='xy', scale=1, color='blue')

    # Create animation
    fig, ax = plt.subplots()
    animation = FuncAnimation(fig, update, frames=len(range(T)), interval=1)
    writer = PillowWriter(fps=30)
    animation.save(f'simulationgifs/animation,N={N},L={L},v={v},T={T},R={R},sigma={sigma}.gif', writer=writer)
    #plt.show()
    #plt.show()