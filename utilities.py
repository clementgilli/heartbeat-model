import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def F1(x, y, epsilon):
    """fonction second membre 1"""
    if epsilon <= 0:
        raise ValueError("epsilon < 0")
    return epsilon*(y - (x*x*x)/3 + x)

def F2(x, y):
    """fonction second membre 2"""
    return -x

def F(X, t, epsilon):
    """fonction second membre pour le système"""
    x,y = X
    return np.array([F1(x,y,epsilon),F2(x,y)])


def trace_courbes(x0,y0,epsilon,tmax=20):
        
    T, N = tmax, 1000
    
    t = np.linspace(-T,T,N+1)
    
    x, y = np.meshgrid(
        np.linspace(-4, 4, 20),
        np.linspace(-4, 4, 20)
    )
    fx, fy = F1(x, y, epsilon), F2(x, y)
    n_sndmb = np.sqrt(fx**2 + fy**2)
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(2, 2, 1)
    ax.quiver(x, y, fx/n_sndmb, fy/n_sndmb, angles='xy')
    ax.axis('equal')
    ax.set_title("Champ de vecteur")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    #ax.set_xlim(-5,5)
    #ax.set_ylim(-5,5)
    ax.axhline(0,color="black")
    ax.axvline(0,color="black")

    X = odeint(F,[x0,y0],t,args=(epsilon,))

    # Extraction des X des solutions approchées [H0,H1,...HN] et [P0,P1,...,PN]
    H =  X[:,0] 
    P =  X[:,1]
    ax.plot(H, P, linewidth=2,linestyle="--",color="red")
    ax.scatter(x0,y0,color="red");
    
    
    t2 = np.linspace(0,tmax,N+1) 
    # Extraction des X des solutions approchées [H0,H1,...HN] et [P0,P1,...,PN]
    bx = fig.add_subplot(2, 2, 2)
    bx.set_title("Intensité du courant")
    bx.set_xlabel("t")
    bx.set_ylabel("x")
    X = odeint(F,[x0,y0],t2,args=(epsilon,))
    H =  X[:,0] 
    P =  X[:,1]
    bx.plot(t2, H, linewidth=2)
    bx.scatter(0,x0)
    #bx.plot(t, P, linewidth=2)
    bx.axhline(0,color="black")
    
    
def trace_courbes2(xy0,epsilon,tmax=20):
        
    T, N = tmax, 1000
    
    t = np.linspace(-T,T,N+1)
    
    x, y = np.meshgrid(
        np.linspace(-4, 4, 20),
        np.linspace(-4, 4, 20)
    )
    fx, fy = F1(x, y, epsilon), F2(x, y)
    n_sndmb = np.sqrt(fx**2 + fy**2)
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(2, 2, 1)
    ax.quiver(x, y, fx/n_sndmb, fy/n_sndmb, angles='xy')
    ax.axis('equal')
    ax.set_title("Diagramme des phases")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    #ax.set_xlim(-5,5)
    #ax.set_ylim(-5,5)
    ax.axhline(0,color="black")
    ax.axvline(0,color="black")
    
    t2 = np.linspace(0,tmax,N+1) 
    # Extraction des X des solutions approchées [H0,H1,...HN] et [P0,P1,...,PN]
    bx = fig.add_subplot(2, 2, 2)
    bx.set_title("Intensité du courant")
    bx.set_xlabel("t")
    bx.set_ylabel("x(t)")
    bx.axhline(0,color="black")
    
    for xy in xy0:
        x0,y0 = xy[0], xy[1]
        X = odeint(F,[x0,y0],t,args=(epsilon,))

        # Extraction des X des solutions approchées [H0,H1,...HN] et [P0,P1,...,PN]
        H =  X[:,0] 
        P =  X[:,1]
        ax.plot(H, P, linewidth=2,linestyle="--",label=f"((x0,y0) = {x0},{y0})")
        ax.legend(loc='lower right')
        ax.scatter(x0,y0)
        
        X = odeint(F,[x0,y0],t2,args=(epsilon,))
        H =  X[:,0] 
        P =  X[:,1]
        bx.plot(t2, H, linewidth=2,label=f"((x0,y0) = {x0},{y0})")
        bx.legend(loc='lower right')
        bx.scatter(0,x0);
        
        
def periode(xy0, epsilon, n0=200, err=10e-3, N=1000, tmax=20):
    
    t2 = np.linspace(0,tmax,N+1) 
    x0,y0 = xy0[0], xy0[1]
    X = odeint(F,[x0,y0],t2,args=(epsilon,))
    H =  X[:,0]
    
    fig, ax = plt.subplots()
    ax.plot(t2, H)
    
    maxi = H[n0]
    posMaxi = -1
    c = 0
    
    i = n0 + 1
    while(i < N or abs(H[i]-maxi)>err):
        
        if H[i] > maxi:
            maxi = H[i]
            posMaxi = i
            c = 0
            c += 5
            i += 5
            
        elif np.abs(H[i] - maxi) < err:
            ax.scatter(tmax*posMaxi/N, maxi,color="red")
            ax.scatter(tmax*(posMaxi + c)/N, maxi,color="red")
            ax.plot([tmax*posMaxi/N, tmax*(posMaxi + c)/N], [maxi, maxi],color="red")
            return c*tmax/N
        
        else:
            c += 1
            
        i += 1
    