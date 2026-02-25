import torch
import matplotlib.pyplot as plt
import numpy as np
import time

if torch.cuda.is_available():
    device= "cuda"
elif torch.backends.mps.is_available():
    device= "mps"
else:
    device= "cpu" 
#device= "cpu"
print(f"Using device: {device}")
d=2
N= 10
z_c = 50
h_shape = (d,)+(N + 1,) * d 
print(h_shape)
#h = torch.randint(10, 100, h_shape)  # integers in [0, 10)
h= torch.zeros(h_shape, device=device)
z_shape = (N,) * d
z = torch.zeros(z_shape, device=device) 
def compute_slopes(h): 
    diff_shape = (d,)+(N,) * d
    diff = torch.zeros(diff_shape, device=device) 
    for i in range(d): 
        shape =[slice(1, N+1)] * d
        shape[i]= slice(0, N)
        h_a = h[i][shape]
        h_b = h[i][(slice(1, N+1),) * d]
        diff[i]= h_b-h_a

    return diff.sum(dim=0)

def update_heights(z,h): 
    """
    relaxation algorithm with open boundary conditions
    """
    Z= (z>z_c).to(torch.int64)
    for i in range(d): 
        shape =[slice(1, N+1)] * d
        h[i][shape]= h[i][shape] - Z
        shape[i]= slice(0, N)
        h[i][shape]= h[i][shape] + Z 
        
    return h

def relax(z,h): 
    """
    relaxation algorithm with open boundary conditions
    """
    center_mask = (z > z_c).to(torch.int32)
    shape =[slice(0, N)] * d
    adjacent_mask= torch.empty((N,) * d ,device=device)
    for i in range(d): 
        shape_a = [slice(0, N)] * d
        shape_b = [slice(0, N)] * d
        shape_a[i]= slice(1, N)
        shape_b[i]= slice(0, N-1)
        adjacent_mask[shape_a] += center_mask[shape_b]
        adjacent_mask[shape_b] += center_mask[shape_a]
    return z- 2 * d * center_mask + adjacent_mask
means=[]
T= 10000
t1= time.clock_gettime(0)


for t in range(T): 
    z= relax(z,h)#compute_slopes(h)
    #h= update_heights(z,h)
    #a= t%2
    #print(z)
    b= np.random.randint(0,N)
    c= np.random.randint(0,N)
    #[a][b][c]+=1
    #print(z)
    
    means.append(z.mean().cpu())
    z[b][c]+=1
t2= time.clock_gettime(0)

print(f"Took {t2-t1}s on device: {device}")
def plot(h,z): 
    fig, axes = plt.subplots(nrows=2, ncols=2,figsize=(12, 6), constrained_layout=True)  
    ax = axes.ravel()
    A = z.detach().cpu().numpy()   
    im = ax[0].imshow(A, origin='lower', aspect='auto')
    fi = ax[0].figure                     
    fi.colorbar(im, ax=ax[0]) 

    ax[1].plot(range(T),means)


    A = h[0].detach().cpu().numpy()   
    im = ax[2].imshow(A, origin='lower', aspect='auto')
    fi = ax[2].figure                     
    fi.colorbar(im, ax=ax[2]) 

    A = h[1].detach().cpu().numpy()   
    im = ax[3].imshow(A, origin='lower', aspect='auto')
    fi = ax[3].figure                     
    fi.colorbar(im, ax=ax[3]) 

    plt.show()



plot(h,z)