import torch
import matplotlib.pyplot as plt
import numpy as np
d=2
N= 10
z_c = 8
h_shape = (d,)+(N + 1,) * d 
print(h_shape)
#h = torch.randint(10, 100, h_shape)  # integers in [0, 10)
h= torch.zeros(h_shape)
z_shape = (N,) * d
z = torch.zeros(z_shape)  
def compute_slopes(h): 
    diff_shape = (d,)+(N,) * d
    diff = torch.zeros(diff_shape)  
    for i in range(d): 
        shape =[slice(1, N+1)] * d
        shape[i]= slice(0, N)
        h_a = h[i][shape]
        h_b = h[i][(slice(1, N+1),) * d]
        diff[i]= h_a-h_b

    return diff.sum(dim=0)

def update_heights(z,h): 
    Z= (z>z_c).to(torch.int64)
    for i in range(d): 
        shape =[slice(1, N+1)] * d
        h[i][shape]= h[i][shape] + Z
        shape[i]= slice(0, N)
        h[i][shape]= h[i][shape] - Z 
        
    return h


means=[]
T= 100001

for t in range(T): 
    z= compute_slopes(h)
    h= update_heights(z,h)
    a= np.random.randint(0,1)
    b= np.random.randint(3,7)
    c= np.random.randint(3,7)
    h[a][b][c]+=1
    #print(z)
    means.append(z.mean())



A = z.detach().cpu().numpy()   # T is your torch tensor (2D)

plt.figure()
plt.imshow(A, origin="lower", aspect="auto")
plt.colorbar()
plt.show()

plt.plot(range(T),means)
plt.show()