
import numpy as np
import time 
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

#%%
path = os.getcwd()
start_time = time.time()

dt = 1.0e-2
M = 1.0
coef = 0.5
ntimestep = 20000
nprint = 50
numsim = 200
Nx = 256
Ny = 256
dx = 0.5
dy = 0.5
A = 1.0
c0 = 0.4
#%%

def initial_con (Nx, Ny, dx, dy, c0):
    con2d = np.zeros(shape = (Nx, Ny))
    noise = 0.02
    for i in range (Nx):
        for j in range (Ny):
            
            con2d [i, j] = c0 + (0.5-np.random.rand()) * noise
    return con2d

con2d = initial_con(Nx, Ny, dx, dy, c0) # the initial condition for considered domain


#plt.imshow(con2d)
#%%
kxx =  2*np.pi*np.fft.fftfreq(Nx, d=dx) #Directional derivative in the x-direction in Fourier space
kyy =  2*np.pi*np.fft.fftfreq(Ny, d=dy) #Directional derivative in the y-direction in Fourier space

kx = np.append(kxx, 0)
ky = np.append(kyy, 0)

k2= np.zeros((Nx, Ny)) #Laplacian in Fourier space

for i in range (Nx):
    for j in range (Ny):
        k2[i,j]=kx[i]*kx[i]+ky[j]*ky[j]
        
k4=k2*k2 #square of laplacian in Fourier space

#%%

f_val = [] #list for free energy changes over time
checkk = [] #list for gradian of cocentration over time
def plot(step, conc):
    plt.imsave('%d.png'%step, conc)
    
counter = 0    
output = os.path.join(path, 'output')
os.makedirs(output)
full = np.zeros (numsim, (ntimestep/nprint)+1, Nx, Ny)
ii=0
#%%
    
for i in tqdm(range(numsim)):
    con = initial_con(Nx, Ny, dx, dy, c0)
    jj=0
    # os.chdir(output)
    # folder = os.path.join(output, "%d"%counter)
    # os.makedirs(folder)
    # os.chdir(folder)
    counter += 1 

    for step in range(1, ntimestep+1):
        
        check = np.diff(con)
        check2 = np.sum(check)*.5*coef
        checkk.append(check2)
        
        
        f = (A*con**2)*(1-con)**2
        ff=np.sum(f)
        f_val.append(np.sum(f))
        
        dfdc= 2*A*(con-1)*con*(2*con-1) #derivative of free energy with respect to concentration
        
        dfdck = np.fft.fft2(dfdc)
        conk = np.fft.fft2(con)
        ss=(1.0+ dt*k4*M*coef)
        number = dt *k2*M*dfdck
        conk = (conk-number)/ss
        con = np.fft.ifft2(conk)
        con = con.real
        
        if step ==1 or step%nprint == 0 :
            full[ii,jj, :,:] = con
            jj+=1
            # plot(step, con)
            # vtk(Nx, Ny, dx, dy, step, con)
    ii+=1
#%%
end_time = time.time()
etime=-(start_time-end_time)
print ("\n compute time is :%1d seconds"%etime) 
full = np.float32(full)
np.save("Dataset260_2F.npy", full)
#%%    
free_energy = [x + y for x, y in zip(f_val, checkk)]
plt.plot(free_energy)
plt.xlabel("Times step")
plt.ylabel("Free Energy")        

    

