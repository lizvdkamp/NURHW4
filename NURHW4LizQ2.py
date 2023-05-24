import numpy as np
import matplotlib.pyplot as plt
import timeit

#Question 2

#2a

#Importing the Cloud-In-Cell script

np.random.seed(121)

n_mesh = 16
n_part = 1024
positions = np.random.uniform(low=0, high=n_mesh, size=(3, n_part))

grid = np.arange(n_mesh) + 0.5
densities = np.zeros(shape=(n_mesh, n_mesh, n_mesh))
cellvol = 1.

for p in range(n_part):
    cellind = np.zeros(shape=(3, 2))
    dist = np.zeros(shape=(3, 2))

    for i in range(3):
        cellind[i] = np.where((abs(positions[i, p] - grid) < 1) |
                              (abs(positions[i, p] - grid - 16) < 1) | 
                              (abs(positions[i, p] - grid + 16) < 1))[0]
        dist[i] = abs(positions[i, p] - grid[cellind[i].astype(int)])

    cellind = cellind.astype(int)

    for (x, dx) in zip(cellind[0], dist[0]):    
        for (y, dy) in zip(cellind[1], dist[1]):
            for (z, dz) in zip(cellind[2], dist[2]):
                if dx > 15: dx = abs(dx - 16)
                if dy > 15: dy = abs(dy - 16)
                if dz > 15: dz = abs(dz - 16)

                densities[x, y, z] += (1 - dx)*(1 - dy)*(1 - dz) / cellvol

mean_dens = n_part/n_mesh**3
dens_contrast = (densities - mean_dens)/mean_dens

print(grid)

for i in range(4):
	indxs = [4,9,11,14]
	zs = ["4.5", "9.5", "11.5", "14.5"]
	plt.imshow(dens_contrast[:,:,indxs[i]], origin="lower")
	plt.colorbar(label="density contrast")
	plt.title("z = "+zs[i])
	plt.xlabel("x")
	plt.ylabel("y")
	plt.savefig(f'2Dslicez{indxs[i]}.png')
	plt.close()

#2b

#Importing non-recursive FFT and inverse FFT

def FFT(x):
    """Takes an array x and calculates the FFT of this array by making use of the non-recursive algorithm for the FFT"""
    FT = x.copy()
    N = len(FT)
    
    inds = np.zeros(N,dtype=int)
    half = 2
    #Swapping indices by bitreversing
    while half <= N:
        #Save the previous half
        halfprev = int(half/2)
        #Rewrite the indices by setting the next set of 2*iterations equal to the non-zero part of the array + N/(2*iterations)
        inds[halfprev:half] = inds[0:half-halfprev]+N/half
        #print(i+halfprev,i+half,i,i+half-halfprev)
        #next step
        half *= 2
        #print(i,half,inds)
        
    #print(inds)
    FTnew = np.array(FT[inds], dtype=complex)
    #print(FTnew)
    
    Nj = 2
    #First loop
    while Nj <= N:
        for i in range(0,N,Nj):
            #Third loop
            for k in range(0,int(Nj/2)):
                m = i + k
                #print("Nj, i, k, m, m+Nj/2", Nj,i,k,m,int(m+Nj/2))
                t = FTnew[m].copy()
                exponent = np.exp(2j*np.pi*k/Nj)
                #print("t, exp, x_m+Nj/2", t,exponent, FTnew[int(m+Nj/2)])
                FTnew[m] = t + exponent*FTnew[int(m+Nj/2)]
                FTnew[int(m+Nj/2)] = t - exponent*FTnew[int(m+Nj/2)]
                #print("x_m, x_m+Nj/2", FTnew[m], FTnew[int(m+Nj/2)])
                #print("New array", FTnew)
                    
        Nj *= 2
                
    #print(FTnew)
    return FTnew

def invFFT(x):
    """Takes an array x and calculates the inverse FFT of this array by making use of a slightly adjusted version of the non-recursive algorithm for the FFT"""
    FT = x.copy()
    N = len(FT)
    
    inds = np.zeros(N,dtype=int)
    half = 2
    #Swapping indices by bitreversing
    while half <= N:
        #Save the previous half
        halfprev = int(half/2)
        #Rewrite the indices by setting the next set of 2*iterations equal to the non-zero part of the array + N/(2*iterations)
        inds[halfprev:half] = inds[0:half-halfprev]+N/half
        #print(i+halfprev,i+half,i,i+half-halfprev)
        #next step
        half *= 2
        #print(i,half,inds)
        
    #print(inds)
    FTnew = np.array(FT[inds], dtype=complex)
    #print(FTnew)
    
    Nj = 2
    #First loop
    while Nj <= N:
        for i in range(0,N,Nj):
            #Third loop
            for k in range(0,int(Nj/2)):
                m = i + k
                #print("Nj, k, m, m+Nj/2", Nj,k,m,int(k+Nj/2))
                t = FTnew[m].copy()
		#Now taking the exponent with the negative i
                exponent = np.exp(-2j*np.pi*k/Nj)
                #print("t, exp, x_m+Nj/2", t,exponent, FTnew[int(m+Nj/2)])
                FTnew[m] = (t + exponent*FTnew[int(m+Nj/2)])
                FTnew[int(m+Nj/2)] = (t - exponent*FTnew[int(m+Nj/2)])
                #print("x_m, x_m+Nj/2", FTnew[m], FTnew[int(k+Nj/2)])
                    
        Nj *= 2
                
    #Returning 1/N * the FT to get the right inverse FFT
    return 1/N * FTnew


#3D FFT
densFFT = np.array(dens_contrast, dtype=complex)
#First dimension
for i in range(dens_contrast[:,0,0].size):
	densFFT[i,:,:] = FFT(densFFT[i,:,:])
#Second dimension
for i in range(dens_contrast[0,:,0].size):
	densFFT[:,i,:] = FFT(densFFT[:,i,:])
#Third dimension
for i in range(dens_contrast[0,0,:].size):
	densFFT[:,:,i] = FFT(densFFT[:,:,i])

#Now we have k**2 * FFT(potential), so we have to divide by k**2
#Dividing by k is the same as dividing by the grid points
potFFT = densFFT / grid**2


#3D inverse FFT
potential = np.array(potFFT, dtype=complex)
for i in range(potFFT[:,0,0].size):
	potential[i,:,:] = invFFT(potential[i,:,:])
for i in range(potFFT[0,:,0].size):
	potential[:,i,:] = invFFT(potential[:,i,:])
for i in range(potFFT[0,0,:].size):
	potential[:,:,i] = invFFT(potential[:,:,i])


#Now taking only the real part, because the potential (and inverse FFT) should be real, but the FFT routines can leave small imaginary parts (of order 10^(-16))
potential = np.real(potential)

#Plotting

for i in range(4):
	indxs = [4,9,11,14]
	zs = ["4.5", "9.5", "11.5", "14.5"]
	plt.imshow(potential[:,:,indxs[i]], origin="lower")
	plt.colorbar(label="potential")
	plt.title("z = "+zs[i])
	plt.xlabel("x")
	plt.ylabel("y")
	plt.savefig(f'Potentialslice{indxs[i]}.png')
	plt.close()

for i in range(4):
	indxs = [4,9,11,14]
	zs = ["4.5", "9.5", "11.5", "14.5"]
	plt.imshow(np.log10(np.abs(potFFT[:,:,indxs[i]])), origin="lower")
	plt.colorbar(label=r"log$_{10}$(|$\~\Phi$|)")
	plt.title("z = "+zs[i])
	plt.xlabel("x")
	plt.ylabel("y")
	plt.savefig(f'Potentiallog{indxs[i]}.png')
	plt.close()






