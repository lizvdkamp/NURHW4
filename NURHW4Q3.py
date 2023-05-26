import numpy as np
import matplotlib.pyplot as plt
import timeit

#Question 3

#3a

gal_data = np.loadtxt("https://home.strw.leidenuniv.nl/~belcheva/galaxy_data.txt")

classes = gal_data[:,4]
features = np.vstack((np.ones(classes.size), gal_data[:,0], gal_data[:,1], gal_data[:,2], gal_data[:,3]))
features = features.T
print(features.shape)
values = np.array(["bias",r"ordered rotation parameter $\kappa_{CO}$", "color", "measure of extendedness", "emission line flux"])

#standardization
for i in range(1,features[0,:].size):
	#Taking the mean and standard deviation of the features
	mean0 = np.mean(features[:,i])
	std0 = np.std(features[:,i])
	#Scaling
	features[:,i] = (features[:,i] - mean0)/std0

	#Plotting
	plt.hist(features[:,i], bins=20)
	plt.yscale("log")
	plt.title("Distribution of scaled feature "+str(i)+", "+values[i])
	plt.xlabel("scaled ~"+values[i])
	plt.ylabel("Number")
	plt.savefig(f'FeatureDistributionplot{i-1}.png')
	plt.close()

# Save a text file
np.savetxt('Featuresoutput.txt',np.transpose([features[:,0], features[:,1], features[:,2], features[:,3], features[:,4]]))

#3b

#Importing the cost function and logistic regression function from my tutorial

def costfunk(h,y=classes):
    """Calculates the cost function for logistic regression, with model outcome h and data y."""
    m = y.size
    return -1/m * np.sum(y*np.log(h)+(1-y)*np.log(1-h))

def sigmoid(theta,x):
	"""Calculates the sigmoid function for given model values theta and features x"""
	#z = theta.T*x
	z = np.sum(theta*x, axis=1)
	#sigma (z)
	h = 1/(1+np.exp(-z))
	return h

def goldsecsearch(func,a,b,c,acc,maxit):
    """Finds the minimum of a given function func, within [a,c], with b the first guess for the minimum. Terminates when either the maximum amount of iterations maxit is reached, or when the interval is small than our target accuracy."""

    #Defining the golden ratio
    phi = (1+np.sqrt(5))*0.5
    w = 2-phi
    
    #Iterate a maximum of maxit times
    for k in range(0,maxit):
	#Find the largest interval
        if np.abs(b-a) > np.abs(c-b):
            x = a
        else:
            x = c

	#Set the next bracket
        d = b + (x-b)*w

	#If target accuracy is reached, terminate
        if np.abs(c-a) < acc:
            print("Accuracy", np.abs(c-a))
            if func(d) < func(b):
                return d
            else:
                return b
	#Else, keep looping with the new bracket
        else:
            #print("Loop")
            if x == c:
                if func(d) < func(b):
                    a,b = b,d
                else:
                    c = d

            if x == a:
                if func(d) < func(b):
                    c,b = b,d
                else:
                    a = d

    #print("Max it. Done",d)
    return d
    
    

def logreg(x,theta,y, maxit=10**3, acc=10**(-6), H0=None, it=0, tht_pts=None, func=costfunk, sigmoid=sigmoid):
	"""Takes features x, model values theta and data y and calculates the theta that minimizes the function, func, by using a Quasi-Newton method, based on logistic regression. Will terminate after either the target accuracy is reached or the maximum number of iterations is reached. Returns both the best-fit theta, as well as all previous theta found by the routine."""

	#m = x[:,0].size
	#print("it", it)
	h0 = sigmoid(theta,x)

	def gradient(x,h,y):
		#the gradient of the logistic regression cost function
		m = x[:,0].size
		grad = 1/m * np.sum(x.T * (h-y), axis=1)
		#print(grad)
		return grad


	#Quasi Newton minimization
	if it == 0:
		Hessian_0 = np.identity(x[0,:].size)
	else:
		Hessian_0 = H0
	#Matrix vector multiplication to calculate the direction of stepping
	n_initial = -np.sum(Hessian_0 * gradient(x,h0,y), axis=1)

	#Create a function to minimize
	def minimum(lambd):
		tht_new = theta + lambd*n_initial
		h_new = sigmoid(tht_new,x)
		return func(h_new, y)

	#Find the best stepsize with the golden section search
	lambd_i = goldsecsearch(minimum,-20,0.5,20,10**(-20),50)

	delta = lambd_i *n_initial

	tht_new = theta + delta

	#print("it before tht_pts", it)
	#Save the steps taken in case we want to see the road to the minimum
	if it == 0:
		tht_pts = np.vstack((theta,tht_new)).copy() 
	else:
		tht_pts = np.vstack((tht_pts,tht_new)).copy()

	#Function values
	h0, h_new = h0, sigmoid(tht_new,x)
	costfunk0, costfunk_new = func(h0,y), func(h_new,y)
	gradient0, gradient_new = gradient(x,h0,y), gradient(x,h_new,y)

	#Check if we've reached the accuracy return
	if np.abs(costfunk_new - costfunk0) == 0 or np.abs(costfunk_new - costfunk0)/(0.5*np.abs(costfunk_new - costfunk0)) < acc:
		print("accuracy return, it = ", it)
		return tht_new, tht_pts

	#Calculate differences in gradient with new solutions
	Diff_i = gradient_new - gradient0

	#Check if the gradient is smaller than the target accuracy
	if np.abs(np.amax(gradient_new, axis=0)) < acc:
		print("gradient convergence", np.amax(gradient_new, axis=0))
		print("it = ", it)
		return tht_new, tht_pts


	HessianDiff = np.sum(Hessian_0*Diff_i, axis=1)

	u_mat = delta/np.sum(delta*Diff_i) - HessianDiff/np.sum(Diff_i * HessianDiff)

	Hessian_i = Hessian_0 + np.outer(delta, delta)/np.sum(delta*Diff_i) - np.outer(HessianDiff, HessianDiff)/np.sum(Diff_i * HessianDiff) + np.sum(Diff_i * HessianDiff)*np.outer(u_mat,u_mat)

	it+= 1
	
	if it >= maxit:
		print("max its reached")
		return tht_new, tht_pts
	else:
		return logreg(x,tht_new,y, maxit=maxit, acc=acc, it=it, tht_pts=tht_pts, H0=Hessian_i, func=func, sigmoid=sigmoid)

#first taking only 2 features (and the bias), I will take rotation and color, since I feel like those are quite unrelated (unlike for example color and star formation rate)
ftrs = features[:,:3]
theta_first = np.ones(3)
theta_sol, thetas = logreg(ftrs, theta_first, classes, maxit=300, acc=10**(-12), it=0)

#print(thetas)

costfunks = np.zeros(thetas[:,0].size)
for i in range(thetas[:,0].size):
	hs = sigmoid(thetas[i,:], ftrs)
	costfunks[i] = costfunk(hs,classes)

plt.plot(costfunks)
plt.title("Only the first two features")
plt.xlabel("Number of iterations")
plt.ylabel("Value of the cost function")
plt.savefig('First2Features.png')
plt.close()

#Taking the last 2 features (and the bias)
ftrs = np.vstack((features[:,0],features[:,3],features[:,4])).T
theta_first = np.ones(3)
theta_sol, thetas = logreg(ftrs, theta_first, classes, maxit=300, acc=10**(-12), it=0)

#print(thetas)

costfunks = np.zeros(thetas[:,0].size)
for i in range(thetas[:,0].size):
	hs = sigmoid(ftrs, thetas[i,:])
	costfunks[i] = costfunk(hs,classes)

plt.plot(costfunks)
plt.title("Only the last two features")
plt.xlabel("Number of iterations")
plt.ylabel("Value of the cost function")
plt.savefig('Last2Features.png')
plt.close()

#Now taking all 4 features
#print(features)
theta_first = np.ones(5)
theta_sol, thetas = logreg(features, theta_first, classes, maxit=300, acc=10**(-12), it=0)

#print(thetas)

costfunks = np.zeros(thetas[:,0].size)
for i in range(thetas[:,0].size):
	hs = sigmoid(thetas[i,:], features)
	costfunks[i] = costfunk(hs,classes)

plt.plot(costfunks)
plt.title("All features")
plt.xlabel("Number of iterations")
plt.ylabel("Value of the cost function")
plt.savefig('All2Features.png')
plt.close()

#Not much changes between the first and last one, but the value of the cost function for only the last two features is a lot higher. 
#This is because the last two values do not add much to the function (their theta is a lot smaller than of the first two features)


#3c

#Calculating the predicted classes by using the sigmoid function
h_sol = sigmoid(theta_sol, features)
models = np.zeros(h_sol.size)

#If the sigmoid >= 1/2, the class is 1 otherwhise it is zero
for i in range(h_sol.size):
    if h_sol[i] >= 0.5:
        models[i] = 1
        

#making a mask for the correct classifications
correct_classifications = models == classes

#making a mask for the correct and incorrect classifications
models_correct = models[correct_classifications]
models_incorrect = models[~correct_classifications]

#Making the confusion matrix
confusion = np.zeros((2,2))

#Assuming 1 = Positive and 0 = Negative
#True Negatives
confusion[0,0] = len(models_correct[models_correct==0])
#False Positives
confusion[0,1] = len(models_incorrect[models_incorrect==1])
#True Positives
confusion[1,1] = len(models_correct[models_correct==1])
#False Negatives
confusion[1,0] = len(models_incorrect[models_incorrect==0])

print(confusion)
precision = confusion[1,1]/len(models_correct)
recall = confusion[1,1]/(confusion[1,1]+confusion[0,1])

#Calculating F1 score
F1 = 2 * precision * recall / (precision + recall)

print(F1)

#Saving everything
names  = np.array(['True Negatives', 'False Positives', 'True Positives', 'False Negatives', 'F1 Score'])
testresults = np.transpose([confusion[0,0], confusion[0,1], confusion[1,1], confusion[1,0], F1])

testres = np.zeros(names.size, dtype=[('var1', 'U20'), ('var2', float)])
testres['var1'] = names
testres['var2'] = testresults

np.savetxt('Testsetoutput.txt',testres, fmt="%s %10.16f")

#Plotting
for k1 in range(1,5):
	for k2 in range(1,5):
		if k1 < k2:
			x_ftr = features[:,k1]
			y_ftr = features[:,k2]
			#print(np.amin(x_ftr),np.amin(y_ftr))
			plt.figure(figsize=(12,12))
			plt.scatter(x_ftr, y_ftr, marker='.')
			
			#Decision boundary is given by sum(theta_i)*feature_i = 0, so for two features, you have that feature_j = -(theta_i/theta_j)*feature_i -theta_0/theta_j
			plt.plot(x_ftr, -(theta_sol[k1]/theta_sol[k2])*x_ftr-theta_sol[0]/theta_sol[k2], color='k', label='decision boundary')
			plt.title("Two features against each other plus decision boundary (zoomed to [-3,3])")
			plt.xlabel("scaled ~"+values[k1])
			plt.ylabel("scaled ~"+values[k2])
			plt.xlim(-3, 3)
			plt.ylim(-3, 3)
			plt.legend()
			plt.savefig(f'DecisionBoundary{k1-1}{k2-1}.png')
			plt.close()

#For almost all of the features except the first two, the decision boundary seems a bit arbitrary, since the features seem correlated. This confirms my suspicion of the first two features mattering the most because they are the least correlated with each other.






