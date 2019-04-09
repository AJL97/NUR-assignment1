import functions as fc
import matplotlib.pyplot as plt
import pickle
import numpy as np

#Initialize a seed for the rest of the program
seed = 5283597438

#Importing the class 'functions' and initialise the seed
F = fc.functions(seed)

#Generating N=1000 random numbers and saving it in rand_nums
N = 1000
rand_nums = list([0])*N
for i in range(1000):
	rand_nums[i] = F.RNG()

#Scatter plot the random numbers for x[i+1] (x[1:1000]) against x[i] (x[0:999])
plt.scatter(rand_nums[1:1000],rand_nums[0:999],s=1,color='black')
plt.xlabel(r'$x_{i+1}$')
plt.ylabel(r'$x_{i}$')
plt.savefig('plots/rng_1000.png')
plt.close()

#Generating one million random numbers and saving it in rand_nums
N = 1000000
rand_nums = list([0])*N
for i in range(N):
	rand_nums[i] = F.RNG()

#plotting the result in a histogram to check for uniformity
plt.hist(rand_nums,bins=20)
plt.xlabel('Random Value')
plt.savefig('plots/rng_mil.png')
plt.close()

#Saving the last produced random number from the LCG to continue generate 
#unique pseudo random numbers in the next exercises without initializing 
#the seed (and other instances) as user
with open('instances.pkl','wb') as output:
	pickle.dump(F,output,pickle.HIGHEST_PROTOCOL)
output.close()