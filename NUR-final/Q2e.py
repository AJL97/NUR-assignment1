import numpy as np
import pickle
import functions as fc
import matplotlib.pyplot as plt

#Loading the previous instances
with open('instances.pkl','rb') as _input:
	constants = pickle.load(_input)
_input.close()

#Importing the class 'functions' and initialising the objects used
F = fc.functions(constants.x,constants.d1,constants.d2,constants.d3,constants.A,100)

#Finding the analytical maximum - see derivation in the paper.
x_max = constants.d2*(((constants.d1-1)/(constants.d3))**(1/constants.d3))
y_max = F.prob_function(x_max)

#Sampling 1000 haloes, each containing 100 satellites
n=100 #Number of satellites
x_min=1e-4 #Minimum of radii
x_max=5 #Maximum of radii
y_min=0 #Minimum of probability (Maximum of prob. is defined as y_max (see above))

N = 1000 #Number of haloes
satellites = list([0])*20 #Satellite bins
all_satels_nested = list([[]])*N #Saving all the satellites in a nested list (Used for Q2g)
bins = np.logspace(np.log10(1e-4),np.log10(5), 21) #Log Space bins
widths = bins[1:len(bins)] - bins[0:len(bins)-1] #Widths of the log space bins
#Loop over 1000 haloes each producing 100 satellites
for i in range(N):
	#Sample the distribution with rejection sampling
	all_satels_nested[i] = F.reject_sampling(n,x_min,x_max,y_min,y_max,F.prob_function)
	#Creating a histogram of the produced 100 satellites
	hist,bins = np.histogram(all_satels_nested[i],bins=bins)
	#Adding the new histogram to the total satellite (1000*100) histogram
	satellites += hist

#Setting the positions of the bins by taking the middle of each bin
positions = (bins[0:len(bins)-1]+bins[1:len(bins)])/2
#Normalizing the bins by dividing each bin by the total sum times its width
satellites = satellites/(np.sum(satellites)*widths)

#Plot the produced histogram 
#(satellites *100 to create the average number at radius r 
#(instead of using a probability)
plt.bar(positions,satellites*100,log=True,color='green',edgecolor='black',width=np.array(widths),label='Binned haloes')
x = np.linspace(1e-4,5,1000)
y = F.N_function(x)
plt.loglog(x,y,color='red',label=r'$N(x)$')
plt.xlabel(r'$r$')
plt.ylabel('Number of occurances')
plt.title('Average number of satellites')
plt.legend()
plt.savefig("plots/samples_dis.png")
plt.close()

#Saving all the instances for the next exercises
with open('instances.pkl','wb') as output:
	pickle.dump(F,output,pickle.HIGHEST_PROTOCOL)
output.close()

#Save all the satellites that are in the nested list and save all the
#bins of the total 1000 haloes
np.savetxt('haloes.txt',all_satels_nested)
np.savetxt('bin_haloes.txt',satellites)