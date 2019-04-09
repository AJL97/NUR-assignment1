import numpy as np
import functions as fc
import matplotlib.pyplot as plt

#Initialise the class 'functions' 
F = fc.functions()

#Generated data of Q2e
satels = np.loadtxt('haloes.txt')
bin_satels = np.loadtxt('bin_haloes.txt')

#Reshape the nested satellites to create one big array
satels_arr = np.array(satels.reshape(-1))

#Creating the bins and widths in logspace
bins = np.logspace(np.log10(1e-4),np.log10(5), 21)
widths = bins[1:len(bins)] - bins[0:len(bins)-1] 

#Finding the bin that contains the largest number of galaxies 
max_bin_satellites = bins[list(bin_satels).index(max(bin_satels))]
#The width of the bin that contains the largest number of galaxies
bin_width = widths[list(bin_satels).index(max(bin_satels))]
#Storing the satellites that belong in this bin
r_satels = satels_arr[((satels_arr < max_bin_satellites + bin_width)&(satels_arr >= max_bin_satellites))]

#Sorting the above created array with quicksort
#Note: another option would be F.sorting_index which sorts the index array.
F.quicksort(r_satels,min=0,max=len(r_satels)-1,sort=F.sorting)

#Creating an index array of r_satels to use for bisection
x_data = np.arange(0,len(r_satels),1)
#Percentiles to be evaluated
percentiles = [16,50,84]
#Finding the rank of the given percentile and array
x_eval = [F.rank(16,r_satels),F.rank(50,r_satels),F.rank(84,r_satels)]
for i in range(len(x_eval)):
	#Using bisection to find the indexes
	left_idx,right_idx = F.bisection_index_finding(x_eval[i],x_data)
	y = [r_satels[left_idx],r_satels[right_idx]]
	x = [x_data[left_idx],x_data[right_idx],x_eval[i]]
	#Linear interpolate to find the percentiles
	print ('{0}th percentile = {1}'.format(percentiles[i],F.linear_interp(x,y)))

#Finding all the satellites that are in the largest bin of the 1000 haloes histogram
nr_galaxy_halo = list([[]])*len(satels)
for i in range(len(satels)):
	nested_satels = np.array(satels[i])
	#Getting the satellites from the right bin
	new_satels = nested_satels[((nested_satels <= max_bin_satellites + bin_width)&(nested_satels >= max_bin_satellites))]
	#Saving the number of satellites from this bin
	nr_galaxy_halo[i] = len(new_satels)	

#Creating linear spaced bins 
bins = np.arange(min(nr_galaxy_halo),max(nr_galaxy_halo)+1,1)
#The mean of the number of galaxies 
mean = np.sum(nr_galaxy_halo)/len(nr_galaxy_halo)
#Plotting the poisson distribution
for i in range(len(bins)-1):
	plt.scatter(bins[i],F.poisson_prob(mean,bins[i]),color='red')
plt.scatter(bins[len(bins)-1],F.poisson_prob(mean,bins[len(bins)-1]),color='red',label='Poisson Dist.')
#Making a histogram of the number of galaxies that were in the largest bin of the
#1000 haloes for each halo (=100 galaxies)
plt.hist(nr_galaxy_halo,bins=bins,density=True,fc=(0, 0, 1, 0.5),label='Data')
plt.xlabel(r'Number of galaxies in radial bin $r$')
plt.ylabel('Normalized number of occurances')
plt.title(r'Radial bin $r = {0:.2}$ up until ${1:.2}$'.format(max_bin_satellites,max_bin_satellites+bin_width))
plt.legend()
plt.savefig('plots/radial_bin.png')
plt.close()