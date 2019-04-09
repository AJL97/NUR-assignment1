import numpy as np
import functions as fc
import matplotlib.pyplot as plt
import pickle

#The files with all the data
files = ['satgals_m11.txt','satgals_m12.txt',
		'satgals_m13.txt','satgals_m14.txt','satgals_m15.txt']

#Using the created table from the previous run to reduce computation time
data = np.loadtxt('abc_matrix.txt')
full_cube = data.reshape(15,16,26)

#Importing the class 'functions'
F = fc.functions()
F.cube = full_cube
#Creating a list containing 5 (number of datafiles) empty lists
#to save the found a,b,c values for each datafile
m_abc = [[]]*5
save_l = [0]*5
m=['m11','m12','m13','m14','m15']
#Target 'error' of the downhill simplex algorithm
#(basically a measure of size of the simplex)
targ_err = 1e-10
#Initial a,b,c guesses for each file
start = [[1.1,1.3,3.1],
		 [1.7,0.9,3.3],
		 [1.5,0.8,3.1],
		 [2.1,0.6,2.5],
		 [1.9,0.7,2]]
#Looping over each file
for i in range(len(files)):
	#Getting all the data from the files by ignoring all the '#' symbols
	x = []
	for line in open('Datafiles/'+files[i]):
		l = line.strip()
		if not l.startswith('#'):
			x.append(np.float64(l.split()[0]))
	
	#The number of haloes is the first element of the array
	haloes = x[0]
	#All the other elements are the r-coordinates of the satellites
	x = x[1:len(x)]
	#The average total number of satellites of 1 'super' halo
	#is the total number satellites in all haloes
	F.Nsat = len(x)
	
	#Saving the best points
	save_ans = list([])
	#Number of iterations
	iter = 0
	#Setting the initial guesses
	points = start[i]
	
	#First run of the downhill simplex algorithm with the
	#rough initial guesses
	points = F.downhill_simpl(x,F.log_function,points,targ_err)
	#Saving the a, b, and c values, and also the log-likelihood
	#given these values
	save_ans = points
	lowest = F.log_function(x,*points)

	#Storing the best values in the m_abc list
	m_abc[i] = save_ans

	#Steepness test - NOTE, this is not a cube but works in the same manner
	delta = 0.2
	unit_vec = np.zeros(3)
	log_l = 0
	indx = 0
	for j in range(6):
		if j>2:
			unit_vec[indx] = -1
		else: unit_vec[j] = 1
		log_l += abs(abs(lowest) - abs(F.log_function(x,*(save_ans+(delta*unit_vec)))))
		unit_vec[indx] = 0
		indx += 1
		if j==2:
			indx=0	
	save_l[i] = log_l/abs(lowest)
	#Printing the found a,b,c for a given mass
	print ('Mass 1e{0:} Msun/h: a={1:5.5f}, b={2:5.5f}, c={3:5.5f}'.format(i+11,save_ans[0],save_ans[1],save_ans[2]))

#Storing all the found values as instances and save it with pickle dump
F.m_abc = m_abc
F.logs_l = save_l
with open('instances.pkl','wb') as output:
	pickle.dump(F,output,pickle.HIGHEST_PROTOCOL)
output.close()