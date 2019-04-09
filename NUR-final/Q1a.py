import functions as fc

#Importing the class 'functions' to generate the poisson values
F = fc.functions()
	
#The poisson values to be evaluated
LKS = [[1,0],[5,10],[3,21],[2.6,40]]

#looping over the poisson values to be evaluated and printing them
for i in range(len(LKS)):
	print ('P(lambda={0},k={1}) = {2}'.format(LKS[i][0],LKS[i][1],
									   F.poisson_prob(LKS[i][0],LKS[i][1])))