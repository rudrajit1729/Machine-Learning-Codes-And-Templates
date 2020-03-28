#Thompson Sampling

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

#Implementing the Thompson Sampling algorithm
import random
N = 10000
d = 10
#Step 1
ads_selected = []
number_of_rewards1 = [0] * d#N1i(n)
number_of_rewards0 = [0] * d #N0i(n)
total_reward = 0
for n in range(0,N):#No. of obs
    ad = 0
    max_random = 0
    #Step 2
    for i in range(0,d):#No. of ads
        random_beta = random.betavariate(number_of_rewards1[i] + 1, number_of_rewards0[i] + 1)
        if random_beta > max_random:#We pick the max random sampled result
            max_random = random_beta
            ad = i
    #Step 3
    #Append ad i to ads selected list.ads chosen on basis on n obs.
    ads_selected.append(ad)# ad selected for round n
    reward = dataset.values[n,ad]
    if reward == 1:
        number_of_rewards1[ad] += 1
        total_reward += reward
    else:
        number_of_rewards0[ad] += 1
#Total reward ~= 2600. Random selection generated 1251
#Ad finally selected for campaign = Max selected ad in number_of_selection 
        
#Visualizing the result 
plt.hist(ads_selected)
plt.title('Histogram of ads selection')
plt.xlabel('Ads')
plt.ylabel('No. of times each ad was selected')
plt.show()
