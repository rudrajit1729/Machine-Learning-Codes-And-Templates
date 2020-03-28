#Upper Confidence Bound

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

#Implementing the UCB algorithm
import math
N = 10000
d = 10
#Step 1
ads_selected = []
number_of_selections = [0] * d #Each version of ad selected vector Ni(n)
sum_of_rewards = [0] * d # Sum of rewards for each i Ri(n)
total_reward = 0
for n in range(0, N): #No. of rounds
    ad = 0
    max_upper_bound = 0
    #Step 2
    for i in range(0, d): # No. of ads
        if number_of_selections[i]> 0:
            average_reward = sum_of_rewards[i] / number_of_selections[i] #ri(n)
            delta_i = math.sqrt(3/2 * math.log(n+1)/number_of_selections[i]) # n+1 as index starts with 0
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400 # 10^400 Set to large number--> initially for d rounds ad d selected at round d
        
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
            
    #Step3
    #Append ad i to ads selected list.ads chosen on basis on n obs.
    ads_selected.append(ad) # ad selected for round n
    number_of_selections[ad] += 1 # No. of selection for ad i updated
    reward = dataset.values[n,ad]
    sum_of_rewards[ad] += reward
    total_reward += reward 
#Total reward = 2178. Random selection generated 1251
#Ad finally selected for campaign = Max selected ad in number_of_selection

#Visualizing the result
plt.hist(ads_selected)
plt.title('Histogram of ads selection')
plt.xlabel('Ads')
plt.ylabel('No. of times each ad was selected')
plt.show()
    
    
        