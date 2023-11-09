import random
import itertools
import math
import csv
import numpy as np
from scipy.stats import norm
import pickle

# Generating the dataset of queries and clicks

queries = []
clicks = []
query = 'Q'
click = 'C'

with open('YandexRelPredChallenge.txt', 'r') as csvfile:
  reader = csv.reader(csvfile, delimiter='\t')
  for row in reader:
    if row[2] == query:
      queries.append(row)
    elif row[2] == click:
      clicks.append(row)

print("There are %i queries." % len(queries))
print("The shortest query contains %i documents." % (len(min(queries, key = len)) - 5))
print("The longest query contains %i documents." % (len(max(queries, key = len)) - 5))
print("There are %i clicks." % len(clicks))

# Optimizing the gamma and alpha parameters based on the Yandex query log 
# Using EM algorithm and formulas given by Markov  

#def obtain_click_parameters(clicks,queries):
"""
uniquepairs = []
for quer in queries:
urlids = quer[5:11]
for i in range(len(urlids)):
  unique = quer[3] + ',' + urlids[i]
  if unique not in [i[0] for i in uniquepairs]:
    uniquepairs.append([unique,1])
    count = count + 1
  else:
    indexx = uniquepairs.index([ind for ind in uniquepairs if unique in ind][0])
    uniquepairs[indexx][1] = uniquepairs[indexx][1] + 1
"""
with open('uniquepairs.data', 'rb') as filehandle:  
  uniquepairs = pickle.load(filehandle)

clicked = []
clickid = []
temp = 0
tempclic = [0]

for quer in queries:
	if quer[0] not in clickid:
	  	count = 0
	  	for clic in clicks:
	  		if temp == int(quer[0]):
	  			tempclic.append(clic[3])
	  		else: 
	  			clicked.append(tempclic)
	  			temp = temp + 1
	  			tempclic = [temp]
	  			clickid.append(clic[0])

#clicked[0] = clicked[0][0:4]
#clicked.append([13868,'411'])
#clicked.append([13869])
print(len(clicked))
with open('clicked.data', 'wb') as filehandle:  
    pickle.dump(clicked, filehandle)
with open('clickid.data', 'wb') as filehandle:  
    pickle.dump(clickid, filehandle)

with open('clicked.data', 'rb') as filehandle:  
	clicked = pickle.load(filehandle)
with open('clickid.data', 'rb') as filehandle:  
	clickid = pickle.load(filehandle)

print(len(clicked))
print(clicked[0:100])
#print(uniquepairs)
#print(len(clicked))

oldgammas = np.zeros(6)
oldalphas = []
gammas = np.zeros(6)
alphas = []
for i in range(len(uniquepairs)):
	temp = uniquepairs[i][0:2]
	temp.append(0)
	alphas.append(temp)
	oldalphas.append(temp)
alphas = np.asarray(alphas)
oldalphas = np.asarray(oldalphas)
goforward = True
difference = 0.01
iter = 0
while (goforward):
	count = 0
	for quer in queries:
	  id = quer[0]
	  urlid = quer[5:11]
	  qurlid = ''
	  j = 0
	  for url in urlid:
	    qurlid = quer[3] + ',' + url
	    alphaindex = np.where(alphas[:,0] == qurlid)
	    alphaindex = alphaindex[0]
	    if url in clicked[int(id)]:
	      gammas[j] = gammas[j] + 1
	      alphas[alphaindex,2] = float(alphas[alphaindex,2]) + 1
	    else:
	      gammas[j] = gammas[j] + oldgammas[j]*(1-float(oldalphas[alphaindex,2])) / (1 - oldgammas[j]*float(oldalphas[alphaindex,2]))  
	      alphas[alphaindex,2] = float(alphas[alphaindex,2]) + float(oldalphas[alphaindex,2])*(1-oldgammas[j]) / (1 - oldgammas[j]*float(oldalphas[alphaindex,2]))  
	    j = j + 1
	  count = count + 1
	  if (count % 5000) == 0:
	      print(count)
	      print(gammas)
	      print(alphas[0:100])
	for j in range(len(alphas)):
	  alphas[j][2] = (1 + float(alphas[j][2])) / (int(alphas[j][1])+2)
	  oldalphas[j][2] = float(alphas[j][2])
	for j in range(len(gammas)):
	  gammas[j] = (1+gammas[j])/(len(queries)+2)
	  if (abs(gammas[j] - oldgammas[j]) < difference):
	    goforward = False
	  oldgammas[j] = gammas[j]
	iter = iter + 1
	print('iteration: %i' % iter)
	print(gammas)
#return gammas, alphas
  
#4 [0.56411314 0.37068179 0.29549367 0.24244711 0.19852613 0.17065473]
#4 [0.50022533 0.29569979 0.22755148 0.18252781 0.14754686 0.1253613]
#gammas, alphas = obtain_click_parameters(clicks,queries)
#gammas = [0.50022533, 0.29569979, 0.22755148, 0.18252781, 0.14754686, 0.1253613]
#alpha = 0.9