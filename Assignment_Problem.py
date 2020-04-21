import numpy as np
import pulp as p
import re
import seaborn as sns
import matplotlib.pyplot as plt

#similarity is 18 to 18 matrix 
#it contains number of overlapping points between each cluster and each region 
#It is considered as the weight of corresponding edge.

w = np.load("similarity.npy")
x_axis_labels = np.load("x_axis_labels.npy")

n = w.shape[0]


#Assigning a maximizing Linear Programming problem
prob = p.LpProblem("ClusterAssigment", p.LpMaximize)

#Create a binary variable
#It determines which edge should be included
#And which one should be eliminated
assign_vars = p.LpVariable.dicts("X",
                                 [(i, j) for i in range(n)
                                  for j in range(n)],
                                 0, 1, p.LpBinary)

#Objective function: Maximizing the sum of weights of included links
prob += p.lpSum(w[i,j] * assign_vars[i,j] for i in range(n) for j in range(n))

#Constrain: There must be either one or zero link originating from a node
for j in range(n):
    prob += p.lpSum(assign_vars[i,j] for i in range(n)) <= 1

#Constrain: There must be either one or zero link going to a node
for i in range(n):
    prob += p.lpSum(assign_vars[i,j] for j in range(n)) <= 1
    
#Running the optimization
prob.solve()

#Result of "X"
for v in prob.variables():
    print(v.name,v.value())

#Converting the result to a numpy array
result = np.zeros(w.shape)
matching_points = 0
for v in prob.variables():
    string  = str(v.name)
    r = re.findall(r"\w+",string)
    i = int(r[1])
    j = int(r[2][1:])
    if v.value() == 1: 
        matching_points += w[i,j]
    result[i,j] = v.value()

# Visualising the result
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(240,10,as_cmap=True)
sns.heatmap(result,cmap=cmap, vmax=1, center=0,
            square=True, linewidths=.5, xticklabels=x_axis_labels)

plt.show()

#Percentage of matching points 
n_points = np.sum(w)
print("Percentage of matching points: %.4f"%(matching_points/n_points))


