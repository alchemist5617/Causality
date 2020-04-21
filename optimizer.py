#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 13:47:41 2019

@author: Mohammad Noorbakhsh
"""

import pandas as pd
import numpy as np
import functions as ff
import math
import random
from collections import Counter
import csv   
import os

pair = "gbpjpy"
start_date = 2018
end_date = 2018
freq = 30

file_csv = '{}_{}_{}_{}.csv'.format(pair.upper(), freq, start_date, end_date)
file_csv_temp = '{}_{}_{}_{}Temp.csv'.format(pair.upper(), freq, start_date, end_date)
file_csv_clean = '{}_{}_{}_{}Clean.csv'.format(pair.upper(), freq, start_date, end_date)

curr = os.getcwd()
if not os.path.split(curr)[1]=='FX_data': os.chdir("./FX_data/")
if os.path.exists(file_csv_clean):
    data = pd.read_csv(file_csv_clean)
else:
    data = ff.construct_data(pair,start_date,end_date,freq)
    data.to_csv(file_csv_clean)

Close = data["Close"]
High = data["High"]
Low = data["Low"]
Time = data["Time"]
Volume = data["Volume"]

stochastic = ff.sstoc(data)
N = data.shape[0]

performance = []
limit_levels = []
stop_levels = []
distance_levels = []
min_levels = []
trailing_stop_levels = []

#PIP_RATIO = math.pow(10,-len(str(Close[0]).split('.')[1])+1)

sample = random.sample(list(Close.values), 1000)

pip_list = []
for i in range(1000):
    pip_list.append(math.pow(10,-len(str(sample[i]).split('.')[1])+1))

c = Counter(pip_list)
PIP_RATIO, count = c.most_common()[0]


INVERSE_PIP_RATIO = 1/PIP_RATIO
MARGIN = 4

def distant(first,second, Distance = 50):
    if((first-second)> Distance * PIP_RATIO):
        return True
    else:
        return False

#limits = np.arange(5,100,5)
#stops = np.arange(5,100,5)
#distances = [10,20,50,100,150,200,300,400,500,600,700,800,900,1000]
#distances = [50,100,150,200]

def performace(limit, stop, distance,trailing_stop):
    first_min = 1000
    first_max = 0
    second_min = 1000
    second_max = 0
    p = ff.position(PIP_RATIO)
    
    cumpips = 0
    state = -3
    uptrend = False
    max_value = 0
    min_value = 1000

    for i in range(N):
        if p.have_position:
                cumpips, state = p.close_position(cumpips, Time[i], Low[i], High[i], state)
                    
        if state == -3:
                if first_max < High[i]: first_max = High[i]
                if first_min > Low[i]: first_min = Low[i]
                    
                SC = ff.stochastic_crossover(i, stochastic)
                if SC==2: 
                    state = -21
                    first_min = first_max
                elif SC==1:
                    state = -20
                    first_max = first_min
        
        elif state == -20:
                if first_max < High[i]: first_max = High[i]
                    
                SC = ff.stochastic_crossover(i, stochastic)
                if SC==2: 
                    state = -22
                    first_min = first_max
                    
        elif state == -22:
                if first_min > Low[i]: first_min = Low[i]
                    
                SC = ff.stochastic_crossover(i, stochastic)
                if SC==1: 
                    state = -24
                    second_max = first_min
                    
        elif state == -24:
                if second_max < High[i]: second_max = High[i]
                    
                SC = ff.stochastic_crossover(i, stochastic)
                if SC==2: 
                    if second_max <= first_max:
                        uptrend = False
                        triggerprice = first_max
                        trendprice = first_min
                        state = 0
                    else:
                        uptrend = True
                        triggerprice = first_min
                        trendprice = second_max
                        state = 0
    
        elif state == -21:
                if first_min > Low[i]: first_min = Low[i]
                    
                SC = ff.stochastic_crossover(i, stochastic)
                if SC==1: 
                    state = -23
                    first_max = first_min
                    
        elif state == -23:
                if first_max < High[i]: first_max = High[i]
                    
                SC = ff.stochastic_crossover(i, stochastic)
                if SC==2: 
                    state = -25
                    second_min = first_max
                    
        elif state == -25:
                if second_min > Low[i]: second_min = Low[i]
                    
                SC = ff.stochastic_crossover(i, stochastic)
                if SC==2: 
                    if second_min <= first_min:
                        uptrend = False
                        triggerprice = first_max
                        trendprice = second_min
                        state = 0
                    else:
                        uptrend = True
                        triggerprice = first_min
                        trendprice = first_max
                        state = 0        
    
        elif state == 0:
            if uptrend:
                
                if max_value < High[i]: max_value = High[i]
                if min_value > Low[i]: min_value = Low[i]
                
                if distant(max_value,trendprice, distance):
                    triggerprice = min_value
                    #min_value = High[i]
                    state = 10
                  
                if distant(triggerprice,min_value, distance):  
                    state = 1
                    max_value = min_value
            else: 
                
                if max_value < High[i]: max_value = High[i]
                if min_value > Low[i]: min_value = Low[i]
                
                if distant(trendprice, min_value, distance):
                    triggerprice = max_value
                    #max_value = Low[i]
                    state = 10
                
                if distant(max_value,triggerprice, distance):
                    state = 1
                    min_value = max_value
        
        elif state == 10:  
            if uptrend:
                
                if max_value < High[i]: max_value = High[i]
                if min_value > Low[i]: min_value = Low[i]
    
                SC = ff.stochastic_crossover(i, stochastic)
                if SC==2:
                    trendprice = max_value 
                   # triggerprice = min_value
                    min_value = High[i]
                    state = 0
                
                if distant(triggerprice,min_value, distance):
                    trendprice = max_value
                    state = 1
            else:
                
                if max_value < High[i]: max_value = High[i]
                if min_value > Low[i]: min_value = Low[i]
                
                SC = ff.stochastic_crossover(i, stochastic)
                if SC == 1:
                    trendprice = min_value
                  #  triggerprice = max_value
                    max_value = Low[i]
                    state = 0
                    
                if distant(max_value,triggerprice, distance):
                    trendprice = min_value
                    state = 1
    
        elif state == 1:
            if uptrend:
                
                if max_value < High[i]: max_value = High[i]
                if min_value > Low[i]: min_value = Low[i]
                
                if distant(max_value,trendprice, distance):
                    if p.have_position:
                        state = 3
                        continue                
                    else:
                        state = 0
                        continue
                
                SC = ff.stochastic_crossover(i, stochastic)
                if SC==1: 
                    shootingprice = min_value
                    max_value = min_value
                    state = 2
            else:
                
                if max_value < High[i]: max_value=High[i]
                if min_value > Low[i]: min_value=Low[i]
                
                if distant(trendprice,min_value, distance):
                    if p.have_position:
                        state = 3
                        continue 
                    else:
                        state = 0
                        continue
                
                SC = ff.stochastic_crossover(i, stochastic)
                if SC==2:
                    shootingprice = max_value
                    min_value = max_value
                    state = 2 
        
        elif state == 2:
            
            if uptrend:
                
                if max_value < High[i]: max_value=High[i]
                if min_value > Low[i]: min_value=Low[i]
                
                if distant(max_value,trendprice, distance):
                    if p.have_position:
                        triggerprice = min_value
                        state = 13
                        min_value = High[i]
                        continue
                    else:
                        triggerprice = shootingprice
                        state = 10
                        min_value = High[i]
                        continue
                
                if distant(shootingprice,min_value, distance):
                    limit_price = shootingprice - (distance + limit) * PIP_RATIO
                    stop_price = shootingprice  - (distance - stop) * PIP_RATIO
                    uptrend, cumpips = p.take_position("Sell",uptrend,cumpips,shootingprice - distance * PIP_RATIO, abs(shootingprice - trendprice)*100, abs(triggerprice - trendprice)*100, Time[i], limit_price, stop_price, trailing_stop)
                    state = 3
                    trendprice = shootingprice
                    triggerprice = max_value
                    continue
            else:
                if max_value<High[i]: max_value=High[i]
                if min_value>Low[i]: min_value=Low[i];
                
                if distant(trendprice, min_value, distance):
                    if p.have_position:
                        triggerprice = max_value
                        state = 13
                        max_value = Low[i]
                        continue                 
                    else:
                        triggerprice = shootingprice
                        state = 10
                        max_value = Low[i]
                        continue
                
                if distant(max_value,shootingprice, distance):
                    limit_price = shootingprice + (distance + MARGIN + limit) * PIP_RATIO
                    stop_price = shootingprice + (distance  - stop) * PIP_RATIO
                    uptrend, cumpips = p.take_position("Buy",uptrend,cumpips,shootingprice + (distance + MARGIN) * PIP_RATIO, abs(shootingprice - trendprice)*100, abs(triggerprice - trendprice)*100, Time[i], limit_price, stop_price, trailing_stop)
                    state = 3
                    trendprice = shootingprice
                    triggerprice = min_value
                    continue
        
        elif state == 3:
            if uptrend:
                if max_value < High[i]: max_value = High[i]
                if min_value > Low[i]: min_value = Low[i]
                        
                if distant(max_value,trendprice, distance):
                    triggerprice = min_value
                    state = 13
                  
                if distant(triggerprice,min_value, distance):  
                    state = 1 
            else: 
                
                if max_value < High[i]: max_value = High[i]
                if min_value > Low[i]: min_value = Low[i]
                
                if distant(trendprice, min_value, distance):
                    triggerprice = max_value
                    state = 13
                
                if distant(max_value,triggerprice, distance):
                    state = 1
                    continue
        
        elif state == 13:  
            if uptrend:
                
                if max_value < High[i]: max_value = High[i]
                if min_value > Low[i]: min_value = Low[i]
    
                SC = ff.stochastic_crossover(i, stochastic)
                if SC==2:
                    trendprice = max_value
                    min_value = High[i]
                    state = 3
                
                if distant(triggerprice,min_value, distance):
                    trendprice = max_value
                    state = 1
                    continue
            
            else:
                
                if max_value < High[i]: max_value = High[i]
                if min_value > Low[i]: min_value = Low[i]
                
                SC = ff.stochastic_crossover(i, stochastic)
                if SC == 1:
                    trendprice = min_value
                    max_value = Low[i]
                    state = 3
                    
                if distant(max_value,triggerprice, distance):
                    trendprice = min_value
                    state = 1 
                    continue
    

    del p
 