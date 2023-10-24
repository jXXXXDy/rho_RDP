

from tkinter.ttk import Style
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.spatial import distance
import random

iter = 20

rho = 0.1
colSize = 1000
rowSize = 10
N = colSize
M = rowSize

metric = pd.DataFrame(np.zeros((3, 8)), columns=['ori', 'con0', 'con01','con001','uni01','lap01','uni001','lap001'], index=['Avg', 'MAE', 'VAE'])

jsdist = pd.Series(np.zeros(7))

lin1 = pd.DataFrame(np.zeros((colSize,8)), columns=['ori', 'Con0','Con01','Con001','uni01','lap01','uni001','lap001'])
lin2 = pd.DataFrame(np.zeros((colSize,8)), columns=['ori', 'Con0','Con01','Con001','uni01','lap01','uni001','lap001'])
lin3 = pd.DataFrame(np.zeros((colSize,8)), columns=['ori', 'Con0','Con01','Con001','uni01','lap01','uni001','lap001'])
lin4 = pd.DataFrame(np.zeros((colSize,8)), columns=['ori', 'Con0','Con01','Con001','uni01','lap01','uni001','lap001'])



for _ in range(iter):
    print("iteration: ", _+1)



    df1 = pd.DataFrame(np.random.uniform(0,1,(colSize, rowSize)))
    df2 = pd.DataFrame(np.zeros((colSize, rowSize)))
    df3 = pd.DataFrame(np.zeros((colSize, rowSize)))
    df4 = pd.DataFrame(np.zeros((colSize, rowSize)))
    df5 = pd.DataFrame(np.zeros((colSize, rowSize)))
    df6 = pd.DataFrame(np.zeros((colSize, rowSize)))
    df7 = pd.DataFrame(np.zeros((colSize, rowSize)))
    df8 = pd.DataFrame(np.zeros((colSize, rowSize)))


    noise = pd.DataFrame(np.zeros((colSize, rowSize)))

    # normal
    #df1 = pd.DataFrame(np.random.normal(0.5, 0.1, (colSize,rowSize)))


    for j in range(N):
        for i in range(M):
            noise[i][j] = np.random.uniform(-1, 1)
            perturb = df1[i][j]+ noise[i][j]
            df2[i][j] = perturb

    for j in range(N):
        for i in range(M):
            noise[i][j] = 0
            while(abs(noise[i][j]) <= 0.1):
                noise[i][j] = np.random.uniform(-1, 1)
            perturb = df1[i][j]+ noise[i][j]
            df3[i][j] = perturb

    for j in range(N):
        for i in range(M):
            noise[i][j] = 0
            while(abs(noise[i][j]) <= 0.01):
                noise[i][j] = np.random.uniform(-1, 1)
            perturb = df1[i][j]+ noise[i][j]
            df4[i][j] = perturb


    for j in range(N):
        dist = 0
        while(dist / M <= 0.1):
            dist = 0
            noise_rho = np.random.uniform(-1,1,M)
            for i in range(M):
                perturb = df1[i][j]+ noise_rho[i]
                df5[i][j] = perturb
                dist += abs(df1[i][j]-df5[i][j])
            
    for j in range(N): 
        dist = 0
        while(dist / M <= 0.1): 
            dist = 0
            noise_rho = np.random.laplace(0.1,0.1,M) 
            for i in range(M):
                if random.random() < 0.5:
                    perturb = df1[i][j]+ noise_rho[i]
                else:
                    perturb = df1[i][j]- noise_rho[i]
                df6[i][j] = perturb
                dist += abs(df1[i][j]-df6[i][j])

    for j in range(N):
        dist = 0
        while(dist / M <= 0.01):
            dist = 0
            noise_rho = np.random.uniform(-1,1,M)
            for i in range(M):
                perturb = df1[i][j]+ noise_rho[i]
                df7[i][j] = perturb
                dist += abs(df1[i][j]-df7[i][j])
            
    for j in range(N): 
        dist = 0
        while(dist / M <= 0.01): 
            dist = 0
            noise_rho = np.random.laplace(0.01,0.1,M) 
            for i in range(M):
                if random.random() < 0.5:
                    perturb = df1[i][j]+ noise_rho[i]
                else:
                    perturb = df1[i][j]- noise_rho[i]
                df8[i][j] = perturb
                dist += abs(df1[i][j]-df8[i][j])


    js_con0 = list(distance.jensenshannon(df1, df2,axis=1))
    js_con01 = list(distance.jensenshannon(df1, df3,axis=1))
    js_con001 = list(distance.jensenshannon(df1, df4,axis=1))
    js_uni01 = list(distance.jensenshannon(df1, df5,axis=1))
    js_lap01 = list(distance.jensenshannon(df1, df6,axis=1))
    js_uni001 = list(distance.jensenshannon(df1, df7,axis=1))
    js_lap001 = list(distance.jensenshannon(df1, df8,axis=1))

    # # # if P and Q have not the same support, there exists some point x′ where p(x′)≠0 and q(x′)=0, making KL go to infinity. 

            
    jsdist += (np.average(js_con0), np.average(js_con01),np.average(js_con001),np.average(js_uni01),np.average(js_lap01),np.average(js_uni001),np.average(js_lap001))



    # if two distributions are the same, the Jensen-Shannon distance between them is 0.

    metric.loc['Avg'] += (np.average(df1), np.average(df2), np.average(df3),np.average(df4),np.average(df5),np.average(df6),np.average(df7),np.average(df8))
    metric.loc['MAE'] += (0, np.average(abs(df1-df2)), np.average(abs(df1-df3)),np.average(abs(df1-df4)),np.average(abs(df1-df5)),np.average(abs(df1-df6)),np.average(abs(df1-df7)),np.average(abs(df1-df8)))
    metric.loc['VAE'] += (0, (df1 - df2).abs().var().mean(),(df1 - df3).abs().var().mean(),(df1 - df4).abs().var().mean(),(df1 - df5).abs().var().mean(),(df1 - df6).abs().var().mean(),(df1 - df7).abs().var().mean(),(df1 - df8).abs().var().mean())
    simplesum = pd.Series(np.ones(rowSize))
    oddsum = pd.Series(np.ones(rowSize))
    increasesum = pd.Series(np.arange(1,rowSize+1))
    increaseoddsum = pd.Series(np.arange(1,rowSize+1))
    
    for i in range(1,rowSize,2):
        oddsum[i] = oddsum[i]*(-1)
        increaseoddsum[i] = increaseoddsum[i]*(-1)

    lin1 += pd.concat([df1.dot(simplesum),df2.dot(simplesum),df3.dot(simplesum),df4.dot(simplesum),df5.dot(simplesum),df6.dot(simplesum),df7.dot(simplesum),df8.dot(simplesum)], axis= 1, keys=['ori', 'Con0','Con01','Con001','uni01','lap01','uni001','lap001'])
    lin2 += pd.concat([df1.dot(oddsum),df2.dot(oddsum),df3.dot(oddsum),df4.dot(oddsum),df5.dot(oddsum),df6.dot(oddsum),df7.dot(oddsum),df8.dot(oddsum)], axis= 1, keys=['ori', 'Con0','Con01','Con001','uni01','lap01','uni001','lap001'])
    lin3 += pd.concat([df1.dot(increasesum),df2.dot(increasesum),df3.dot(increasesum),df4.dot(increasesum),df5.dot(increasesum),df6.dot(increasesum),df7.dot(increasesum),df8.dot(increasesum)], axis= 1, keys=['ori', 'Con0','Con01','Con001','uni01','lap01','uni001','lap001'])
    lin4 += pd.concat([df1.dot(increaseoddsum),df2.dot(increaseoddsum),df3.dot(increaseoddsum),df4.dot(increaseoddsum),df5.dot(increaseoddsum),df6.dot(increaseoddsum),df7.dot(increaseoddsum),df8.dot(increaseoddsum)], axis= 1, keys=['ori', 'Con0','Con01','Con001','uni01','lap01','uni001','lap001'])
# average result

pd.options.display.float_format = '{:.8f}'.format

metric = metric.div(iter)
jsdist = jsdist.div(iter)
lin1 = lin1.div(iter)
lin2 = lin2.div(iter)
lin3 = lin3.div(iter)
lin4 = lin4.div(iter)

print(jsdist)

# plt.title('jensen-shannon distance')
# plt.grid(True, axis='y')
# plt.bar(['con0', 'con01','con001','uni01','lap01','uni001','lap001'], jsdist, width =0.4)
# plt.show()

print(metric)
