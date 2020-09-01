import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys
from PIL import Image
from pylab import  xlim,ylim,ylabel, legend, boxplot, setp, axes
import io
import os

matplotlib.rcParams["font.size"] = 16
statsumn = 0
statavesum = 0
np.random.seed(1)
index = np.arange(0,50000001,200000)
curve_es=[]
cdir = os.path.dirname(os.path.realpath(__file__))

if len(sys.argv) == 1:
    cpath = cdir +'/' #+'/xhalfcheetah/'
    files = os.listdir(cpath)
    print("Plotting data contained in:")
    for f in files:
        if "statS" in f:
            print(f)
            f_toload = cpath+f
            #print(f_toload)
            #input("eee")
            stat = np.load(f_toload)
            size = np.shape(stat)
            #newsize = (int(size[0] / 7), 7)
            #stat = np.resize(stat, newsize)
            #stat = np.transpose(stat)    
            tmp = stat[0]
            tmp -= tmp % -2
            best_values = stat[1]
            indexes = []
            for j in index:
                val = min(tmp, key=lambda tmp: abs(tmp - j))
                ind = np.where(tmp==val)
                indexes.append(ind[0][0])       
            temp = list(set(indexes))
            temp.sort()
            if len(best_values[temp]) == len(index):
                curve_es.append(best_values[temp])
            #if (statsumn == 0):
                #statl = len(stat[0])
                #statsum = np.zeros((6,statl))
            #col = np.random.uniform(low=0.0, high=1.0, size=3)
            #plt.plot(stat[0],stat[2],label=f, linewidth=1,  color=col)
            #statsum = statsum + stat
            #statavesum += 1
            statsumn = statsumn + 1
        #statsum = statsum / float(statavesum)
        #plt.plot(statsum[0],statsum[2],label='ave', linewidth=1,  color='r')
    if (statsumn == 0):
        print("\033[1mERROR: No stat*.npy file found\033[0m") 
    #else:
        #curve_toplot = np.mean(np.asarray(curve),axis =0)
        #plt.plot(index, curve_toplot)
        #plt.legend()
        #plt.show()
#curve_es = curve_es *0.0165
# Number of replications
N = 19
b=[]
curve = []
eval = []
index = np.arange(0,50000001,200000)



#curve_toplot = np.mean(np.asarray(curve),axis =0)
curve_toplot_es = np.mean(np.asarray(curve_es),axis =0)
# confidence intervals
alpha = 0.90
p = ((1.0-alpha)/2.0) * 100

lower = np.percentile(curve_es, p, axis=0)
p = (alpha+((1.0-alpha)/2.0)) * 100
upper = np.percentile(curve_es, p, axis=0)

plt.plot(index, curve_toplot_es, 'k', color='#1B2ACC',label = 'es')
#plt.fill_between(index, lower, upper,
#    alpha=0.5, edgecolor='#1B2ACC', facecolor='#089FFF',
#    antialiased=True)
#coeff * 0.0165
'''
curve_toplot = np.mean(np.asarray(curve),axis =0)
# confidence intervals
alpha = 0.90
p = ((1.0-alpha)/2.0) * 100

lower = np.percentile(curve, p, axis=0)
p = (alpha+((1.0-alpha)/2.0)) * 100
upper = np.percentile(curve, p, axis=0)

plt.plot(index, curve_toplot* 0.0165, 'k', color='#CC4F1B',label = 'ppo')
plt.fill_between(index, lower* 0.0165, upper* 0.0165,
    alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848',
    antialiased=True)
'''
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(9, 5)
#ylim(0,60)
#plt.title("HalfCheetahBullet")
plt.xlabel('step * 1e7')
plt.ylabel('performance')

plt.legend(loc="upper left")
#plt.savefig('/home/nicola/figureFrontiers/Cheetah_prog_new.png')
plt.show()


