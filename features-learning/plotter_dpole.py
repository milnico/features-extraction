from baselines.common import plot_util as pu
import matplotlib.pyplot as plt
import numpy as np
import sys
from PIL import Image
from pylab import  xlim,ylim,ylabel, legend, boxplot, setp, axes
import io
import os

# Number of replications
N = 21
b=[]
curve = []
eval = []
index = np.arange(0,50000001,100000)

# demonstrate how to toggle the display of different elements:
fig = plt.figure()

cdirr = os.path.dirname(os.path.realpath(__file__))
for i in range(16,N):
	fname = cdirr 
	f_es = fname+ '/statS' + str(i)+'.npy'
	if os.path.exists(f_es):
		es_vec = np.load(f_es)
		b.append(es_vec[1,-1])
		best_values = es_vec[3]
		tmp = es_vec[0]
		tmp -= tmp % -2

		indexes = []
		for j in index:
			val = min(tmp, key=lambda tmp: abs(tmp - j))
			ind = np.where(tmp==val)
			indexes.append(ind[0][0])

		temp = list(set(indexes))
		temp.sort()
		curve.append(best_values[temp])

	# Plot figure
	#plt.plot(index, best_values[temp])
	#plt.show()
	#print(bp)
	#input("ee")
curve_toplot = np.mean(np.asarray(curve),axis =0)
plt.plot(index, curve_toplot)
fig.savefig('curve.png')
plt.xlabel('evals')
plt.ylabel('reward')
plt.title('cheetah')
plt.show()


# thrid boxplot pair
fig = plt.figure()
bp = boxplot(b)

# set axes limits and labels
#xlim(0,5000)
ylim(100,5000)
ylabel('fitness')
fig.savefig('boxcompare.png')
plt.show()
# Save the image in memory in PNG format
#png1 = io.BytesIO()
#fig.savefig(png1, format="png")

# Load this image into PIL
#png2 = Image.open(png1)

# Save as TIFF
#png2.save("walkerESA10_ppo.tiff")
#png1.close()

