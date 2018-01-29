import sys, platform, os
from matplotlib import pyplot as plt
import numpy as np
import matplotlib
import math
import pickle
print('Using CAMB installed at %s'%(os.path.realpath(os.path.join(os.getcwd(),'..'))))
#uncomment this if you are running remotely and want to keep in synch with repo changes
#if platform.system()!='Windows':
#    !cd $HOME/git/camb; git pull github master; git log -1
sys.path.insert(0,os.path.realpath(os.path.join(os.getcwd(),'..')))
import camb
from camb import model, initialpower

####
# save path:


# Let's read and plot Behroozi
plot_path = "/Users/caojunzhi/Desktop/NYU/2018 Spring/Data_analysis/HW1/plots/"


def log10(x):
    if x > 0:
        return math.log10(x)
    else:
        return -np.inf


def exp(x):
    try:
        return math.exp(x)
    except:
        return np.inf


exp = np.vectorize(exp)
log10 = np.vectorize(log10)


# read cl at r=0:


pkl_file = open('Pk_linear_z_0.50.pkl', 'rb')
Pk = pickle.load(pkl_file)
pkl_file.close()

Pk = np.array(Pk[1:,:],dtype=float)


# noise

N = 686370

## Attention!! V is Gpc^3 and convert that to Mpc^3
V = 4.2*1000**3

n = N/V

P_n = 1/n*(0.67)**3

print(P_n)


P_total = 2.1**2*Pk[:,1]+np.ones(len(Pk[:,0]))*P_n


font = {'family': 'normal',
                'weight': 'bold',
                'size': 25}

matplotlib.rc('font', **font)


plt.loglog(Pk[:,0],2.1**2*Pk[:,1],"ro",label=r"$P^s(k)$")
plt.loglog(Pk[:,0],np.ones(len(Pk[:,0]))*P_n,"bo",label=r"$P^N(k)$")
plt.loglog(Pk[:,0],2.1**2*Pk[:,1]+np.ones(len(Pk[:,0]))*P_n,"ko",label=r"$P^{total}(k)$")



plt.legend()
plt.xlabel('log[k/h]')
plt.ylabel('log[P]')
plt.title('Matter power at z=0.5')

###save

fig = matplotlib.pyplot.gcf()

# adjust the size based on the number of visit

fig.set_size_inches(16.5, 16.5)

save_path = plot_path + "Pk_s" + ".png"
fig.savefig(save_path, dpi=300)

plt.close()


# fraction:

fraction = 2.1**2*Pk[:,1]/P_total


font = {'family': 'normal',
                'weight': 'bold',
                'size': 25}

matplotlib.rc('font', **font)


plt.loglog(Pk[:,0],fraction,"ko",label="Fraction")



plt.legend()
plt.xlabel('log[k/h]')
plt.ylabel('log[fraction]')
plt.title('Fraction at z=0.5')

###save

fig = matplotlib.pyplot.gcf()

# adjust the size based on the number of visit

fig.set_size_inches(16.5, 16.5)

save_path = plot_path + "P_fraction" + ".png"
fig.savefig(save_path, dpi=300)

plt.close()



