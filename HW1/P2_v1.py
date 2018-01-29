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


pkl_file = open('Cl_r_0.00.pkl', 'rb')
Cl = pickle.load(pkl_file)
pkl_file.close()

Cl = np.array(Cl[1:,0],dtype=float)

lmax=2000

x_range = np.arange(lmax+1)

# calculate Bl and calculate Cl^s

def l_l1(l):

    return l*(l+1)/(2*np.pi)

l_l1 = np.vectorize(l_l1)

Cl = Cl/l_l1(x_range)

def Bl_beam(l):
    sigma_b = 0.4245*10/60/180*np.pi

    return exp(-l**2*sigma_b**2/2)


Bl_beam = np.vectorize(Bl_beam)

# print(Bl_beam(x_range))
delta_omega = 10/60/180*np.pi
sigma_n = 2.5

cln = delta_omega*sigma_n**2

# calculate cls:

cls = Bl_beam(x_range)**2*Cl

# print(Cl)

cl_tot = cls+cln

# signal fraction:

signal_fraction = cls/cl_tot


# plot cls cln and total:



# Let's read and plot Behroozi
plot_path = "/Users/caojunzhi/Desktop/NYU/2018 Spring/Data_analysis/HW1/plots/"

font = {'family': 'normal',
                'weight': 'bold',
                'size': 25}

matplotlib.rc('font', **font)


plt.plot(x_range,l_l1(x_range)*cls,"ro",label=r"$C_l^s$")
plt.plot(x_range,l_l1(x_range)*cln*np.ones(len(x_range)),"bo",label=r"$C_l^N$")
plt.plot(x_range,l_l1(x_range)*(cls+cln),"ko",label=r"$C_l^{total}$")


plt.xlim([2,lmax])
plt.legend()
plt.ylabel(r'$\ell(\ell+1)C_\ell/ (2\pi \mu{\rm K}^2)$')
plt.xlabel(r'$\ell$')
plt.title('Angular power spectrum')

# plt.tight_layout()

###save

fig = matplotlib.pyplot.gcf()

# adjust the size based on the number of visit

fig.set_size_inches(16.5, 16.5)

save_path = plot_path + "Cls_Cln_Cltotal" + ".png"
fig.savefig(save_path, dpi=300)

plt.close()


# ratio:


font = {'family': 'normal',
                'weight': 'bold',
                'size': 25}

matplotlib.rc('font', **font)


mask = signal_fraction>0.8

# print(x_range[mask])


plt.plot(x_range,signal_fraction,"ko")
plt.plot(x_range[mask],signal_fraction[mask],"ro",label="Signal fraction >0.8")


print(x_range[mask])

plt.xlim([2,lmax])
plt.legend()
plt.ylabel(r'Signal fraction')
plt.xlabel(r'$\ell$')
plt.suptitle('Signal fraction')

###save

fig = matplotlib.pyplot.gcf()

# adjust the size based on the number of visit

fig.set_size_inches(16.5, 16.5)

save_path = plot_path + "Signal_fraction" + ".png"
fig.savefig(save_path, dpi=300)

plt.close()


