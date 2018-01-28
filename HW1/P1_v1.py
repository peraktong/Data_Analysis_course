import sys, platform, os
from matplotlib import pyplot as plt
import numpy as np
import matplotlib
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

font = {'family': 'normal',
                'weight': 'bold',
                'size': 15}

matplotlib.rc('font', **font)

#Set up a new set of parameters for CAMB
pars = camb.CAMBparams()
#This function sets up CosmoMC-like settings, with one massive neutrino and helium set using BBN consistency
pars.set_cosmology(H0=67., ombh2=0.02222, omch2=0.1199, mnu=0.06, omk=0, tau=0.078)
pars.InitPower.set_params(ns=0.9652, r=0)
pars.set_for_lmax(2500, lens_potential_accuracy=0)

#calculate results for these parameters
results = camb.get_results(pars)

#get dictionary of CAMB power spectra
powers =results.get_cmb_power_spectra(pars, CMB_unit='muK')
for name in powers: print(name)

#plot the total lensed CMB power spectra versus unlensed, and fractional difference
totCL=powers['total']
unlensedCL=powers['unlensed_scalar']
print(totCL.shape)


##############

# Angular power spectra:


#You can calculate spectra for different primordial power spectra without recalculating everything
#for example, let's plot the BB spectra as a function of r
pars.WantTensors = True
results = camb.get_transfer_functions(pars)
lmax=2000

# There are 4 sub-plots:

plt.subplot(2,2,1)

rs = np.linspace(0,0.2,6)
for r in rs:
    inflation_params = initialpower.InitialPowerParams()
    inflation_params.set_params(ns=0.96, r=r)
    results.power_spectra_from_transfer(inflation_params)
    cl = results.get_total_cls(lmax, CMB_unit='muK')
    # save cl vs r:

    # save the total Cl:

    title = np.array(["TT", "EE", "BB", "TE"])

    data_save = np.vstack((title,cl))

    output = open("Cl_r_zz.pkl".replace("zz","{0:.2f}".format(r)), 'wb')
    pickle.dump(data_save, output)
    output.close()

    plt.loglog(np.arange(lmax+1),cl[:,0])
plt.xlim([2,lmax])
plt.legend(["$r = %s$"%r for r in  rs], loc='lower right');
plt.ylabel(r'$\ell(\ell+1)C_\ell^{TT}/ (2\pi \mu{\rm K}^2)$')
plt.xlabel(r'$\ell$')


plt.subplot(2,2,2)

rs = np.linspace(0,0.2,6)
for r in rs:
    inflation_params = initialpower.InitialPowerParams()
    inflation_params.set_params(ns=0.96, r=r)
    results.power_spectra_from_transfer(inflation_params)
    cl = results.get_total_cls(lmax, CMB_unit='muK')
    plt.loglog(np.arange(lmax+1),cl[:,1])
plt.xlim([2,lmax])
plt.legend(["$r = %s$"%r for r in  rs], loc='lower right');
plt.ylabel(r'$\ell(\ell+1)C_\ell^{EE}/ (2\pi \mu{\rm K}^2)$')
plt.xlabel(r'$\ell$')


plt.subplot(2,2,3)

rs = np.linspace(0,0.2,6)
for r in rs:
    inflation_params = initialpower.InitialPowerParams()
    inflation_params.set_params(ns=0.96, r=r)
    results.power_spectra_from_transfer(inflation_params)
    cl = results.get_total_cls(lmax, CMB_unit='muK')
    plt.loglog(np.arange(lmax+1),cl[:,2])
plt.xlim([2,lmax])
plt.legend(["$r = %s$"%r for r in  rs], loc='lower right');
plt.ylabel(r'$\ell(\ell+1)C_\ell^{BB}/ (2\pi \mu{\rm K}^2)$')
plt.xlabel(r'$\ell$')


plt.subplot(2,2,4)

rs = np.linspace(0,0.2,6)
for r in rs:
    inflation_params = initialpower.InitialPowerParams()
    inflation_params.set_params(ns=0.96, r=r)
    results.power_spectra_from_transfer(inflation_params)
    cl = results.get_total_cls(lmax, CMB_unit='muK')
    plt.loglog(np.arange(lmax+1),cl[:,3])
plt.xlim([2,lmax])
plt.legend(["$r = %s$"%r for r in  rs], loc='lower right');
plt.ylabel(r'$\ell(\ell+1)C_\ell^{TE}/ (2\pi \mu{\rm K}^2)$')
plt.xlabel(r'$\ell$')
plt.title('Angular power spectrum')

# plt.tight_layout()

###save

fig = matplotlib.pyplot.gcf()

# adjust the size based on the number of visit

fig.set_size_inches(16.5, 16.5)

save_path = plot_path + "Angular_power_spectra" + ".png"
fig.savefig(save_path, dpi=300)

plt.close()


#####
# Cl total = TT + TE+ BB + EE

font = {'family': 'normal',
                'weight': 'bold',
                'size': 25}

matplotlib.rc('font', **font)


rs = np.linspace(0,0.2,6)
for r in rs:
    inflation_params = initialpower.InitialPowerParams()
    inflation_params.set_params(ns=0.96, r=r)
    results.power_spectra_from_transfer(inflation_params)
    cl = results.get_total_cls(lmax, CMB_unit='muK')
    plt.loglog(np.arange(lmax+1),np.nansum(cl,axis=1))
plt.xlim([2,lmax])
plt.legend(["$r = %s$"%r for r in  rs], loc='lower right');
plt.ylabel(r'$\ell(\ell+1)C_\ell^{total}/ (2\pi \mu{\rm K}^2)$')
plt.xlabel(r'$\ell$')

plt.title('Total angular power spectrum')

# plt.tight_layout()

###save

fig = matplotlib.pyplot.gcf()

# adjust the size based on the number of visit

fig.set_size_inches(16.5, 16.5)

save_path = plot_path + "Angular_power_spectra_total" + ".png"
fig.savefig(save_path, dpi=300)

plt.close()




#######
# Matter spectra:


font = {'family': 'normal',
                'weight': 'bold',
                'size': 25}

matplotlib.rc('font', **font)


#Now get matter power spectra and sigma8 at redshift 0 and 0.8
pars = camb.CAMBparams()
pars.set_cosmology(H0=67.26, ombh2=0.02222, omch2=0.1199)
pars.set_dark_energy() #re-set defaults
pars.InitPower.set_params(ns=0.965)
#Not non-linear corrections couples to smaller scales than you want


# print(results.get_sigma8())

color_array = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

# red shift from 0 to 1 with bin size=5

for count in range(0,5):

    pars.set_matter_power(redshifts=[count*0.25], kmax=2.0)

    # Linear spectra
    pars.NonLinear = model.NonLinear_none
    results = camb.get_results(pars)
    kh, z, pk = results.get_matter_power_spectrum(minkh=1e-4, maxkh=1, npoints=200)
    s8 = np.array(results.get_sigma8())

    # Non-Linear spectra (Halofit)
    pars.NonLinear = model.NonLinear_both
    results.calc_power_spectra(pars)
    kh_nonlin, z_nonlin, pk_nonlin = results.get_matter_power_spectrum(minkh=1e-4, maxkh=1, npoints=200)



    for i, (redshift, line) in enumerate(zip(z, ['-', '--'])):

        plt.loglog(kh, pk[i, :], "o", color=color_array[count], ls=line)
        plt.loglog(kh_nonlin, pk_nonlin[i, :], "x", color=color_array[count], ls=line)

    # save linear
    title = np.array(["kh","pk"])

    data_save = np.vstack((title,np.c_[kh,pk.ravel()]))

    output = open("Pk_linear_z_zz.pkl".replace("zz","{0:.2f}".format(count*0.25)), 'wb')
    pickle.dump(data_save, output)
    output.close()

    # save non-linear:
    title = np.array(["kh", "pk"])

    data_save = np.vstack((title, np.c_[kh_nonlin, pk_nonlin.ravel()]))

    output = open("Pk_non_linear_z_zz.pkl".replace("zz", "{0:.2f}".format(count * 0.25)), 'wb')
    pickle.dump(data_save, output)
    output.close()

    
    plt.plot([],[],"o", color=color_array[count],label="Linear z=%.2f"%(count*0.25))
    plt.plot([], [], "x", color=color_array[count], label="Non-linear z=%.2f" % (count * 0.25))

    count += 1
    print("Doing z=%.2f" % (count * 0.25))

plt.legend()
plt.xlabel('k/h Mpc')
plt.ylabel('P(k)')
plt.title('Matter power from z=0 to 1')

# save:


###save

fig = matplotlib.pyplot.gcf()

# adjust the size based on the number of visit

fig.set_size_inches(16.5, 16.5)

save_path = plot_path + "Matter_spectra" + ".png"
fig.savefig(save_path, dpi=300)

plt.close()




