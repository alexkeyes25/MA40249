import numpy as np
from matplotlib import pyplot as plt
plt.rcParams.update({'font.size':22})
n_vals = np.array([4,8,16,32,64])

th_vel_COFS = np.array([338+81,1250+289,4802+1089,18818+4225, 74498+16641])
rt_dg_vel_COFS = np.array([360+192,1392+768,5472+3072,21696+12288,86400+49152])
bdm_dg_vel_COFS = np.array([480+192,1856+768,7296+3072,28928+12288])


#pres_DOFS = np.array([81, 289, 1089, 4225, 16641])


rt_dg_vel_errs_l2 = np.array([0.006259598,0.000679978,8.07E-05,9.91E-06,1.23E-06])
th_vel_errs_l2 = np.array([0.000472426, 2.82E-05, 1.71E-06, 1.06E-07, 6.58E-09])
bdm_dg_vel_errs_l2 = np.array([0.000550111,3.35E-05,1.94E-06,1.16E-07])

th_val_errs_h1 = np.array([0.019037311, 0.002363978, 0.00029284, 3.64E-05, 4.54E-06])

#plt.loglog(th_vel_COFS, th_val_errs_h1)

rt_dg_pres_errs_l2 = np.array([0.245521903, 0.057550307, 0.013417648, 0.003208063, 0.000782442])
th_pres_errs_l2 = np.array([0.003386666, 0.000349244, 3.15E-05, 2.79E-06, 2.46E-07])
bdm_dg_pres_errs_l2 = np.array([0.015862748, 0.001802236, 0.000198369, 2.26E-05])


#plt.loglog(vel_COFS, vel_errs_h1)
plt.loglog(th_vel_COFS, th_pres_errs_l2,marker='o' ,label='Taylor-Hood') # taylor-hood
plt.loglog(rt_dg_vel_COFS, rt_dg_pres_errs_l2,marker='s' , label='RT-DG') # rt-dg
plt.loglog(bdm_dg_vel_COFS, bdm_dg_pres_errs_l2, marker='^', label='BDM-DG') # bdm-dg
plt.xlabel('Degrees of Freedom (log scale)')
plt.ylabel('Error (log scale)')
plt.title('Pressure error in the L2 norm against DOFs - k = 3')
plt.legend()
plt.show()
