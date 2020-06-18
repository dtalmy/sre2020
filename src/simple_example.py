from pandas import *
from batch_fitting_class import *
import matplotlib.backends.backend_pdf

###########################################################
# load and prep data (the way you do this may be different)
###########################################################

# This will read in the entire excel
ss = read_excel("../data/DMSP_dosage.xlsx", "substrate_forpy")

# sampling times
dtimes = array(ss['T'])

# control data
a1 = array(ss['A_2090'])
a1sd = array(ss['A_2090_sd'])

# treatment data
a2 = array(ss['A_379_DMSP'])
b2 = array(ss['B_379_DMSP'])
a2sd = array(ss['A_379_DMSP_sd'])
b2sd = array(ss['B_379_DMSP_sd'])

###########################################################
# fit a control curve
###########################################################

# pack data into a dictionary
cont_2090 = {'htimes': dtimes, 'hms': a1, 'hss': a1sd}
# the class is setup so that you can choose a list of parameters and it will automatically determine which model structure to use
model = ['aff', 'mum', 'delth']
# setup the object for this sample dataset and the specific model
pmod = all_mods(cont_2090, model,nits=1000,pits=100,burnin=500)

# plot
f1, ax1 = subplots()
pmod.plot_data(ax1)
# this function will do the fitting and also plot the best fits to the axes (not sure if this is best)
pmod.do_fitting(ax1)
# automatically add axes labels with appropriate scales
pmod.double_labels(ax1)

ax1.semilogy()

###########################################################
# first fit the control curve
###########################################################
# pack data into a dictionary
inf_2090 = {'htimes': dtimes, 'vtimes': dtimes, 'hms': a2, 'vms': b2, 'hss': a2sd, 'vss': b2sd}

model = ['mum','phi', 'beta', 'lambd'] # parameters you're more familiar with
pmod = all_mods(inf_2090, model,nits=3200,pits=320,burnin=1600)

f2, ax2 = subplots(1,2,figsize=[9,4.5])
pmod.plot_data(ax2)
pmod.do_fitting(ax2)
pmod.double_labels(ax2)

ax2[0].semilogy()
ax2[1].semilogy()

# save fig
#f1.savefig('../figures/sample_control')
#f2.savefig('../figures/sample_infected')

show()
