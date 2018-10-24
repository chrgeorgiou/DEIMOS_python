from SNR_calculation import *

# Test the weight function function:
r = weight_function(10, 0.0, 0.0)
x = np.arange(-100, 100, step=1)
y = np.arange(-100, 100, step=1)
x, y = np.meshgrid(x,y)
# How much time does the reshaping take?
#import time
#t1 = time.time()
wf = np.reshape(map(r, x.flatten(), y.flatten()), (len(x[0]), len(y[0])))
#t2=time.time()
#wf_temp = map(r, x.flatten(), y.flatten()), (len(x[0]), len(y[0]))
#t3=time.time()
#print 'just func: %s, func and reshape: %s' %(t3-t2, t2-t1)

import matplotlib.pyplot as plt
plt.pcolormesh(wf)
plt.colorbar()
plt.show()

# Test the postage stamp cutout function:
image = '/data/KIDS450_129.0_-0.5_r_sci.fits'
x_cen, y_cen = 7754.01, 2662.7
x_lim, ylim = 50, 50
ps = create_poststamp(image, x_cen, y_cen, x_lim, ylim)
import matplotlib.pyplot as plt
plt.pcolormesh(ps)
plt.show()

# test the weight function centroid change
image = '/data/KIDS450_129.0_-0.5_r_sci.fits'
x_cen, y_cen = 7754.01, 2662.7
x_lim, ylim = 50, 50
ps = create_poststamp(image, x_cen, y_cen, x_lim, ylim)
wf = weight_function(10000, 0.0, 0.0, x_cen=6, y_cen=-3)
print SNR(ps, wf, 2e-12)


x = np.arange(-ps.shape[0]/2, ps.shape[0]/2, step=1) # Desired values in x-axis where to evaluate the weight function
y = np.arange(-ps.shape[1]/2, ps.shape[1]/2, step=1) # Desired values in y-axis where to evaluate the weight function
x, y = np.meshgrid(x,y) # Produces two (300, 300) arrays with the same column (or row) for each row (or column).
wf_ = map(wf, x.flatten(), y.flatten()) # Returns values of the weight function at all 300x300 points in a single 1-D array
wf_ = np.reshape(wf_, (len(x[0]), len(y[0])))
import matplotlib.pyplot as plt
plt.pcolormesh(ps)
plt.colorbar()
plt.contour(wf_)
plt.show()
