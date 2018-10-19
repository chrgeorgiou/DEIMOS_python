from SNR_calculation import *

# Test the weight function function:
r = weight_function(0.1, 0.0, 0.0)
x = np.linspace(-1, 1, 300)
y = np.linspace(-1, 1, 300)
x, y = np.meshgrid(x,y)
# How much time does the reshaping take?
import time
t1 = time.time()
wf = np.reshape(map(r, x.flatten(), y.flatten()), (len(x[0]), len(y[0])))
t2=time.time()
wf_temp = map(r, x.flatten(), y.flatten()), (len(x[0]), len(y[0]))
t3=time.time()
print 'just func: %s, func and reshape: %s' %(t3-t2, t2-t1)

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