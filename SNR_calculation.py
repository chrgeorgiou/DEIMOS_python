import numpy as np
from astropy.io import fits

def wf(scale, e1, e2):
	return lambda x, y : np.exp(-(x * x + y * y)/(2 * scale**2.))

def SNR(image, wf):
	# TODO: What if the image is too large? Have a check
	imagedata = fits.getdata(image)

"""
r = wf(0.5,0,0)
x = np.linspace(-1,1,100)
y = np.linspace(-1,1,100)
x,y = np.meshgrid(x,y)
import matplotlib.pyplot as plt
plt.imshow(r(x,y))
plt.colorbar()
plt.show()
"""

def create_poststamp(image, x_cen, y_cen, x_lim, y_lim):
	imagedata = fits.getdata(image)

image = '/data/KIDS450_129.0_-0.5_r_sci.fits'
x_cen, y_cen = 7754.01, 2662.7
x_lim, ylim = 300, 300
# TODO: Look at Segmap pipeline for this.