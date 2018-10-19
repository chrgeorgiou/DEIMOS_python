import numpy as np
from astropy.io import fits


def SNR(image, wf):
	# TODO: What if the image is too large? Have a check
	imagedata = fits.getdata(image)


def weight_function(scale, e1=0.0, e2=0.0, x_cen=0.0, y_cen=0.0):
	"""
	Generate a multivariate elliptical gaussian weight function.
	:param scale: Scale factor of the gaussian. Units should be the same as the coordinate system's units.
	:param e1: ellipticity component along the x-axis
	:param e2: ellipticity component along the y-axis
	:return: a python lambda function that can be called at any (x,y) point.

	Usage:
	r = weight_function(1., 0.5, 0.6)
	x = np.linspace(-1, 1, 300) # Desired values in x-axis where to evaluate the weight function
	y = np.linspace(-1, 1, 300) # ...in y-axis
	x, y = np.meshgrid(x,y) # Produces two (300, 300) arrays with the same column (or row) for each row (or column).
	wf = r(0.2, 0.5) # returns the value of the weight function at (x,y)=(0.2,0.5)
	wf = map(r, x.flatten(), y.flatten()) # Returns values of the weight function at all 300x300 points in a single 1-D array
	wf = np.reshape(wf, (len(x[0]), len(y[0]))) # Reshapes the 1-D array into (300,300)
	import matplotlib.pyplot as plt
	plt.pcolormesh(wf)
	plt.colorbar()
	plt.show()
	flux = np.trapz(np.trapz(wf, x[0,:]), y[:,0]) # This is equal to 1.
	"""
	assert e1**2.+e2**2. < 1., "Ellipticity larger than 1."
	C = np.matrix([[1-e1, -e2], [-e2, 1+e1]])
	detC = (1-e1)*(1+e1)-e2*e2
	detCinv = 1/detC
	return lambda x, y : np.exp( - ( float(np.array([x-x_cen, y-y_cen]).dot(C).dot(np.array([x-x_cen, y-y_cen]))) )/(2 * scale**2.))/np.sqrt(4*np.pi**2.*detCinv*scale**4.)


def create_poststamp(image, x_cen, y_cen, x_size, y_size):
	'''

	:param image: image where to cut the postage stamp
	:param x_cen: centroid of the postage stamp on the x-axis
	:param y_cen: on the y-axis
	:param x_size: size of the postage stamp on the x direction
	:param y_size: on the y direction
	:return: postage stamp(s)
	'''
	imagedata = fits.getdata(image)
	# TODO: Are sizes odd? make them.
	# TODO: Work with RA,DEC instead of image pixels
	try:
		# TODO: Are input arrays? Do they have the same length?
		len(x_cen)
		assert len(x_cen)==len(y_cen), "Centroid arrays have different lengths."
		x_cen_int = np.array(x_cen+0.5, dtype=np.int)
		y_cen_int = np.array(y_cen + 0.5, dtype=np.int)
	except:
		x_cen_int = int(x_cen + 0.5)
		y_cen_int = int(y_cen + 0.5)
	# FIXME: THE THING BELOW NOW WORKS ONLY FOR FLOAT CENTROIDS NOT ARRAYS
	image_ps = imagedata[y_cen_int - y_size/2:y_cen_int + y_size/2, x_cen_int - x_size/2:x_cen_int + x_size/2]
	return image_ps
