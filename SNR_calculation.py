import numpy as np
from astropy.io import fits


def SNR(imagedata, wf, noise_var):
	"""

	:param imagedata: 2-D array containing the image data.
	:param wf: Weight-function function (see "weight_function()).
	:param noise_var: Variance of the noise in the image. Assumed to be constant.
	:return: Signal-to-Noise ratio of your image, weighted with the weight function employed.
	"""
	# TODO: What if the image is too large? Have a check
	#imagedata = fits.getdata(image, 1) # FIXME: what should the number be here? Put a try/except statement or something?
	x = np.arange(-imagedata.shape[0]/2, imagedata.shape[0]/2, step=1, dtype=np.int)
	y = np.arange(-imagedata.shape[1] / 2, imagedata.shape[1] / 2, step=1, dtype=np.int)
	x, y = np.meshgrid(x, y)
	wf_= np.reshape(map(wf, x.flatten(), y.flatten()), (len(x[0]), len(y[0])))
	print np.sum(wf_), np.trapz(np.trapz(wf_, x[0,:]), y[:,0])
	nom = np.sum(imagedata*wf_)**2.#np.trapz(np.trapz(imagedata*wf_, x[0,:]), y[:,0])**2.
	denom = np.sum(noise_var*wf_**2.)#np.trapz(np.trapz(noise_var*wf_**2., x[0,:]), y[:,0])
	print np.sqrt(nom), np.sqrt(denom)
	# TODO: Cross-check with DEIMOS. Questions: Do I get the flux right? np.sum or 2*np.trapz (which is faster)? Is the nom/denom correct?
	# TODO: SNR increases with wf scale, flux decreases (greatly!!). Very large wf scale does not return image sum.
	return np.sqrt(nom/denom)


def weight_function(scale, e1=0.0, e2=0.0, x_cen=0.0, y_cen=0.0):
	"""
	Generate a multivariate elliptical gaussian weight function.
	:param scale: Scale factor of the gaussian. Units should be the same as the coordinate system's units.
	:param e1: ellipticity component along the x-axis
	:param e2: ellipticity component along the y-axis
	:param x_cen: Shift of the weight function in the direction of the x-axis.
	:param y_cen: ...of the y-axis.
	:return: a python lambda function that can be called at any (x,y) point.

	# Usage:
	wf = weight_function(1., 0.5, 0.6)
	x = np.linspace(-1, 1, 300) # Desired values in x-axis where to evaluate the weight function
	y = np.linspace(-1, 1, 300) # ...in y-axis
	x, y = np.meshgrid(x,y) # Produces two (300, 300) arrays with the same column (or row) for each row (or column).
	wf_ = wf(0.2, 0.5) # returns the value of the weight function at (x,y)=(0.2,0.5)
	wf_ = map(wf, x.flatten(), y.flatten()) # Returns values of the weight function at all 300x300 points in a single 1-D array
	wf_ = np.reshape(wf_, (len(x[0]), len(y[0]))) # Reshapes the 1-D array into (300,300)
	import matplotlib.pyplot as plt
	plt.pcolormesh(wf)
	plt.colorbar()
	plt.show()
	flux = np.trapz(np.trapz(wf_, x[0,:]), y[:,0]) # This is equal to 1.
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

# test
image = 'testimage.fits'
x_cen, y_cen = 150, 150
x_lim, ylim = 100, 100
ps = create_poststamp(image, x_cen, y_cen, x_lim, ylim)
wf = weight_function(10)
print SNR(ps, wf, 0.01)
# Flux should be ~25.
# TODO: Compare with image_moments.py