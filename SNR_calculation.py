import numpy as np
from astropy.io import fits

def wf(scale, e1, e2):
	return lambda x, y : np.exp((x * x + y * y)/(2 * scale**2.))

#def SNR(image, wf):
