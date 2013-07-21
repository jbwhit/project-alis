import numpy as np
import almsgs
import alfunc_base
msgs=almsgs.msgs()

class PowerLaw(alfunc_base.Base) :
	"""
	Returns a 1-dimensional gaussian of form:
	p[0] = amplitude
	p[1] = x offset
	p[2] = dispersion (sigma)
	"""
	def __init__(self, prgname="", getinst=False, verbose=2):
		self._idstr   = 'powerlaw'							# ID string for this class
		self._pnumr   = 2								# Total number of parameters fed in
		self._keywd   = dict({'specid':[], 'blind':False})			# Additional arguments to describe the model --- 'input' cannot be used as a keyword
		self._keych   = dict({'specid':0,  'blind':0})				# Require keywd to be changed (1 for yes, 0 for no)
		self._keyfm   = dict({'specid':"", 'blind':""})				# Format for the keyword. "" is the Default setting
		self._parid   = ['coefficient', 'index']		# Name of each parameter
		self._defpar  = [ 0.0,          0.0 ]			# Default values for parameters that are not provided
		self._fixpar  = [ None,         None ]			# By default, should these parameters be fixed?
		self._limited = [ [1  ,0  ],   [0  ,0  ] ]		# Should any of these parameters be limited from below or above
		self._limits  = [ [0.0,0.0],   [0.0,0.0] ]		# What should these limiting values be
		self._svfmt   = [ "{0:.8g}",  "{0:.8g}"]		# Specify the format used to print or save output
		self._prekw   = []								# Specify the keywords to print out before the parameters
		# DON'T CHANGE THE FOLLOWING --- it tells ALIS what parameters are provided by the user.
		tempinput = self._parid+self._keych.keys()                             #
		self._keywd['input'] = dict(zip((tempinput),([0]*np.size(tempinput)))) #
		########################################################################
		self._verbose = verbose
		if getinst: return

	def call_CPU(self, x, p, ae='em', mkey=None, ncpus=1):
		"""
		Define the functional form of the model
		--------------------------------------------------------
		x  : array of wavelengths
		p  : array of parameters for this model
		--------------------------------------------------------
		"""
		def model(par):
			"""
			Define the model here
			"""
			return par[0]*(x**par[1])
		#############
		yout = np.zeros((p.shape[0],x.size))
		for i in range(p.shape[0]):
			yout[i,:] = model(p[i,:])
		if ae == 'em': return yout.sum(axis=0)
		else: return yout.prod(axis=0)

	def parin(self, i, par, parb):
		"""
		This routine converts a parameter in the input model file
		to the parameter used in 'call'
		--------------------------------------------------------
		When writing a new function, one should change how each
		input parameter 'par' is converted into a parameter used
		in the function specified by 'call'
		--------------------------------------------------------
		"""
		if   i == 0: pin = par
		elif i == 1: pin = par
		return pin

	def set_vars(self, p, level, mp, ival, wvrng=[0.0,0.0], spid='None', levid=None, nexbin=None, getinfl=False):
		"""
		Return the parameters for a Gaussian function to be used by 'call'
		The only thing that should be changed here is the parb values
		"""
		levadd=0
		params=np.zeros(self._pnumr)
		parinf=[]
		for i in range(self._pnumr):
			parb = None
			if mp['mtie'][ival][i] != -1:
				getid = mp['tpar'][mp['mtie'][ival][i]][1]
			else:
				getid = level+levadd
				levadd+=1
			params[i] = self.parin(i, p[getid], parb)
			if mp['mfix'][ival][i] == 0: parinf.append(getid)
		if nexbin is not None:
			if nexbin[0] == "km/s": return params, 1
			elif nexbin[0] == "A" : return params, 1
			else: msgs.bug("bintype "+nexbin[0]+" should not have been specified in model function: "+self._idstr, verbose=self._verbose)
		elif getinfl: return params, parinf
		else: return params
