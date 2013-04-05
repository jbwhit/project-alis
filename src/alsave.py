import os
import copy
import numpy as np
import datetime
import almsgs
import alfunc_base
from matplotlib import pyplot as plt
from matplotlib import cm as pltcm
from alutils import getreason
msgs=almsgs.msgs()

def save_asciifits(fname, wave, flux, error, model):
	"""
	Save the best-fitting model into an ascii file.
	"""
	np.savetxt(fname+".dat", np.transpose(( wave, flux, error, model )) )
	return

def save_fitsfits():
	"""
	Save the best-fitting model into fits files.
	"""


def save_onefits(fname, wvarr, fxarr, erarr, mdarr):
	"""
	Save the best-fitting model into a single fits file with multiple extensions.
	"""
	msgs.error("Single fits file output has not been implemented yet.")

def save_modelfits(slf):
	msgs.info("Writing out the model fits", verbose=slf._argflag['out']['verbose'])
	stf = 0
	fit_fnames = np.array([])
	fnames = np.array([])
#	stf, enf = [0 for all in slf._posnfull], [0 for all in slf._posnfull]
	usdtwice, usdtwind, usdtwext = np.array([]), np.array([]).astype(np.int64), np.array([]).astype(np.str)
	if slf._argflag['out']['onefits']: wvarr, fxarr, erarr, mdarr = [], [], [], []
	# If we are generating fakedata, find the peak value of the model
	if slf._argflag['generate']['data'] and (slf._argflag['generate']['peaksnr'] > 0.0 or slf._argflag['generate']['skyfrac'] > 0.0):
		modmax = 0.0
		for sp in range(len(slf._posnfull)):
			for sn in range(len(slf._posnfull[sp])-1):
				ll = slf._posnfull[sp][sn]
				lu = slf._posnfull[sp][sn+1]
				maxval = np.max(slf._modfinal[sp][ll:lu])
				if maxval > modmax: modmax = maxval
		peakerr = modmax/slf._argflag['generate']['peaksnr']
		objterr = np.sqrt(peakerr**2 - modmax*slf._argflag['generate']['skyfrac'])
		objtsnr = modmax/objterr
		slf._fluxfull = copy.deepcopy(slf._modfinal)
	# Now iterate through the spectra and save the output
	for sp in range(len(slf._posnfull)):
		for sn in range(len(slf._posnfull[sp])-1):
			ll = slf._posnfull[sp][sn]
			lu = slf._posnfull[sp][sn+1]
			# Check if the outfile already exists:
			fname = slf._snipnames[sp][sn]
			fspl = fname.split('.')
			if slf._argflag['generate']['data']:
				if os.path.exists(slf._snipnames[sp][sn]):
					fnoext = fname.rstrip('.'+fspl[-1])+'_model'
				else:
					fnoext = fname.rstrip('.'+fspl[-1])
			else:
				fnoext = fname.rstrip('.'+fspl[-1])+'_fit'
			an = np.where(fit_fnames == fnoext)
			if np.size(an) != 0: # The same snip is used more than once as input
				un = np.where(usdtwice == fnoext)
				if np.size(un) == 0: # First time this snip has been used twice
					usdtwice = np.append(usdtwice, fnoext)
					usdtwind = np.append(usdtwind, 2)
					usdtwext = np.append(usdtwext, fspl[-1])
				else: # This snip is seen more than twice
					usdtwind[un][0] += 1
				# Now that the relevant additions have been made to the arrays, get the index
				un = np.where(usdtwice == fnoext)
				fnoext += "%02i" % (usdtwind[un][0])
			# Prepare the model array:
			modelout = -9.999999999E9*np.ones(slf._wavefull[sp][ll:lu].size)
			w = np.where((slf._wavefull[sp][ll:lu] >= slf._posnfit[sp][2*sn+0]) & (slf._wavefull[sp][ll:lu] <= slf._posnfit[sp][2*sn+1]))
			modelout[w] = slf._modfinal[sp][ll:lu][w]
			# Add noise if we are generating fakedata
			if slf._argflag['generate']['data']:
				if not os.path.exists(slf._snipnames[sp][sn]):
					if slf._argflag['generate']['peaksnr'] > 0.0 or slf._argflag['generate']['skyfrac'] > 0.0:
						slf._fluefull[sp][ll:lu] = slf._modfinal[sp][ll:lu]/objtsnr
						slf._fluxfull[sp][ll:lu] += np.random.normal(0.0, np.sqrt(slf._fluefull[sp][ll:lu]**2 + modmax*slf._argflag['generate']['skyfrac']))
				else:
					slf._fluxfull[sp][ll:lu] += np.random.normal(0.0, slf._fluefull[sp][ll:lu])
			# Now that we have the output name, send the data away to be written to file
			if slf._argflag['out']['onefits']:
				# Store the fits files in an array and write them out at the end of the for loop
				ext = '.fits'
				wvarr.append(slf._wavefull[sp][ll:lu])
				fxarr.append(slf._fluxfull[sp][ll:lu])
				erarr.append(slf._fluefull[sp][ll:lu])
				mdarr.append(modelout)
			elif fspl[-1] in ["fits", "fit"]:
				# Write out this snip to a fits file
				ext = '.fits'
				save_fitsfits(fnoext, slf._wavefull[sp][ll:lu], slf._fluxfull[sp][ll:lu], slf._fluefull[sp][ll:lu], modelout)
			else:
				# Write out the data to an ascii (.dat) file.
				ext = '.dat'
				save_asciifits(fnoext, slf._wavefull[sp][ll:lu], slf._fluxfull[sp][ll:lu], slf._fluefull[sp][ll:lu], modelout)
			fit_fnames = np.append(fit_fnames, fnoext)
			fnames = np.append(fnames, fnoext+ext)
	if slf._argflag['out']['onefits']: # The user has requested that all model fits be written into a single fits file:
		outspl = slf._argflag['run']['modname'].split('.')
		outname = slf._argflag['run']['modname'].rstrip('.'+outspl[-1])+'_fit'
		save_onefits(outname, wvarr, fxarr, erarr, mdarr)
	else: # For snips that were used twice, rename the first instance to have suffix "01"
		for i in range(len(usdtwice)):
			os.rename(usdtwice[i]+"."+usdtwext[i],usdtwice[i]+"01."+usdtwext[i])
	msgs.info("Saved absorption line fits", verbose=slf._argflag['out']['verbose'])
	return fnames

def print_model(params, mp, errs=None, reletter=False, blind=False, verbose=2):
	function=alfunc_base.call(getfuncs=True, verbose=verbose)
	funcinst=alfunc_base.call(getinst=True, verbose=verbose)
	funccall=alfunc_base.call(verbose=verbose)
	level=0
	outstring = ""
	errstring = "#\n# Errors:\n#\n"
	donecv=[]
	donezl=[]
	lastemab=""
	for i in range(len(mp['mtyp'])):
		#if errs is not None and mp['emab'][i] == "cv": continue
		mtyp = mp['mtyp'][i]
		if mp['emab'][i] != lastemab:
			if   mp['emab'][i]=="em": aetag = "emission"
			elif mp['emab'][i]=="ab": aetag = "absorption"
			elif mp['emab'][i]=="cv": aetag = "convolution"
			elif mp['emab'][i]=="zl": aetag = "zerolevel"
			outstring += " "+aetag+"\n"
			errstring += "#"+aetag+"\n"
			lastemab = mp['emab'][i]
		if errs is None:
			funcinst[mtyp]._keywd = mp['mkey'][i]
			outstr, level = funccall[mtyp].parout(funcinst[mtyp], params, mp, i, level)
			if outstr in donecv or outstr in donezl: continue
			if mp['emab'][i] == "cv": donecv.append(outstr) # Make sure we don't print convolution more than once.
			if mp['emab'][i] == "zl": donezl.append(outstr) # Make sure we don't print zerolevel more than once.
			outstring += outstr
		else:
			funcinst[mtyp]._keywd = mp['mkey'][i]
			outstr, errstr, level = funccall[mtyp].parout(funcinst[mtyp], params, mp, i, level, errs=errs)
			if outstr in donecv or outstr in donezl: continue
			if mp['emab'][i] == "cv": donecv.append(outstr) # Make sure we don't print convolution more than once.
			if mp['emab'][i] == "zl": donezl.append(outstr) # Make sure we don't print zerolevel more than once.
			outstring += outstr
			errstring += errstr
	if blind: return outstring
	if errs is None:
		return outstring
	else:
		return outstring, errstring

def save_model(slf,params,errors,info,printout=True,extratxt=["",""]):
	"""
	Save the input model into an output script
	that can be run as input.
	"""
	msgs.info("Saving the best-fitting model parameters", verbose=slf._argflag['out']['verbose'])
	filename = extratxt[0]+slf._argflag['run']['modname']+'.out'+extratxt[1]
	prestring = "#\n#  Generated by ALIS on {0:s}\n#\n".format(datetime.datetime.now().strftime("%d/%m/%y at %H:%M:%S"))
	prestring += "#   Running Time (hrs)  = {0:f}\n".format(info[0])
	prestring += "#   Initial Chi-Squared = {0:f}\n".format(slf._chisq_init)
	prestring += "#   Bestfit Chi-Squared = {0:f}\n".format(info[1])
	prestring += "#   Degrees-of-Freedom  = {0:d}\n".format(info[2])
	prestring += "#   Num. of Iterations  = {0:d}\n".format(info[3])
	prestring += "#   Convergence Reason  = {0:s}\n".format(getreason(info[4],verbose=slf._argflag['out']['verbose']))
	prestring += "\n"
	inputmodl = "#\n"
	for i in range(len(slf._parlines)):
		prestring += slf._parlines[i]
		inputmodl += "#   "+slf._parlines[i]
	prestring +="\ndata read\n"
	inputmodl += "#   data read\n"
	msgs.bug("INCLUDE THE BEST-FIT FWHM HERE --- PLACE THE ERROR BELOW", verbose=slf._argflag['out']['verbose'])
	for i in range(len(slf._datlines)):
		prestring += slf._datlines[i]
		inputmodl += "#   "+slf._datlines[i]
	prestring +="data end\n"
	inputmodl += "#   data end\n"
	prestring +="\nmodel read\n"
	inputmodl += "#   model read\n"
	modcomlin=[]
	modcomind=[]
	toutstring=''
	for i in range(len(slf._modlines)):
		if slf._modlines[i].split()[0] in ["fix", "lim"]: toutstring += slf._modlines[i].replace('\t',' ')
		if slf._modlines[i].lstrip()[0] == '#':
			modcomlin.append(slf._modlines[i].rstrip('\n'))
			modcomind.append(i)
		inputmodl += "#   "+slf._modlines[i]
	outstring, errstring = print_model(params,slf._modpass,errs=errors,verbose=slf._argflag['out']['verbose'])
	if printout and slf._argflag['out']['verbose'] != -1:
		print "\n####################################################"
		print outstring
		print errstring
		print "####################################################\n"
	# Reinsert the comments at the original locations
	outstrspl = (toutstring+outstring).split('\n')
	for i in range(len(modcomlin)): outstrspl.insert(modcomind[i],modcomlin[i])
	outstring = '\n'.join(outstrspl)
	# Include an end tag for the model
	outstring += "model end\n"
	inputmodl += "#   model end\n#\n\n"
	if slf._argflag['out']['overwrite']: ans='y'
	else: ans=''
	if os.path.exists(filename):
		while ans != 'y' and ans != 'n' and ans !='r':
			msgs.warn("File %s exists!" % (filename), verbose=slf._argflag['out']['verbose'])
			ans = raw_input(msgs.input()+"Overwrite? (y/n) or rename? (r) - ")
			if ans == 'r':
				fileend=raw_input(msgs.input()+"Enter new filename - ")
				filename = fileend
				if os.path.exists(filename): ans = ''
	if ans != 'n':
		infile = open(filename,"w")
		infile.write(prestring)
		infile.write(outstring)
		infile.write("\n"+errstring+"\n")
		infile.write("\n###################################################")
		infile.write("\n#                                                 #")
		infile.write("\n#          HERE IS A COPY OF THE INPUT MODEL      #")
		infile.write("\n#                                                 #")
		infile.write("\n###################################################\n")
		infile.write(inputmodl)
		infile.close()
		msgs.info("Saved output file successfully:"+msgs.newline()+filename, verbose=slf._argflag['out']['verbose'])

def save_covar(slf, covar):
	"""
	Save the covariance matrix into an output ascii file
	"""
	msgs.info("Writing out the covariance matrix for the best-fitting model parameters", verbose=slf._argflag['out']['verbose'])
	if covar is None:
		msgs.warn("Covariance matrix is 'None', did you interupt the fit?", verbose=slf._argflag['out']['verbose'])
		msgs.info("Not writing our covariance matrix", verbose=slf._argflag['out']['verbose'])
		return
	if slf._argflag['out']['overwrite']: ans='y'
	else: ans=''
	filename=slf._argflag['out']['covar']
	if os.path.exists(filename) or filename == "":
		while ans != 'y' and ans != 'n' and ans !='r':
			if filename == "":
				msgs.warn("You must provide a filename to save the covariance matrix!", verbose=slf._argflag['out']['verbose'])
				ans = 'r'
			else:
				msgs.warn("File %s exists!" % (filename), verbose=slf._argflag['out']['verbose'])
				ans = raw_input(msgs.input()+"Overwrite? (y/n) or rename? (r) - ")
			if ans == 'r':
				fileend=raw_input(msgs.input()+"Enter new filename - ")
				filename = fileend
				if os.path.exists(filename): ans = ''
	if ans != 'n':
		fnspl = filename.split('.')
		if fnspl[-1] in ['fit','fits']:
			msgs.bug("Incorporate save fits file for covariance", verbose=slf._argflag['out']['verbose'])
		else:
			np.savetxt(filename, covar)
		msgs.info("Saved covariance matrix successfully:"+msgs.newline()+filename, verbose=slf._argflag['out']['verbose'])
		# Generate the correlation matrix from the covariance matrix
		outsize = np.int(np.sqrt(np.shape(np.where(covar!=0.0))[1]))
		if np.float(outsize) != np.sqrt(np.shape(np.where(covar!=0.0))[1]): msgs.bug("Error when deriving correlation matrix.", verbose=slf._argflag['out']['verbose'])
		corrM = np.zeros((outsize,outsize))
		sig    = np.sqrt(np.diag(covar))
		sigsig = np.dot(sig[:,np.newaxis],sig[np.newaxis,:])
		w = np.where(covar != 0.0)
		wc = np.where(corrM == 0.0)
		corrM[wc] = covar[w]/sigsig[w]
#		corrM = np.zeros(covar.shape)
#		corrM = covarNZ/sigsig
		# Mask out the fixed parameters (where the covariance matrix is 0.0)
#		w = np.where(covarNZ == 0.0)
#		mask = np.zeros(corrM.shape)
#		mask[w] = 1.0
		# Create a new array with the mask applied and plot it
#		cplt = np.ma.array(corrM, mask=mask)
		# Choose a colormap without white and set the bad pixels to white
		cmap = pltcm.get_cmap('jet',10)
#		cmap.set_bad('w')
		plt.imshow(corrM, interpolation="nearest", cmap=cmap, vmin=-1.0, vmax=1.0)
		plt.title("Correlation Matrix for: "+filename)
		tks=np.linspace(-1.0,1.0,11,endpoint=True)
		cbar=plt.colorbar(ticks=tks)
		plt.savefig(filename.rstrip(fnspl[-1])+'png')
		msgs.info("Saved image of covariance matrix to:"+msgs.newline()+filename.rstrip(fnspl[-1])+'png', verbose=slf._argflag['out']['verbose'])
	return

def modlines(slf, params, mp, reletter=False, blind=False, verbose=2):
	level=0
	linesarr = []
	donezl=[]
	lastemab=""
	for i in range(len(mp['mtyp'])):
		if mp['emab'][i] != lastemab:
			if   mp['emab'][i]=="em": aetag = "emission"
			elif mp['emab'][i]=="ab": aetag = "absorption"
			elif mp['emab'][i]=="cv": aetag = "convolution"
			elif mp['emab'][i]=="zl": aetag = "zerolevel"
			if aetag != "convolution": linesarr += [aetag]
			lastemab = mp['emab'][i]
		mtyp = mp['mtyp'][i]
		slf._funcinst[mtyp]._keywd = mp['mkey'][i]
		outstr, level = slf._funccall[mtyp].parout(slf._funcinst[mtyp], params, mp, i, level)
		if mp['emab'][i] == "zl": donezl.append(outstr) # Make sure we don't print zerolevel more than once.
		if aetag != 'convolution' and outstr not in donezl: linesarr += [outstr]
	return linesarr

def save_smfiles(filename, fnames, elnames, elwaves, elcomps, rdshft):
	pages = int(np.ceil((elnames.size+3.0)/24.0))
	panels_left=elnames.size
	panels_done=0
	ps_names, ps_waves, ps_fnams, ps_cparr = [], [], [], []
	for pg in range(0,pages):
		ps_names.append([])
		ps_waves.append([])
		ps_fnams.append([])
		ps_cparr.append([])
		# Determine the number of panels for this page
		if pg == 0: panppg = 21
		else: panppg = 24
		if panels_left <= panppg:
			ps_names[pg] = elnames[panels_done:]
			ps_waves[pg] = elwaves[panels_done:]
		else:
			ps_names[pg] = elnames[panels_done:panels_done+panppg]
			ps_waves[pg] = elwaves[panels_done:panels_done+panppg]
		# Now prepare the text to go into each panel
		for i in range(0,ps_waves[pg].size):
			ps_fnams[pg].append(fnames[panels_done+i])
			ps_cparr[pg].append(elcomps[panels_done+i])
		panels_done += ps_names[pg].size
		panels_left -= panppg
	# Now prepare the SuperMongo file for each page
	for pg in range(0,pages):
		numlines = len(ps_names[pg])
		outstring = "#\n#  Generated by Voigt_Fit on %s\n#\n\n" % (datetime.datetime.now().strftime("%d/%m/%y at %H:%M:%S"))
		outstring += "dev ppcf %s_page%02i.ps\nload default\nxtcolours\n\n" % (filename,pg+1)
		outstring += "set redshift=%8.7f\n" % (rdshft)
		outstring += "set yvalcomps = { 1.0 1.13 }\n"
		outstring += "set xvalcomps = { 0.0 0.0 }\n\n"
		if pg == 0:
			# Create a panel for the user to plot a damped Lya line if they wish.
			outstring += "#\n#  First Row, across the page\n#\n\n"
			outstring += "angle 0.000001\nlweight 2\nltype 0\nctype black\n"
			outstring += "location 3000 29000 28900 32000\n"
			outstring += "limits 1180 1251 -0.05 1.25\n"
			outstring += "ticksize 5 10 0.5 0.5\n"
			outstring += "expand 0.70\nbox 1 0\n"
			outstring += "relocate 1180 0.0\nputlabel 4 \"0.0 \"\n"
			outstring += "relocate 1180 0.5\nputlabel 4 \"0.5 \"\n"
			outstring += "relocate 1180 1.0\nputlabel 4 \"1.0 \"\n"
			outstring += "relocate 1251 0.0\nputlabel 6 \" 0.0\"\n"
			outstring += "relocate 1251 0.5\nputlabel 6 \" 0.5\"\n"
			outstring += "relocate 1251 1.0\nputlabel 6 \" 1.0\"\n\n"
			outstring += "#\n# Plot Lya line profile\n#\n\n"
			outstring += "data \"<INSERT_HI_Lya_FILENAME_HERE>\"\n"
			outstring += "xcolumn 1\nycolumn 2\n"
			outstring += "set x_r = x/(1.0+redshift)\n"
			outstring += "histogram x_r y if(y > -999)\n\n"
			outstring += "data \"<INSERT_HI_Lya_MODELNAME_HERE>\"\n"
			outstring += "xcolumn 1\nycolumn 2\n"
			outstring += "ctype red\nlweight 2\n"
			outstring += "connect x y if(y > -0.9)\n\n"
			outstring += "ctype blue\nrelocate 1180 1.0\nltype 3\nlweight 2\ndraw 1251 1.0\n"
			outstring += "ctype green\nrelocate 1251 0.0\nltype 1\nlweight 2\ndraw 1180 0.0\n\n"
			outstring += "#\n# x-axis label\n#\n\n"
			outstring += "relocate 1216 -0.5\n"
			outstring += "expand 0.75\nctype black\nlweight 2\n"
			outstring += "putlabel 5 Wavelength (\\AA)\n\n"
			x1, x2 = 3000, 11000
			y1, y2 = 24300, 27400
			rown = 2
		else:
			x1, x2 = 3000, 11000
			y1, y2 = 28800, 31900
			rown = 1
		# Now loop through all the metal lines.
		drawxlabel = True
		if numlines >= 4: drawxlabel = False
		posn = 1
		for i in range(0,numlines):
			outstring += "#\n# Row %2i, Panel %2i\n#\n\n" % (rown,posn)
			outstring += "angle 0.000001\nlweight 2\nltype 0\nctype black\n"
			outstring += "location %5i %5i %5i %5i\n" % (x1,x2,y1,y2)
			outstring += "limits -125 75 -0.05 1.25\n"
			outstring += "ticksize 25 50 0.5 0.5\nexpand 0.70\n"
			if drawxlabel: outstring += "box 1 0\n"
			else: outstring += "box 0 0\n"
			if posn == 1:
				outstring += "relocate -125 0.0\nputlabel 4 \"0.0 \"\n"
				outstring += "relocate -125 0.5\nputlabel 4 \"0.5 \"\n"
				outstring += "relocate -125 1.0\nputlabel 4 \"1.0 \"\n"
			elif posn == 3:
				outstring += "relocate 75 0.0\nputlabel 6 \" 0.0\"\n"
				outstring += "relocate 75 0.5\nputlabel 6 \" 0.5\"\n"
				outstring += "relocate 75 1.0\nputlabel 6 \" 1.0\"\n"
			outstring += "\n#\n# Plot %s %4i line profile\n#\n\n" % (ps_names[pg][i],ps_waves[pg][i])
			outstring += "relocate -125 1.0\nltype 3\nlweight 1\nctype blue\ndraw 75 1.0\n"
			outstring += "relocate -125 0.0\nltype 1\nlweight 1\nctype green\ndraw 75 0.0\n\n"
			outstring += "ltype 0\nctype black\nlweight 2\n\n"
			outstring += "data \"%s\"\n" % (ps_fnams[pg][i])
			outstring += "read wave 1\nread flux 2\nread modl 4\n"
			outstring += "set x_r=wave/(1+redshift)\n"
			outstring += "set x_vel=(x_r-%8.4f)/%8.4f*299792.458\n" % (ps_waves[pg][i],ps_waves[pg][i])
			outstring += "histogram x_vel flux if(flux > -999)\n"
			outstring += "lweight 2\nctype red\nltype 0\n"
			outstring += "connect x_vel modl if(modl > -0.9)\n\n"
			outstring += "ltype 0\n"
			for j in range(0,len(ps_cparr[pg][i])/3):
				if ps_cparr[pg][i][2*j] == 0.0: cstr = 'black'
				else: cstr = 'red'
				outstring += "set xvalcomps[0] = %s\n" % ps_cparr[pg][i][3*j+1]
				outstring += "set xvalcomps[1] = %s\n" % ps_cparr[pg][i][3*j+1]
				outstring += "ctype %s\nconnect xvalcomps yvalcomps\n" % (cstr)
			outstring += "ltype 0\nctype black\nlweight 2\n\n"
			outstring += "expand 0.65\n"
			outstring += "relocate -120 0.15\n"
			outstring += "putlabel 6 %s \\lambda %4i\n" % (ps_names[pg][i],ps_waves[pg][i])
			outstring += "expand 0.70\n\n"
			posn += 1
			x1 += 9000
			x2 += 9000
			if posn == 4 and i != numlines-1:
				posn = 1
				x1, x2 = 3000, 11000
				y1 -= 3400
				y2 -= 3400
			if i == numlines-4: drawxlabel = True # Start putting a velocity scale on the plots
		# Now include a label on the x-axis
		outstring += "#\n# x-axis label\n#\n\n"
		outstring += "location 3000 29000 %5i %5i\n" % (y1-1700,y2-4000)
		outstring += "limits -1 1 -1 1\nticksize 10 10 10 10\n\n"
		outstring += "relocate 0 0\nexpand 0.75\nctype black\nlweight 2\n"
		outstring += "putlabel 5 Velocity Relative to {\it z}_{\\rm abs} = $(redshift) (km s^{-1})\n\n"
		outstring += "# End of Plot!\n"
		# Now write the file
		outname = filename+"_page%02i.sm" % (pg+1)
		fout = open(outname, 'w')
		fout.write(outstring)
		fout.close()
	return

