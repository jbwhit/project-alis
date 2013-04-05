import numpy as np
from matplotlib import pyplot as plt
from matplotlib import ticker
import almsgs
msgs=almsgs.msgs()

def make_plots_all(slf, model=None):
	msgs.info("Preparing data to be plotted", verbose=slf._argflag['out']['verbose'])
	wavearr, fluxarr, fluearr, modlarr = slf._wavefull, slf._fluxfull, slf._fluefull, slf._modfinal
	if model is not None: modlarr = model
	posnarr, dims = slf._posnfull, slf._argflag['plot']['dims']
	dspl = dims.split('x')
	if len(dspl) != 2:
		msgs.error("Panel plot dimensions passed incorrectly")
		return
	try:
		dspl[0] = int(dspl[0])
		dspl[1] = int(dspl[1])
	except:
		msgs.error("Panel plot dimensions passed incorrectly")
	panppg = dspl[0]*dspl[1]
	numsub = 0
	subids = []
	pltlst = []
	seen = set()
	sidlst = np.array([x for x in slf._snipid if x not in seen and not seen.add(x)])
	for i in range(len(sidlst)):
		subids.append([])
		pltlst.append(0)
		for j in range(len(slf._datopt['plotone'][i])):
			if slf._datopt['plotone'][i][j]:
				subids[i].append(0)
			else:
				numsub += 1
				subids[i].append(1)
	snpid=[]
	for i in range(len(slf._snipid)): snpid.append(np.where(sidlst==slf._snipid[i])[0][0])
	pages = int(np.ceil(numsub/float(panppg)))
	panels_left=numsub
	subpnl_done=0
	snips_done=0
	numone = 0
	pgcnt_arr = []
	ps_wvarr, ps_fxarr, ps_fearr, ps_mdarr, ps_disps = [], [], [], [], []
	po_wvarr, po_fxarr, po_fearr, po_mdarr, po_disps = [], [], [], [], []
	# Construct the arrays for the subplots
	if pages == 0: # Only doing single plots
		sp = snpid[snips_done]
		sn = pltlst[sp]
		while subids[sp][sn] == 0:
			po_disps.append([])
			po_wvarr.append([])
			po_fxarr.append([])
			po_fearr.append([])
			po_mdarr.append([])
			llo=posnarr[sp][sn]
			luo=posnarr[sp][sn+1]
			po_disps[numone].append(0.5*np.append( (wavearr[sp][llo+1]-wavearr[sp][llo]), (wavearr[sp][llo+1:luo]-wavearr[sp][llo:luo-1]) ))
			po_wvarr[numone].append(wavearr[sp][llo:luo])
			po_fxarr[numone].append(fluxarr[sp][llo:luo])
			po_fearr[numone].append(fluearr[sp][llo:luo])
			po_mdarr[numone].append(modlarr[sp][llo:luo])
			snips_done += 1
			numone += 1
			pltlst[sp] += 1
			sn = pltlst[sp]
			# If the snip number has gone beyond the array size, go to the next specid.
			if sn == np.size(subids[sp]):
				if snips_done == np.size(snpid): break
				sp = snpid[snips_done]
				sn = pltlst[sp]
				# If the next snip for the next specid is a single plot, don't break.
				if subids[sp][sn] != 0: break 
	else:# A combination of single + subplots (or just subplots)
		for pg in range(0,pages):
#			ps_names.append([])
#			ps_waves.append([])
			ps_disps.append([])
			ps_wvarr.append([])
			ps_fxarr.append([])
			ps_fearr.append([])
			ps_mdarr.append([])
#			ps_cparr.append([])
			# Determine the number of panels for this page
			if panels_left <= panppg: pgcnt = numsub-subpnl_done
			else: pgcnt = panppg
			for i in range(pgcnt):
				sp = snpid[snips_done+i]
				sn = pltlst[sp]
				while subids[sp][sn] == 0:
					po_disps.append([])
					po_wvarr.append([])
					po_fxarr.append([])
					po_fearr.append([])
					po_mdarr.append([])
					llo=posnarr[sp][sn]
					luo=posnarr[sp][sn+1]
					po_disps[numone].append(0.5*np.append( (wavearr[sp][llo+1]-wavearr[sp][llo]), (wavearr[sp][llo+1:luo]-wavearr[sp][llo:luo-1]) ))
					po_wvarr[numone].append(wavearr[sp][llo:luo])
					po_fxarr[numone].append(fluxarr[sp][llo:luo])
					po_fearr[numone].append(fluearr[sp][llo:luo])
					po_mdarr[numone].append(modlarr[sp][llo:luo])
					snips_done += 1
					numone += 1
					pltlst[sp] += 1
					sn = pltlst[sp]
					# If the snip number has gone beyond the array size, go to the next specid.
					if sn == np.size(subids[sp]):
						sp = snpid[snips_done+i]
						sn = pltlst[sp]
						# If the next snip for the next specid is a single plot, don't break.
						if subids[sp][sn] != 0: break 
				ll=posnarr[sp][sn]
				lu=posnarr[sp][sn+1]
#				if slf._argflag['plot']['xaxis'] == 'velocity': # For velocity:
#					ps_disps[pg].append(0.5*299792.458*np.append( (wavearr[ll+1]-wavearr[ll])/wavearr[ll], (wavearr[ll+1:lu]-wavearr[ll:lu-1])/wavearr[ll:lu-1]))
#					ps_wvarr[pg].append(299792.458*(wavearr[ll:lu]/(1.0+rdshft)-ps_waves[pg][i])/ps_waves[pg][i])
#				elif slf._argflag['plot']['xaxis'] == 'rest': # For rest wave:
#					ps_disps[pg].append(0.5*np.append( (wavearr[ll+1]-wavearr[ll])/(1.0+rdshft), (wavearr[ll+1:lu]-wavearr[ll:lu-1])/(1.0+rdshft) ))
#					ps_wvarr[pg].append(wavearr[ll:lu]/(1.0+rdshft))
#				else: # For observed wave:
				ps_disps[pg].append(0.5*np.append( (wavearr[sp][ll+1]-wavearr[sp][ll]), (wavearr[sp][ll+1:lu]-wavearr[sp][ll:lu-1]) ))
				ps_wvarr[pg].append(wavearr[sp][ll:lu])
#	
				ps_fxarr[pg].append(fluxarr[sp][ll:lu])
				ps_fearr[pg].append(fluearr[sp][ll:lu])
				ps_mdarr[pg].append(modlarr[sp][ll:lu])
#				ps_cparr[pg].append(comparr[sp][panels_done+i])
				pltlst[sp] += 1
			snips_done += pgcnt
			subpnl_done += pgcnt
			panels_left -= panppg
			pgcnt_arr.append(pgcnt)
#	ps_nw = [ps_names, ps_waves]
	ps_wfem = [ps_wvarr, ps_fxarr, ps_fearr, ps_mdarr]
	po_wfem = [po_wvarr, po_fxarr, po_fearr, po_mdarr]
	msgs.info("Prepared {0:d} panels in subplots".format(subpnl_done), verbose=slf._argflag['out']['verbose'])
	msgs.info("Prepared {0:d} panels in single plots".format(numone), verbose=slf._argflag['out']['verbose'])
	numpagesA = plot_drawplots(pages, ps_wfem, pgcnt_arr, ps_disps, dspl, slf._argflag, verbose=slf._argflag['out']['verbose'])
	numpagesB = plot_drawplots(numone, po_wfem, np.ones(numone).astype(np.int), po_disps, [1,1], slf._argflag, numpages=pages, verbose=slf._argflag['out']['verbose'])
	msgs.info("Plotted {0:d} pages".format(numpagesA+numpagesB), verbose=slf._argflag['out']['verbose'])

def plot_drawplots(pages, wfemarr, pgcnt, disp, dims, argflag, numpages=0, verbose=2):
	"""
	Plot the fitting results in mxn panels.
	"""
	fig = []
	# Determine which pages should be plotted
	plotall = False
	if argflag['plot']['pages'] == 'all': plotall = True
	else: pltpages = argflag['plot']['pages'].split(',')
#	mmpltx = np.array([-120.0,120.0])
	pgnum = 0
	for pg in range(pages):
		if not plotall:
			if str(pg+1+numpages) not in pltpages:
				msgs.info("Skipping plot page number {0:d}".format(pg+1+numpages), verbose=argflag['out']['verbose'])
				continue
		fig.append(plt.figure(figsize=(12.5,10), dpi=80))
		fig[pgnum].subplots_adjust(hspace=0.1, wspace=0.1, bottom=0.07, top=0.98, left=0.04, right=0.98)
		for i in range(pgcnt[pg]):
			w = np.where(wfemarr[3][pg][i] > -0.5)
			if np.size(w[0]) == 0:
				msgs.warn("There was no model data found for a panel", verbose=argflag['out']['verbose'])
			ax = fig[pgnum].add_subplot(dims[0],dims[1],i+1)
			ax.plot([wfemarr[0][pg][i].min(),wfemarr[0][pg][i].max()],[0.0,0.0], 'g--')
			ax.plot([wfemarr[0][pg][i].min(),wfemarr[0][pg][i].max()],[1.0,1.0],'b--')
			ax.plot(wfemarr[0][pg][i]+disp[pg][i],wfemarr[1][pg][i], 'k-', drawstyle='steps')
			ax.plot(wfemarr[0][pg][i]+disp[pg][i],wfemarr[2][pg][i], 'b-', drawstyle='steps')
			if np.size(w[0]) != 0: ax.plot(wfemarr[0][pg][i][w],wfemarr[3][pg][i][w], 'r-')
#			if argx == 2: # For velocity:
#				wmin=np.min([mmpltx[0],1.2*np.min(wfemarr[0][pg][i][w])])
#				wmax=np.max([mmpltx[1],1.2*np.max(wfemarr[0][pg][i][w])])
#			elif argx == 1: # For rest wave:
#				xfacm = elnw[1][pg][i]*(1.0+mmpltx/299792.458)
#				wmin=np.min([xfacm[0],1.2*np.min(wfemarr[0][pg][i][w])-0.2*elnw[1][pg][i]])
#				wmax=np.max([xfacm[1],1.2*np.min(wfemarr[0][pg][i][w])-0.2*elnw[1][pg][i]])
#			else: # For observed wave:
#				xfacm = (1.0+rdshft)*elnw[1][pg][i]*(1.0 + mmpltx/299792.458)
#			wmin=np.min([xfacm[0],1.2*np.min(wfemarr[0][pg][i][w])-0.2*(1.0+rdshft)*elnw[1][pg][i]])
#			wmax=np.max([xfacm[1],1.2*np.max(wfemarr[0][pg][i][w])-0.2*(1.0+rdshft)*elnw[1][pg][i]])
			if np.size(w[0]) != 0:
				wmin=1.3*np.min(wfemarr[0][pg][i][w])-0.3*np.mean(wfemarr[0][pg][i][w])
				wmax=1.3*np.max(wfemarr[0][pg][i][w])-0.3*np.mean(wfemarr[0][pg][i][w])
			else:
				msgs.warn("No model to plot for page {0:d} panel {1:d}".format(pg+1+numpages,i+1)+msgs.newline()+"Check the fitrange for this parameter?", verbose=verbose)
				wmin=np.min(wfemarr[0][pg][i])
				wmax=np.max(wfemarr[0][pg][i])
#			for j in range(0,len(comparr[pg][i])/3):
#				if comparr[pg][i][3*j] == '0': cstr = 'k-'
#				else: cstr = 'r-'
#				if argx == 2: # For velocity:
#					xfact = float(comparr[pg][i][3*j+1])
#				elif argx == 1: # For rest wave:
#					xfact = elnw[1][pg][i]*(1.0+float(comparr[pg][i][3*j+1])/299792.458)
#				else: # For observed wave:
#					xfact = (1.0+rdshft)*elnw[1][pg][i]*(1.0+float(comparr[pg][i][3*j+1])/299792.458)
#				ax.plot([xfact,xfact],[1.05,1.15], cstr)
#				if flags['labels']: ax.text(xfact,1.2,comparr[pg][i][3*j+2],horizontalalignment='center',rotation='vertical',clip_on=True)
#			if not argflag['plot']['labels']: ax.text(wmin+0.09*(wmax-wmin),0.2,elnw[0][pg][i]+' %5.1f' % (elnw[1][pg][i]))
			ax.set_xlim(wmin,wmax)
			ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
			ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
#			if argx == 2: ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%5.1f'))
#			else: ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%6.2f'))
			ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%6.2f'))
			flue_med = 2.0*np.median(wfemarr[2][pg][i])
			modl_max = np.max(wfemarr[3][pg][i])
#			if argflag['plot']['labels']: ymax = np.max([1.0+2.0*flue_med, 2.0])
#			else: ymax = np.max([1.0+2.0*flue_med, 1.2])
			ymax = np.max([modl_max+flue_med, 1.2*np.max(wfemarr[3][pg][i])])
			ax.set_ylim(-flue_med,ymax)
			#ax.set_yticks((0,0.5,1.0))
		pgnum += 1
	return pgnum

def plot_showall():
	plt.show()

def prep_arrs(snip_ions, snip_detl, posnfit, verbose=2):
	"""
	Not presently used in ALIS
	"""
	elnames = np.array([])
	elwaves = np.array([])
	comparr = []
	rdshft=0.0
	max_CD = 0.0
	testrds=1.0
	for sn in range(0,len(slf._snip_ions)):
		comparr.append([])
		max_elm = None
		max_col = 0.0
		max_wav = 0.0
		max_fvl = 0.0
		for ln in range(0,len(slf._snip_ions[sn])):
			wavl = (1.0+slf._snip_detl[sn][ln][3])*slf._snip_detl[sn][ln][0]
			if wavl >= slf._posnfit[2*sn] and wavl <= slf._posnfit[2*sn+1]:
				if slf._snip_detl[sn][ln][1] > max_col:
					max_elm = slf._snip_ions[sn][ln]
					if slf._snip_detl[sn][ln][2] > max_fvl:
						max_wav = slf._snip_detl[sn][ln][0]
						max_fvl = slf._snip_detl[sn][ln][2]
					max_col = slf._snip_detl[sn][ln][1]
					tmp_rds = slf._snip_detl[sn][ln][3]
		if max_elm is None:
			max_elm = "None"
			max_wav = 0.5*(slf._posnfit[2*sn] + slf._posnfit[2*sn+1])
		elnames = np.append(elnames, max_elm)
		elwaves = np.append(elwaves, max_wav)
		if max_col > max_CD:
			max_CD = max_col
			rdshft = tmp_rds
	testrds=rdshft
	tri = np.where(elnames == "None")
	if np.size(tri) != 0: elwaves[tri] /= (1.0+testrds)
	if rdshft == 0.0:
		msgs.warn("Couldn't find the redshift of the main component for plotting", verbose=verbose)
		msgs.info("Assuming z=0", verbose=verbose)
	for sn in range(0,len(slf._snip_ions)):
		for ln in range(0,len(slf._snip_ions[sn])):
			compvel = 299792.458 * (slf._snip_detl[sn][ln][0]*(1.0+slf._snip_detl[sn][ln][3])/(elwaves[sn]*(1.0+rdshft)) - 1.0)
			if slf._snip_ions[sn][ln] == elnames[sn]: comparr[sn].append('1')
			else: comparr[sn].append('0')
			elnameID = '%s %5.1f' % (slf._snip_ions[sn][ln],slf._snip_detl[sn][ln][0])
			comparr[sn].append( '%8.3f' % (compvel) )
			comparr[sn].append( '%s' % (elnameID) )
	return elnames, elwaves, rdshft, comparr

