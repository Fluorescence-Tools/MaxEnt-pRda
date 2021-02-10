#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import sys
import traceback
import collections
from functools import partial
from glob import glob
from multiprocessing import Pool, current_process, cpu_count
from datetime import timedelta
import time

import numpy as np
from numpy.random import default_rng
from numpy.lib import recfunctions as rfn
from scipy.stats import norm
from scipy.interpolate import interp1d
from scipy.special import erf
from scipy.optimize import basinhopping, minimize
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
#np.seterr(all='raise')

def main(nSteps, saveDirPath, theta, seed, method):
    # Input parameters:
    expPathMask = 'Lif_data/ucfret_20201210/*.txt'
    mdDataDir = 'Lif_data/MD_Milana_Lif_ff99sb-disp'
    sigma = np.sqrt(2.0*6.0**2) #Angstrom
    rda_min = 0.0
    rda_max = 200.0
    saveStride = min(int(nSteps/10)+1, 100)
    minErrFrac = 0.0005 # minimal allowed error for a bin
    
    # Load MD data
    rmpModel = np.genfromtxt(mdDataDir+'/Rmp.dat',names=True,delimiter='\t',deletechars="")
    w0 = np.loadtxt(mdDataDir+'/cluster_weights.dat',delimiter=' ',usecols=[1]) #reference weights
    w0 /= w0.sum()
    wseed = None
    if seed == 'random':
        wseed = default_rng().uniform(0,1,w0.shape[0])
        wseed /= wseed.sum()
    elif seed == 'reference':
        wseed = np.copy(w0)
    else:
        wseed = np.loadtxt(seed+'/weights_final.dat',usecols=[0])
        assert wseed.shape == w0.shape
    assert w0.shape[0] == rmpModel.shape[0]
    
    # Format experimental data
    # Creates list of strings, which are FRET pairs, i.e. ['137_215', '137_268',...].
    # They are taken from the header of 'RDAMean.dat'
    pairsStr = list(rmpModel.dtype.names[1:]) 
    # Appends distances of all FRET pairs into single numpy array of dimension N_clusters x Npairs
    rmpEns = rfn.structured_to_unstructured(rmpModel[pairsStr])
    # Replaces NaN values, if such exist, with -1, hence no need to clean the data from NaN occurences
    # if numpy.version.version<1.17 numpy.nan_to_num does not work -> update numpy
    rmpEns = np.nan_to_num(rmpEns, nan=-100.0)
    bin_edges, pExp, errExp = loadExpData(expPathMask, rda_min, rda_max, pairsStr)
    if np.any(errExp<minErrFrac):
        print(f'WARNING: some experimental errors are below the threshold ({minErrFrac:.1e}), clipping!')
    errExp = np.clip(errExp, minErrFrac, None)
    
    # Setup minimazion logging
    rmp2pR = distHist2D(rmpEns, sigma, bin_edges)
    resiPart = partial(memResi, pExp=pExp, errExp=errExp, 
                    rmp2pR=rmp2pR, w0=w0, theta=theta)
    fG = lambda resi: np.square(resi).sum()
    
    numResi = resiPart(wseed).size
    trajLen = int(nSteps / saveStride) + 1
    wTraj = np.zeros((trajLen,wseed.shape[0]))
    resiTraj = np.zeros((trajLen,numResi))
    gTraj = np.full(trajLen,np.inf)
    disableTbar = len(current_process()._identity)>0
    tbar = trange(nSteps,disable=disableTbar)
    curIt = 0
    def resiTrace(w):
        nonlocal curIt #nonlocal works only in python3 and later
        if curIt > nSteps:
            raise StopIteration("Maximum number of iterations was reached")
        resi = resiPart(w)
        G = fG(resi)
        iRound = int(curIt/saveStride)
        if G < gTraj[iRound]:
            gTraj[iRound] = G
            resiTraj[iRound] = resi
            wTraj[iRound] = w
            tbar.set_description(f'{method}, G={G:.2f}')
        tbar.update()
        curIt += 1
        return resi
    
    #minimize
    start_time = time.monotonic()
    if method == "MH":
        minimizeMH(resiTrace, nSteps, wseed)
    elif method[:6] == "Lmfit_":
        minimizeLmfit(resiTrace, nSteps, wseed, method[6:])
    else:
        minimizeScipy(resiTrace, nSteps, wseed, method)
        

    run_duration = timedelta(seconds=int(time.monotonic() - start_time))
    tbar.close()
    
    gTraj[-1] = np.inf
    lastIt = np.argmax(gTraj==np.inf)
    gTraj = gTraj[:lastIt]
    wTraj = wTraj[:lastIt]
    resiTraj = resiTraj[:lastIt]
    
    # Find the best & first good solutions
    iBest = gTraj.argmin()
    Gmin, chi2r, S, resW = weigths2GTerms(wTraj[iBest], w0, pExp, errExp, rmp2pR, theta)
    iGood = np.argmax(gTraj<Gmin*1.001)
    gGood = gTraj[iGood]
    
    # Save results
    resDict={"G_min": Gmin,
            "chi2r": chi2r,
            "S": S,
            "w_tot": wTraj[iBest].sum(),
            "theta":theta,
            "minimizer":method,
            "runtime":str(run_duration),
            "it_Gmin": iBest*saveStride,
            "G_good": gGood,
            "it_good": iGood*saveStride,
            "saveDirPath": saveDirPath }
    
    if not os.path.exists(saveDirPath): os.mkdir(saveDirPath)
    np.savetxt(saveDirPath+'/residuals_traj.dat', resiTraj, fmt='%.6e',delimiter='\t')
    np.savetxt(saveDirPath+'/weights_traj.dat',wTraj, fmt='%.6e')
    np.savetxt(saveDirPath+'/weights_final.dat', wTraj[iBest], fmt='%.6e',header=str(resDict))
    
    edges_full = np.linspace(0.0,200.0,201)
    pModel = distHist2D(rmpEns, sigma, edges_full)(wTraj[iBest])
    for ip, pair in enumerate(pairsStr):
        pModData = np.column_stack(((edges_full[:-1]+edges_full[1:])*0.5,pModel[:,ip]))
        np.savetxt(saveDirPath+f'/pRda_model_{pair}.dat', pModData, 
                   fmt=['%.2f','%.5e'], header='Rda\tp(Rda)',delimiter='\t',comments='')
    
    plot_energies(saveDirPath+f'/energies.png', resiTraj, theta, saveStride)
    gCur = gTraj[0]*2.0
    for i in range(iBest,iBest+1):
        if gTraj[i] >= gCur:
            continue
        gCur = gTraj[i]
        w = wTraj[i]
        pModel = rmp2pR(w)
        pModIni = rmp2pR(w0)
        for ip, pair in enumerate(pairsStr):
            pRmp = np.histogram(np.nan_to_num(rmpModel[pair],nan=-1.0),bins=bin_edges, weights=w)[0]
            plot_pRda(saveDirPath+f'/{pair}_{i*saveStride:05}.png', bin_edges, pModel[:,ip], pExp[:,ip], errExp[:,ip], pModIni[:,ip], pRmp)
       
    
    if theta>0.0:
        assert np.all(np.isclose([Gmin, chi2r, S, resW], resi2GTerms(resiTraj[iBest],theta)))

    return resDict

def chi2Resi(pModel, pExp, errExp):
    # Calculate optimal pair-specific model to experiment scaling factor
    # scaling factor is obtained via minimization of chi2: d(chi2)/d(modScale)=0
    # add smallest representable float to denominator to avoid division with 0
    modScale = (pExp*pModel/np.square(errExp)).sum(axis=0) / (np.square(pModel/errExp).sum(axis=0)+np.finfo(float).eps) 
    modScale = np.atleast_2d(modScale)
    return (pExp - pModel*modScale)/errExp # =resi of a size [Nbins,Npairs]

def memResi(w, pExp, errExp, rmp2pR, w0, theta):   
    # Here 0*log(0)=0, since lim{x*log(x), x->0} = 0
    # So the smallest representable float (2.220446049250313e-16) is added
    S = np.sum(w * np.log(w / w0 + np.finfo(float).eps)) 
    # the wheights should approximately add up to 1.0
    # resw term penalizes the deviation from 1.0; 0.1 is the force constant
    resw = (w.sum() - 1.0) / 0.1 
    pModel = rmp2pR(w)
    resi = chi2Resi(pModel, pExp, errExp)
    # Here sum(resi**2) ~= chi2r, not chi2, therefore normalisation
    resi *= 1.0 / np.sqrt(2.0*resi.size) #resi.size=Nbins*Npairs
    # here sqrt(S+1)*theta because later squares this whole array
    resi = np.append(resi, [np.sqrt((S+1.0)*theta), resw])
    return resi

def minimizeScipy(residualsFn, nSteps, wseed, minimizer, forceFakeBounds=True):
    """ Minimize using basin hopping from SciPy """
    bound = ['L-BFGS-B', 'Powell', 'TNC', 'trust-constr']
    fG_base = lambda w: np.square(residualsFn(w)).sum()
    def fG_bounds(w):
        #The formula is taken from: https://lmfit.github.io/lmfit-py/bounds.html
        wb = np.sqrt(np.square(w)+1.0)-1.0
        return fG_base(wb)
    
    wseed_b=None
    minimizer_kwargs = {"method":minimizer, 
                        "options":{"maxiter":nSteps}}
    if forceFakeBounds==False and minimizer in bound:
        minimizer_kwargs["bounds"]=[(0,1)]*len(wseed)
        fG=fG_base
        wseed_b=wseed
    else:
        fG=fG_bounds
        wseed_b=np.sqrt((wseed+1)**2-1)
    if minimizer=='L-BFGS-B':
        minimizer_kwargs["options"]["maxfun"]=nSteps
        
    try:
        res=basinhopping(fG, wseed_b, minimizer_kwargs=minimizer_kwargs)
        return np.array(res.x)
    except StopIteration as e:
        return None

def minimizeMH(residualsFn, nSteps, wseed):
    nBuf = 100
    nConv = 2000 # rescramble if the solution does not improve nConv iterations
    convTol = 1E-6
    nMoves = 2
    
    assert nConv >= nBuf
    
    rng = default_rng()
    fG = lambda resi: np.square(resi).sum()
    resi = residualsFn(wseed)
    wTraj = np.empty((nSteps,wseed.shape[0]))
    resiTraj = np.empty((nSteps,resi.size))
    gTraj = np.empty(nSteps)
    accBuf = collections.deque(maxlen=nBuf)
    
    kT = 0.1
    w_step = 2.0
    # Sample solutions
    w = np.copy(wseed)
    accept_ratio = -1.0
    Gprev = fG(resi)
    iAcc = 0
    for i in range(nSteps):
        #adjust sampling
        if i%nBuf == (nBuf-1):
            accept_ratio = sum(accBuf) / nBuf
            gBuf=gTraj[:i][-nBuf:]
            iC = int(nBuf/2)
            Gmin_old = gBuf[:iC].min()
            Gmin_new = gBuf[iC:].min()
            kT = max((Gmin_old-Gmin_new)/iC, np.finfo(float).eps)
            w_step = max(w_step*np.sqrt(0.7 + accept_ratio), 1.001)
        
        # Generate new weights
        w_old = np.copy(w)
        w[rng.integers(0,w.shape[0],nMoves)] *= rng.uniform(1.0/w_step,w_step,nMoves)
        w/=w.sum()
        
        accept = False
        # Rescramble if the solution does not improve
        if i%nConv==(nConv-1):
            iC = int(i-nConv/2)
            g_old = gTraj[i-nConv+1:iC].min()
            g_new = gTraj[iC:i].min()
            if g_old/g_new < 1.0+convTol:
                w = rng.uniform(0, 1, w.shape[0])
                w /= w.sum()
                Gprev = np.inf
        
        # Test new weights
        resi = residualsFn(w)
        G = fG(resi)
        gTraj[i]=G
        if G<=Gprev:
            accept = True
        else:
            gRel = (Gprev-G)/kT
            if gRel > np.log(1E-12): # prevents exp underflow
                r = rng.uniform(0,1)
                accept = r < np.exp(gRel)
        if accept:
            accBuf.append(1)
            wTraj[iAcc] = w
            resiTraj[iAcc] = resi
            iAcc += 1
            Gprev = G
        else:
            w = w_old
            accBuf.append(0)
    
    return wTraj[:iAcc], resiTraj[:iAcc]    

class distHist2D(object):
    """Functor-class that calculates kernel-smoothed distributions, given the cdf of the kernel. 
    If no kernel functions is provided, non-central chi distribution is used."""
    def __init__(self, rmps, sigma, bin_edges, cdfFn=None):
        numConf = rmps.shape[0]
        numBins = bin_edges.shape[0]-1
        numPairs = rmps.shape[1]
        if cdfFn is None:
            cdfFn = self.chi_cdf
        self.p = np.empty((numPairs, numBins, numConf))
        for pair in range(numPairs):
            for conf in range(numConf):
                cdf = cdfFn(bin_edges, rmps[conf,pair], sigma)
                self.p[pair, : ,conf] = cdf[1:] - cdf[:-1]
                
    def chi_cdf(self, x, mu, sigma):
        """
        Calculates the cdf for non-central chi distribution
        chi_cdf = integral( x/mu*(norm.pdf(x, mu, sigma) - norm.pdf(x, -mu, sigma)), x=0..x )
        """
        e1 = np.exp(-0.5*(mu+x)**2 / sigma**2)
        e2 = np.exp(-0.5*(mu-x)**2 / sigma**2)
        erf1 = erf((mu-x)/(np.sqrt(2.0)*sigma))
        erf2 = erf((mu+x)/(np.sqrt(2.0)*sigma))
        chi = sigma/(mu*np.sqrt(2.0*np.pi)) * (e1-e2) + 0.5*(erf2-erf1)
        return chi
    
    def __call__(self, weights):
        """Calculate kernel-smoothed p(Rda). numBins x numPairs"""
        assert weights.shape[0] == self.p.shape[2]
        r = np.sum(self.p*weights.reshape(1, 1, weights.shape[0]), axis=2).T
        return r
        #return r/r.sum(axis=0)

def loadExpData(pExpPathMask, rda_min, rda_max, pairsSorted):
    # glob() allows usage of Unix matching patterns, e.g. '*.txt'. 
    # Creates a list of strings: 'Lif_data/ucfret_20201210/137-251.txt' etc.
    pExpPathList = glob(pExpPathMask)
    numPairs = len(pExpPathList) #determines number of loaded files
    # Checks if experimental data is actually read, by checking if there are more than 0 of read files.
    assert numPairs > 0 
    
    # Load as structured data only the first file "137-215.txt", because at this moment we just need bins information
    data = np.genfromtxt(pExpPathList[0],names=True,delimiter='\t',deletechars="")
    bins_left = data['Rda_left']
    bins_right = data['Rda_right']
    # make sure that no bins were skipped and
    # ensures continuous binning(i.e. 0-10,10-20... and not 0-10, 20-30)
    assert np.all(np.isclose(bins_left[1:],bins_right[:-1]))
    # np.flatnonzero gives the indices of nonzero elements of bins_left array that are >= rda_min. 
    # Then take the first element
    binStart = np.flatnonzero(bins_left>=rda_min)[0] 
    # np.flatnonzero gives the indices of nonzero elements of bins_right array that are <= rda_max. 
    # Then take the last element's index+1
    binEnd = np.flatnonzero(bins_right<=rda_max)[-1]+1 
    numBins = binEnd - binStart
    
    pExp = np.full((numBins, numPairs),-1.0)
    err = np.full((numBins, numPairs),-1.0)
    for pExpPath in pExpPathList:
        # In a list of strings: 'Lif_data/ucfret_20201210/137-251.txt' remove '.txt', 
        # take the last string separated with /, replace - with _ -> '137_251'
        pair = pExpPath.replace('.txt','').split('/')[-1].replace('-','_') 
        # ensure that pair name from exp. file exists in the header of "RDAMean.dat"
        assert pair in pairsSorted
        # index of the given FRET pair in pairsSorted(pairsStr), i.e. in the simulation files
        iP = pairsSorted.index(pair) 
        data = np.genfromtxt(pExpPath,names=True,delimiter='\t',deletechars="")
        assert np.all(bins_left == data['Rda_left'])
        # sorts the pExp in the same order as in pairsSorted(pairsStr), given by iP
        pExp[:,iP] = data['prda'][binStart:binEnd] 
        err[:,iP] = data['prda_error'][binStart:binEnd]
    assert np.all(pExp>=0.0)
    assert np.all(err>0.0)
    
    edges = np.append(bins_left[binStart:binEnd],bins_right[binEnd-1])
    return edges, pExp, err

def plot_pRda(path, rda_edges, model, exp, err, model_initial, pRmp):
    modScale = (exp*model/np.square(err)).sum() / (np.square(model/err).sum()+np.finfo(float).eps)
    initScale = (exp*model_initial/np.square(err)).sum() / (np.square(model_initial/err).sum()+np.finfo(float).eps)
    
    rda = (rda_edges[:-1]+rda_edges[1:])*0.5
    bin_width = rda_edges[1:] - rda_edges[:-1]
    
    fig, ax = plt.subplots()
    ax.errorbar(rda,exp/bin_width,yerr=err/bin_width, color='black', label='experiment', fmt='o', markersize=2, elinewidth=1.0)
    ax.plot(rda,model*modScale/bin_width, 'bo--', label='MD+weights', markersize=2, linewidth=1)
    ax.plot(rda, model_initial*initScale/bin_width, 'ro--', label='MD initial', markersize=2, linewidth=1)
    
    axR = ax.twinx()
    axR.plot(rda, pRmp*modScale/bin_width, 'go--', label='p(Rmp), MD+reweigting', markersize=2, linewidth=1)
    axR.set_ylabel('p(Rmp) per A / A^-1', color='g')
    axR.set_ylim(0.0, None)
    
    ax.set_ylim(0.0, None)
    ax.set_xlabel('Rapp / A')
    ax.set_ylabel('p(Rda) per A / A^-1')
    chi2r=np.square((model*modScale-exp)/err).mean()
    ax.set_title('FRET pair: ' + path.split('/')[-1] + f', chi2r = {chi2r:.1f}')
    fig.legend(loc='upper center', ncol=4, prop={'size': 8}, bbox_to_anchor=(0.5, 0.06))
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.15)
    fig.savefig(path,dpi=300)
    plt.close(fig)
    
def plot_energies(path, resiTraj, theta, saveStride):
    y = np.zeros((resiTraj.shape[0],4))
    for i, resi in enumerate(resiTraj): 
        y[i] = resi2GTerms(resi, theta)
    G, chi2r, S, resW = y.T
    
    fig, ax = plt.subplots()
    x = np.arange(resiTraj.shape[0])*saveStride
    ax.plot(x, G , 'r-', label='G', linewidth=1)
    ax.plot(x, chi2r , 'g-', label='chi2', linewidth=1)
    ax.plot(x, S, 'b-', label='S', linewidth=1)
    ax.set_ylim(0.0, chi2r[0]*1.1)
    ax.set_xlim(0, x[-1])
    ax.set_xlabel('# iteration')
    ax.set_ylabel('G, chi2r, S')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3, prop={'size': 8})
    
    fig.tight_layout()
    fig.savefig(path,dpi=300)
    plt.close(fig)
    
def weigths2GTerms(w, w0, pExp, errExp, rmp2pR, theta):
    chi2r = np.square(chi2Resi(rmp2pR(w), pExp, errExp)).mean()
    S = np.sum(w*np.log(w/w0+np.finfo(float).eps))
    resW = (w.sum() - 1.0) / 0.1
    G = chi2r * 0.5 + theta * (S + 1.0) + resW**2
    return G, chi2r, S, resW

def resi2GTerms(resi, theta):
    chi2r = np.square(resi[:-2]).sum()*2.0
    theta += np.finfo(float).eps
    S = resi[-2]**2/theta-1
    resW = resi[-1]
    G = np.square(resi).sum()
    return G, chi2r, S, resW

def mainWrap(d):
    try:
        return main(**d)
    except Exception as e:
        print(e)
        print(traceback.format_exc())
        return {"saveDirPath": d["saveDirPath"]}

def dictlist2arr(dicts):
    dt_tuples = []
    for key, value in dicts[0].items():
        if not isinstance(value, str):
            value_dtype = np.array([value]).dtype
        else:
            value_dtype = '|U{}'.format(max([len(d.get(key,"")) for d in dicts]))
        dt_tuples.append((key, value_dtype))
    dt = np.dtype(dt_tuples)
    
    values = [tuple(d.get(name,None) for name in dt.names) for d in dicts]
    
    return np.array(values, dtype=dt)


if __name__ == '__main__':
    if len(sys.argv)<4:
        print('Usage:  \t./optimize_lif_mem.py <nSteps> <outDir> <theta>\nExample:\t./optimize_lif_mem.py 50000 results 0.15')
        sys.exit()
    
    nProcesses = 1
    nSteps=int(sys.argv[1])
    saveDirPath=sys.argv[2]
    thetaList = [float(t) for t in sys.argv[3].split(',')]
    seed = 'reference' #'random', 'reference', '<some_path>'
    if len(sys.argv) > 4:
        seed = sys.argv[4]

    methods = ['MH', 'L-BFGS-B', 'BFGS', 'trust-constr', 'CG', 'SLSQP', 'Powell']
    # These methods converge the fastest: 'MH', 'L-BFGS-B', 'BFGS'
    # 'trust-constr' and 'CG' are OK
    # 'SLSQP' optimizes very well, but each iteration takes more time if the ensemble is large
    # Powell takes more iterations to converge
    # These require a jacobian: 'Newton-CG', 'dogleg', 'trust-ncg','trust-exact', 'trust-krylov'
    
    results = []
    if nProcesses == 1:
        d = main(nSteps, saveDirPath, thetaList[0], seed, methods[0])
        results.append(d)
    else:
        pool = Pool(nProcesses)
        if not os.path.exists(saveDirPath): os.mkdir(saveDirPath)
        dictLst = []
        for method in methods:
            for theta in thetaList:
                for cur_seed in [seed, 'random', 'random']:
                    d={'nSteps':nSteps, 'theta':theta, 'seed':cur_seed, 
                    'method':method, 'saveDirPath':f'{saveDirPath}/th{theta:.3f}_{method}_{cur_seed}_{len(dictLst)}' }
                    dictLst.append(d)
        with trange(len(dictLst)) as tbar:
            for d in pool.imap_unordered(mainWrap, dictLst):
                results.append(d)
                tbar.set_description(f'{d.get("saveDirPath","")}, G={d.get("G_min",-1.0):.2f}')
                tbar.update()
    
    restab=dictlist2arr(results)
    header='\t'.join(restab.dtype.names)
    fmtMap = {np.int32:"%d", np.float64:"%.2f"}
    fmt=[fmtMap.get(np.dtype(t).type,'%s') for n, t in restab.dtype.descr]
    for out in [f'{saveDirPath}/summary.dat', sys.stdout]:
        np.savetxt(out, restab, fmt=fmt, delimiter='\t', header=header, comments='')

