#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import sys
import collections
from functools import partial
from glob import glob
from multiprocessing import Pool, current_process, cpu_count
import time
import numpy as np
from numpy.random import default_rng
from numpy.lib import recfunctions as rfn
from scipy.stats import norm
from scipy.interpolate import interp1d
from scipy.special import erf
from scipy.optimize import basinhopping
import matplotlib.pyplot as plt
import lmfit
from tqdm import trange, tqdm
#np.seterr(all='raise')

def main(nSteps, saveDirPath, theta0, seed, method='BH-SLSQP'):
    #input parameters
    sigma = 6.0 #Angstrom
    rda_min = 0.0
    rda_max = 200.0
    saveStride = 100
    minErrFrac = 0.001 #per bin
    expPathMask = 'Lif_data/ucfret_20201210/*.txt'
    rdaModel = np.genfromtxt('Lif_data/545_pdbs_Rda.dat',names=True,delimiter='\t',deletechars="")
    rmpModel = np.genfromtxt('Lif_data/545_pdbs_Rmp.dat',names=True,delimiter='\t',deletechars="")
    w0 = np.loadtxt('Lif_data/545_pdbs_initial_weights.dat',delimiter=' ',usecols=[1]) #weights
    w0 /= w0.sum()
    wseed = np.copy(w0)
    if seed == False:
        wseed = default_rng().uniform(0,1,w0.shape[0])
        wseed /= wseed.sum()
    elif seed is not None:
        wseed = np.loadtxt(seed+'/weights_final.dat',usecols=[0])
        assert wseed.shape == w0.shape
    assert w0.shape[0] == rdaModel.shape[0]
    
    #format experimental data
    pairsStr = list(rdaModel.dtype.names[1:])
    rdaEns = rfn.structured_to_unstructured(rdaModel[pairsStr])
    rdaEns = np.nan_to_num(rdaEns, nan=-1.0)
    bin_edges, pExp, errExp = loadExpData(expPathMask, rda_min, rda_max, pairsStr)
    errExp = np.clip(errExp,minErrFrac/errExp.shape[0],None)
       
    #setup minimazion logging
    resiPart = partial(memResi, pExp=pExp, errExp=errExp, 
                    rdaEns=rdaEns, sigma=sigma, 
                    bin_edges=bin_edges, w0=w0, theta=theta0)
    fG = lambda resi: np.square(resi).sum()
    
    numResi = resiPart(wseed).size
    wTraj = np.zeros((nSteps,wseed.shape[0]))
    resiTraj = np.zeros((nSteps,numResi))
    disableTbar = len(current_process()._identity)>0
    tbar = trange(nSteps,disable=disableTbar)
    curIt = 0
    def resiTrace(w):
        nonlocal curIt
        resi = resiPart(w)
        if curIt<nSteps:
            resiTraj[curIt] = resi
            wTraj[curIt] = w
        tbar.update()
        tbar.set_description(f'{method}, G={fG(resi):.2f}')
        curIt += 1
        return resi
    
    #minimize
    if method == "MH":
        minimizeMH(resiTrace, nSteps, wseed)
    elif method[:3] == "BH-":
        minimizeScipyBH(resiTrace, nSteps, wseed, method[3:])
    else:
        minimizeLmfit(resiTrace, nSteps, wseed, method)

    tbar.close()
    
    if curIt<nSteps:
        wTraj.resize((curIt,wTraj.shape[1]))
        resiTraj.resize((curIt,resiTraj.shape[1]))
    
    #find the first best & good solutions
    gTraj = np.square(resiTraj).sum(axis=1)
    iBest = gTraj.argmin()
    gBest = gTraj[iBest]
    iGood = np.argmax(gTraj<gBest*1.001)
    gGood = gTraj[iGood]
    
    #save results
    G, chi2r, S, resW = weigths2GTerms(wTraj[iBest], w0, pExp, errExp, rdaEns,  sigma, bin_edges)
    summary=f'chi2r={chi2r:.3f}, S={S:.2f}, w.sum()={wTraj[iBest].sum():.3f}, G={G:.2f}, seed={seed}, theta={theta0}, nSteps={nSteps}, {method}'
    if not os.path.exists(saveDirPath): os.mkdir(saveDirPath)
    np.savetxt(saveDirPath+'/residuals_traj.dat', resiTraj[::saveStride], fmt='%.6e',delimiter='\t')
    np.savetxt(saveDirPath+'/weights_traj.dat',wTraj[::saveStride], fmt='%.6e')
    np.savetxt(saveDirPath+'/weights_final.dat', wTraj[iBest], fmt='%.6e',header=summary)
    
    edges_full = np.linspace(0.0,200.0,201)
    pModel = gaussHistogram2D(rdaEns, wTraj[iBest], sigma, edges_full)
    for ip, pair in enumerate(pairsStr):
        pModData = np.column_stack(((edges_full[:-1]+edges_full[1:])*0.5,pModel[:,ip]))
        np.savetxt(saveDirPath+f'/pRda_model_{pair}.dat', pModData, fmt=['%.2f','%.5e'], header='Rda\tp(Rda)',delimiter='\t',comments='')
    
    plot_energies(saveDirPath+f'/energies_{method}', resiTraj, theta0)
    gCur = gTraj[0]*2.0
    for i in range(iBest,iBest+1,100):
        if gTraj[i] >= gCur:
            continue
        gCur = gTraj[i]
        w = wTraj[i]
        pModel = gaussHistogram2D(rdaEns, w, sigma, bin_edges)
        for ip, pair in enumerate(pairsStr):
            pModIni=gaussHistogram2D(rdaEns[:,ip:ip+1], w0, sigma, bin_edges)[:,0]            
            pRmp = np.histogram(np.nan_to_num(rmpModel[pair],nan=-1.0),bins=bin_edges, weights=w)[0]
            plot_pRda(saveDirPath+f'/{pair}_{i:05}', bin_edges, pModel[:,ip], pExp[:,ip], errExp[:,ip], pModIni, pRmp)
       
    
    if disableTbar is False:
        print(summary)
    if theta0>0.0:
        assert np.all(np.isclose([G, chi2r, S, resW], resi2GTerms(resiTraj[iBest],theta0)))
    
    return gBest, iBest, chi2r, S, gGood, iGood, saveDirPath

def residuals(weights, pExp, errExp, rdaEns, sigma, bin_edges):
    pModel = gaussHistogram2D(rdaEns, weights, sigma, bin_edges)
    #calculate optimal pair-specific model to experiment scaling factor
    modScale = (pExp*pModel/np.square(errExp)).sum(axis=0) / (np.square(pModel/errExp).sum(axis=0)+np.finfo(float).eps) 
    modScale = np.atleast_2d(modScale)
    return (pExp - pModel*modScale)/errExp

def memResi(w, pExp, errExp, rdaEns, sigma, bin_edges, w0, theta):   
    #Here 0*log(0)=0, since lim{x*log(x), x->0} = 0
    S = np.sum(w * np.log(w / w0 + np.finfo(float).eps))
    #the wheights should approximately add up to 1.0
    resw = (w.sum() - 1.0) / 0.1 
    resi = residuals(w, pExp, errExp, rdaEns,  sigma, bin_edges)
    #Here sum(resi**2) ~= chi2r, not chi2, therefore normalisation
    resi *= 1.0 / np.sqrt(2.0*resi.size)
    resi = np.append(resi, [np.sqrt((S+1)*theta), resw])
    return resi

def minimizeLmfit(residualsFn, nSteps, wseed, method):
    params2np = lambda pars: np.array([pars[name] for name in pars])
    resiLmfit = lambda w: residualsFn(params2np(w))

    par = lmfit.Parameters()
    for i in range(len(wseed)):
        par.add(f'v{i}', wseed[i], min=0.0, max=1.0)
    res = lmfit.minimize(resiLmfit, par, method=method, max_nfev=nSteps)
    assert len(res.params) == len(wseed)
    #print(lmfit.fit_report(res))
    
    return np.array([res.params[name] for name in res.params])

def minimizeScipyBH(residualsFn, nSteps, wseed, minimizer):
    fG = lambda w: np.square(residualsFn(w)).sum()
    Gseed = fG(wseed)
    itCount = 0
    def Gwrap(w):
        nonlocal itCount
        itCount+=1
        if itCount>=nSteps:
            return Gseed
        return fG(w)
    minimizer_kwargs = {"method":minimizer, "bounds":[(0,1)]*len(wseed)}
    stop = lambda x,f,a: itCount>nSteps
    res=basinhopping(Gwrap, wseed, minimizer_kwargs=minimizer_kwargs, callback=stop)
    return np.array(res.x)

def minimizeMH(residualsFn, nSteps, wseed):
    kT = 1.0
    nMoves = 1
    w_step = 1.1
    nBuf = 100
    nConv = 2000 #rescramble if the solution does not improve nConv iterations
    convTol = 1E-8
    
    assert nConv >= nBuf
    
    rng = default_rng()
    fG = lambda resi: np.square(resi).sum()
    resi = residualsFn(wseed)
    wTraj = np.empty((nSteps,wseed.shape[0]))
    resiTraj = np.empty((nSteps,resi.size))
    gTraj = np.empty(nSteps)
    accBuf = collections.deque(maxlen=nBuf)
    
    #sample solutions
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
            Gave_old = gBuf[:iC].min()
            Gave_new = gBuf[iC:].min()
            kT = max((Gave_old-Gave_new)/nBuf, 1E-12)
            if  Gave_old <= Gave_new or accept_ratio < 0.2:
                #reduce the step size
                nMoves = max(nMoves-1,1)
                w_step = (1.01+w_step*3.0)/4.0
            elif accept_ratio > 0.5:
                #increase the step size
                nMoves += 1
                w_step = (3.0+w_step*3.0)/4.0
        
        #generate new weights
        w_old = np.copy(w)
        for iRS in range(nMoves):
            w[rng.integers(0,w.shape[0])] *= rng.uniform(1.0/w_step,w_step)
        
        accept = False
        #rescramble if the solution does not improve
        if i%nConv==(nConv-1):
            iC = int(i-nConv/2)
            g_old = gTraj[i-nConv+1:iC].min()
            g_new = gTraj[iC:i].min()
            if g_old/g_new < (1.0+convTol)**(nConv*0.5):
                w = rng.uniform(0, 1, w.shape[0])
                w /= w.sum()
                Gprev = np.inf
        
        #test new weights
        resi = residualsFn(w)
        G = fG(resi)
        gTraj[i]=G
        if G<=Gprev:
            accept = True
        else:
            gRel = (Gprev-G)/kT
            if gRel > np.log(1E-12): #prevents exp underflow
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
  
def gaussHistogram2D(rdaEns, weights, sigma, bin_edges, interp=False):
    #calculate gaussian-smoothed p(Rda).
    cdfInt = lambda x: norm.cdf(x,scale=sigma)
    if interp:
        #Use interpolated normal distribution to improve performance
        r =  np.linspace(-7.0*sigma, 7.0*sigma, 2001)
        cdfInt = interp1d(r, norm.cdf(r, scale=sigma), kind='linear', fill_value=(0.0,1.0), bounds_error=False, assume_sorted=True)

    numBins = bin_edges.shape[0]-1
    numPairs = rdaEns.shape[1]
    hist2D = np.full((numBins, numPairs),0.0)
    for ip in range(numPairs):
        dr = np.tile(bin_edges,(rdaEns.shape[0],1)) - rdaEns[:,ip:ip+1]
        cdf = cdfInt(dr)
        hist2D[:,ip] = ((cdf[:,1:]-cdf[:,:-1])*np.atleast_2d(weights).T).sum(axis=0)
    return hist2D

def loadExpData(pExpPathMask, rda_min, rda_max, pairsSorted):
    pExpPathList = glob(pExpPathMask)
    numPairs = len(pExpPathList)
    assert numPairs > 0
    
    data = np.genfromtxt(pExpPathList[0],names=True,delimiter='\t',deletechars="")
    bins_left = data['Rda_left']
    bins_right = data['Rda_right']
    assert np.all(np.isclose(bins_left[1:],bins_right[:-1]))
    binStart = np.flatnonzero(bins_left>=rda_min)[0]
    binEnd = np.flatnonzero(bins_right<=rda_max)[-1]+1
    numBins = binEnd - binStart
    
    pExp = np.full((numBins, numPairs),-1.0)
    err = np.full((numBins, numPairs),-1.0)
    for pExpPath in pExpPathList:
        pair = pExpPath.replace('.txt','').split('/')[-1].replace('-','_')
        assert pair in pairsSorted
        iP = pairsSorted.index(pair)
        data = np.genfromtxt(pExpPath,names=True,delimiter='\t',deletechars="")
        assert np.all(bins_left == data['Rda_left'])
        pExp[:,iP] = data['prda'][binStart:binEnd]
        err[:,iP] = data['prda_error'][binStart:binEnd]
    err=np.clip(err, np.finfo(float).eps, None)
    assert np.all(pExp>=0.0)
    assert np.all(err>=0.0)
    
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
    ax.set_xlabel('Rda / A')
    ax.set_ylabel('p(Rda) per A / A^-1')
    chi2r=np.square((model*modScale-exp)/err).mean()
    ax.set_title('FRET pair: ' + path.split('/')[-1] + f', chi2r = {chi2r:.1f}')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=4, prop={'size': 8})
    fig.tight_layout()
    fig.savefig(path+'.png',dpi=300)
    plt.close(fig)
    
def plot_energies(path, resiTraj, theta):
    y = np.zeros((resiTraj.shape[0],4))
    for i, resi in enumerate(resiTraj): 
        y[i] = resi2GTerms(resi, theta)
    G, chi2r, S, resW = y.T
    
    fig, ax = plt.subplots()
    x = np.arange(resiTraj.shape[0])
    ax.plot(x, G , 'r-', label='G', linewidth=1)
    ax.plot(x, chi2r , 'g-', label='chi2', linewidth=1)
    ax.plot(x, S, 'b-', label='S', linewidth=1)
    ax.set_ylim(0.0, chi2r[0]*1.1)
    ax.set_xlim(0, resiTraj.shape[0])
    ax.set_xlabel('# iteration')
    ax.set_ylabel('G, chi2r, S')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3, prop={'size': 8})
    
    fig.tight_layout()
    fig.savefig(path+'.png',dpi=300)
    plt.close(fig)
    
def weigths2GTerms(w, w0, pExp, errExp, rdaEns,  sigma, bin_edges):
    chi2r = np.square(residuals(w, pExp, errExp, rdaEns,  sigma, bin_edges)).mean()
    S = np.sum(w*np.log(w/w0+1E-12))
    resW = (w.sum()-1.0)/0.1
    G = chi2r*0.5 + theta0*(S+1.0) + resW**2
    return G, chi2r, S, resW

def resi2GTerms(resi, theta0):
    chi2r = np.square(resi[:-2]).sum()*2.0
    theta0 += np.finfo(float).eps
    S = resi[-2]**2/theta0-1
    resW = resi[-1]
    G = np.square(resi).sum()
    return G, chi2r, S, resW

def mainWrap(d):
    try:
        return main(**d)
    except:
        return -1, -1, -1, -1, -1, -1, d['saveDirPath']

if __name__ == '__main__':
    if len(sys.argv)<4:
        print('Usage:  \t./optimize_lif_mem.py <nSteps> <outDir> <theta>\nExample:\t./optimize_lif_mem.py 50000 results 0.15')
        sys.exit()
    
    serial = True
    nSteps=int(sys.argv[1])
    saveDirPath=sys.argv[2]
    theta0=float(sys.argv[3])
    seed = None
    if len(sys.argv) > 4:
        seed = sys.argv[4]

    methods=['BH-SLSQP', 'slsqp', 'basinhopping', 'trust-constr', 'cg', 'ampgo',  'dual_annealing', 'MH']
    #'differential_evolution', 'least_squares', 'shgo', 'BH-trust-constr', 'BH-L-BFGS-B'
    
    header = ['gBest', 'iBest', 'chi2r', 'S', 'gGood', 'iGood', 'saveDirPath']
    results = []
    
    if serial:
        r = main(nSteps, saveDirPath, theta0, seed, methods[1])
        results.append(r)
    else:
        numRuns=len(methods)
        pool = Pool()
        dictLst = []
        for i in range(numRuns):
            d={'nSteps':nSteps, 'theta0':theta0, 'seed':seed, 
               'method':methods[i], 'saveDirPath':f'{saveDirPath}_{i}' }
            dictLst.append(d)
        with trange(numRuns) as tbar:
            for r in pool.imap_unordered(mainWrap, dictLst):
                results.append(r)
                tbar.update()
                tbar.set_description(f'{r[-1]}, G={r[0]:.2f}')
    
    restab=np.core.records.fromrecords(results,formats=[float]*6+['U128'])
    header='\t'.join(header)
    np.savetxt(f'{saveDirPath}_summary.dat',restab, fmt='%s',delimiter='\t',header=header, comments='')
    print(header)
    for r in restab: 
        print(f'{r[0]:.2f}\t{r[1]:.0f}\t{r[2]:.2f}\t{r[3]:.2f}\t{r[4]:.2f}\t{r[5]:.0f}\t{r[6]}')
    
