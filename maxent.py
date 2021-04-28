#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import sys
from typing import NamedTuple
import traceback
import collections
import itertools
from functools import partial
from glob import glob
import argparse
from multiprocessing import Pool, current_process, cpu_count
from datetime import timedelta
import time
import numpy as np
from numpy.random import default_rng
from numpy.lib import recfunctions as rfn
from scipy.stats import norm
from scipy.interpolate import interp1d
from scipy.special import erf
from scipy import optimize
import matplotlib.pyplot as plt
from tqdm import trange, tqdm

# np.seterr(all='raise')


class MinimizerSettings(NamedTuple):
    nSteps: int
    minimizer: str = "L-BFGS-B"
    jac: bool = True
    fakeBounds: bool = True
    timeout: float = np.inf


def runMEM(
    savePath: str,
    expPattern: str,
    w0: np.ndarray,
    rmps: np.ndarray,
    seed: str,
    theta: float,
    sigma: float,
    minParams: MinimizerSettings,
):
    rda_min = 0.0
    rda_max = np.inf
    saveStride = min(int(minParams.nSteps / 100) + 1, 100)

    if seed == "reference":
        wseed = np.copy(w0)
    elif seed == "random":
        wseed = randomWeights(w0.shape[0])
    else:
        wseed = loadWeights(seed)
    assert w0.shape[0] == rmps.shape[0] == wseed.shape[0]
    bin_edges, pExp, errExp = loadExpData(expPattern, rda_min, rda_max, rmps.dtype.names)

    rmp2pR = distHist2D(rmps, sigma, bin_edges)
    energyFn = partial(memEnergy, pExp=pExp, errExp=errExp, rmp2pR=rmp2pR, w0=w0, theta=theta, jac=minParams.jac)
    gTraj, wTraj = minimize(energyFn, wseed, minParams, saveStride)

    # Identify the first best & first good solutions
    iBest = gTraj.argmin()
    wBest = wTraj[iBest]
    Gmin, chi2r, S, resW = energyTerms(wBest, w0, pExp, errExp, rmp2pR(wBest), theta)
    iGood = np.argmax(gTraj < Gmin * 1.001)
    gGood = gTraj[iGood]

    # Save results
    resDict = {
        "G_min": Gmin,
        "S": S,
        "chi2r": chi2r,
        "w_tot": wBest.sum(),
        "theta": theta,
        "seed": seed,
        "it_Gmin": iBest * saveStride,
        "G_good": gGood,
        "it_good": iGood * saveStride,
        "is_uncoverged": iGood * saveStride > minParams.nSteps / 2,
        "savePath": savePath,
    }
    resDict.update(minParams._asdict())

    if not os.path.exists(savePath):
        os.mkdir(savePath)
    weights_final = np.column_stack((np.arange(wseed.shape[0]), wBest))
    np.savetxt(savePath + "/weights_final.dat", weights_final, fmt="%d %.6e", header=str(resDict))

    save_pRda(savePath + f"/pRda_reweighted.dat", rmps, wBest, sigma)
    save_energies(savePath + f"/energies", wTraj, gTraj, w0, theta, saveStride)
    # save p(Rda) plots
    pModIni = rmp2pR(w0)
    pModel = rmp2pR(wBest)
    for ip, pair in enumerate(rmps.dtype.names):
        pRmp = np.histogram(rmps[pair], bins=bin_edges, weights=wBest)[0]
        path = savePath + f"/{pair}.png"
        chi2r_pair = np.square((pModel[:, ip] - pExp[:, ip]) / errExp[:, ip]).mean()
        tpair = pair.replace('_','\_')
        comm = f'pair: {pair}\n$\chi^2_{{r,{tpair}}}$={chi2r_pair:.1f}\n$\chi^2_{{r,tot}}$={chi2r:.1f}\nS={S:.1f}'
        plot_pRda(path, bin_edges, pModel[:, ip], pExp[:, ip], errExp[:, ip], pModIni[:, ip], pRmp, comm)
    
    return resDict


def memWrap(d):
    try:
        return runMEM(**d)
    except Exception as e:
        print(e)
        print(traceback.format_exc())
        return {"savePath": d["savePath"]}


def minimize(energyFn, wseed, minParams, stride):
    # Setup minimazion logging
    nSteps = minParams.nSteps
    trajLen = int(nSteps / stride) + 2
    wTraj = np.zeros((trajLen, wseed.shape[0]))
    gTraj = np.full(trajLen, np.inf)
    wTraj[0] = wseed
    gTraj[0] = energyFn(wseed)[0]
    disableTbar = len(current_process()._identity) > 0
    tbar = trange(nSteps, disable=disableTbar)
    curIt = 0
    start_time = time.monotonic()

    def traceEnergy(w):
        nonlocal curIt
        if curIt > nSteps:
            raise StopIteration("Maximum number of iterations was reached")
        dt = time.monotonic() - start_time
        if dt > minParams.timeout:
            raise StopIteration("Timeout was reached")
        G, jac = energyFn(w)
        iRound = int(curIt / stride) + 1
        if G < gTraj[iRound]:
            gTraj[iRound] = G
            wTraj[iRound] = w
            tbar.set_description(f"{minParams.minimizer}, G={G:.3f}")
        tbar.update()
        curIt += 1
        return G, jac

    # run the minimizer
    try:
        if minParams.minimizer == "MH":
            mcmc(traceEnergy, nSteps, wseed)
        else:
            basinhopping(traceEnergy, wseed, minParams)
    except StopIteration as e:
        pass

    run_duration = timedelta(seconds=int(time.monotonic() - start_time))
    tbar.close()

    gTraj[-1] = np.inf
    lastIt = np.argmax(gTraj == np.inf)
    gTraj = gTraj[:lastIt]
    wTraj = wTraj[:lastIt]

    return gTraj, wTraj


def entropy(w, w0):
    # Here 0*log(0)=0, since lim{x*log(x), x->0} = 0
    mask = w > 0
    return np.sum(w[mask] * np.log(w[mask] / w0[mask]))


def normalisationPenalty(w):
    # the wheights should approximately add up to 1.0
    # Usum_w term penalizes the deviation from 1.0; 0.1 is the force constant
    return ((w.sum() - 1.0) / 0.1) ** 2


def energyTerms(w, w0, pExp, errExp, pModel, theta):
    chi2r = np.square((pExp - pModel) / errExp).mean()
    S = entropy(w, w0)
    Usum_w = normalisationPenalty(w)
    G = chi2r * 0.5 + theta * S + Usum_w
    return G, chi2r, S, Usum_w


def memGradient(w, pExp, errExp, pModel, w0, theta, pTens):
    dchi2 = 2.0 * np.mean((pModel - pExp) / np.square(errExp) * pTens.T, axis=(1, 2))
    w_clipped = np.clip(w, np.finfo(float).eps, None)
    dS = theta * (np.log(w_clipped / w0) + 1.0)
    dSum_w = (w.sum() - 1.0) * 2.0 / (0.1 ** 2)
    jac = dchi2 * 0.5 + dS + dSum_w
    return jac


def memEnergy(w, pExp, errExp, rmp2pR, w0, theta, jac=False):
    pModel = rmp2pR(w)
    G = energyTerms(w, w0, pExp, errExp, pModel, theta)[0]
    if jac == False:
        return G, None
    return G, memGradient(w, pExp, errExp, pModel, w0, theta, rmp2pR.p)


def memFinDiff(w, pExp, errExp, rmp2pR, w0, theta):
    jac = np.zeros(w.shape[0])
    wt = np.copy(w)
    for i in range(w.shape[0]):
        dw = max(w[i] * 0.01, np.finfo(float).eps)
        wt[i] = w[i] + dw
        Gr = energyTerms(wt, w0, pExp, errExp, rmp2pR(wt), theta)[0]
        wt[i] = w[i] - dw
        Gl = energyTerms(wt, w0, pExp, errExp, rmp2pR(wt), theta)[0]
        wt[i] = w[i]
        jac[i] = (Gr - Gl) / (dw * 2.0)
    return jac


def basinhopping(energyFn, wseed, minParams):
    """ Minimize using basin hopping from SciPy """
    nSteps, minimizer, jac, fakeBounds, _ = minParams

    def GJac_bounds(w):
        # The formula is taken from: https://lmfit.github.io/lmfit-py/bounds.html
        wInt = np.sqrt(np.square(w) + 1.0) - 1.0
        if jac == False:
            return energyFn(wInt)[0]
        G, grad = energyFn(wInt)
        return G, grad * w / (wInt + 1.0)  # chain rule

    wseedInt = np.sqrt((wseed + 1) ** 2 - 1)
    energyFunInt = GJac_bounds
    minimizer_kwargs = {"method": minimizer, "options": {"maxiter": nSteps}, "jac": jac}
    if fakeBounds == False:
        minimizer_kwargs["bounds"] = [(0, 1)] * len(wseed)
        energyFunInt = lambda x: energyFn(x) if jac else energyFn(x)[0]
        wseedInt = wseed

    if minimizer == "L-BFGS-B":
        minimizer_kwargs["options"]["maxfun"] = nSteps

    rng = default_rng()

    def take_step(x):
        r = rng.uniform(0.0, 1.0, x.shape[0])
        r /= r.sum() * rng.uniform(0.5, 2.0)
        return r

    optimize.basinhopping(energyFunInt, wseedInt, minimizer_kwargs=minimizer_kwargs, take_step=take_step)


def mcmc(energyFn, nSteps, wseed):
    nBuf = 100
    nConv = 2000  # rescramble if the solution does not improve nConv iterations
    convTol = 1e-6
    nMoves = 2

    assert nConv >= nBuf

    rng = default_rng()
    gTraj = np.empty(nSteps)
    accBuf = collections.deque(maxlen=nBuf)

    kT = 0.1
    w_step = 2.0
    # Sample solutions
    w = np.copy(wseed)
    accept_ratio = -1.0
    Gprev, _ = energyFn(w)
    iAcc = 0
    for i in range(nSteps):
        # adjust sampling
        if i % nBuf == (nBuf - 1):
            accept_ratio = sum(accBuf) / nBuf
            gBuf = gTraj[:i][-nBuf:]
            iC = int(nBuf / 2)
            Gmin_old = gBuf[:iC].min()
            Gmin_new = gBuf[iC:].min()
            kT = max((Gmin_old - Gmin_new) / iC, np.finfo(float).eps)
            w_step = max(w_step * np.sqrt(0.7 + accept_ratio), 1.001)

        # Generate new weights
        w_old = np.copy(w)
        w[rng.integers(0, w.shape[0], nMoves)] *= rng.uniform(1.0 / w_step, w_step, nMoves)
        w /= w.sum()

        accept = False
        # Rescramble if the solution does not improve
        if i % nConv == (nConv - 1):
            iC = int(i - nConv / 2)
            g_old = gTraj[i - nConv + 1 : iC].min()
            g_new = gTraj[iC:i].min()
            if g_old / g_new < 1.0 + convTol:
                w = rng.uniform(0, 1, w.shape[0])
                w /= w.sum()
                Gprev = np.inf

        # Test new weights
        G, _ = energyFn(w)
        gTraj[i] = G
        if G <= Gprev:
            accept = True
        else:
            gRel = (Gprev - G) / kT
            if gRel > np.log(1e-12):  # prevents exp underflow
                r = rng.uniform(0, 1)
                accept = r < np.exp(gRel)
        if accept:
            accBuf.append(1)
            iAcc += 1
            Gprev = G
        else:
            w = w_old
            accBuf.append(0)

    return w


class distHist2D(object):
    """Functor-class that calculates kernel-smoothed distributions, given the cdf of the kernel.
    If no kernel function is provided, non-central chi distribution is used."""

    def __init__(self, rmps, sigma, bin_edges, cdfFn=None):
        numConf = rmps.shape[0]
        numBins = bin_edges.shape[0] - 1
        numPairs = len(rmps.dtype.names)
        if cdfFn is None:
            cdfFn = self.chi_cdf
        self.p = np.empty((numPairs, numBins, numConf))
        for ipair, pair in enumerate(rmps.dtype.names):
            for conf in range(numConf):
                cdf = cdfFn(bin_edges, rmps[conf][pair], sigma)
                self.p[ipair, :, conf] = cdf[1:] - cdf[:-1]

    def chi_cdf(self, x, mu, sigma):
        """
        Calculates the cdf for non-central chi distribution
        chi_cdf = integral( x/mu*(norm.pdf(x, mu, sigma) - norm.pdf(x, -mu, sigma)), x=0..x )
        """
        e1 = np.exp(-0.5 * (mu + x) ** 2 / sigma ** 2)
        e2 = np.exp(-0.5 * (mu - x) ** 2 / sigma ** 2)
        erf1 = erf((mu - x) / (np.sqrt(2.0) * sigma))
        erf2 = erf((mu + x) / (np.sqrt(2.0) * sigma))
        chi = sigma / (mu * np.sqrt(2.0 * np.pi)) * (e1 - e2) + 0.5 * (erf2 - erf1)
        return chi

    def __call__(self, weights):
        """Calculate kernel-smoothed p(Rda). numBins x numPairs"""
        assert weights.shape[0] == self.p.shape[2]
        r = np.sum(self.p * weights.reshape(1, 1, weights.shape[0]), axis=2).T
        return r
        # return r/r.sum(axis=0)


def loadWeights(path):
    w = np.loadtxt(path, usecols=[1])
    assert np.all(w > 0)
    return w


def randomWeights(size):
    w = default_rng().uniform(np.finfo(float).eps, 1.0, size)
    w /= w.sum()
    return w


def loadRmps(rmpPath):
    # Load MD data
    rmpModel = np.genfromtxt(rmpPath, names=True, delimiter="\t", deletechars="")
    pairsStr = list(rmpModel.dtype.names[1:])
    # Appends distances of all FRET pairs into single numpy array of dimension N_clusters x Npairs
    rmpArr = rfn.structured_to_unstructured(rmpModel[pairsStr])
    nanFracs = np.count_nonzero(np.isnan(rmpArr), axis=0) / rmpArr.shape[0]
    if nanFracs.max() >  0.01:
        iBad = nanFracs.argmax()
        nanPerc = nanFracs[iBad]*100.0
        badCol = pairsStr[iBad]
        print(f'Warning! File {rmpPath} contains {nanPerc:.1f}% of "NaN" entries in {badCol} column!')
    # Replaces NaN values, if such exist, with -100, hence no need to additionally clean the data from NaN occurences
    rmpArr = np.nan_to_num(rmpArr, nan=-100.0)
    dt = np.dtype(rmpModel.dtype.descr[1:])
    return rfn.unstructured_to_structured(rmpArr, dt)


def loadExpData(pExpPathMask, rda_min, rda_max, pairsSorted):
    # glob() allows usage of Unix matching patterns, e.g. '*.txt'.
    # Creates a list of strings: 'Lif_data/ucfret_20201210/137-251.txt' etc.
    pExpPathList = glob(pExpPathMask)
    numPairs = len(pExpPathList)  # determines number of loaded files
    # Checks if experimental data is actually read, by checking if there are more than 0 of read files.
    assert numPairs > 0

    # Load as structured data only the first file "137-215.txt", because at this moment we just need bins information
    data = np.genfromtxt(pExpPathList[0], names=True, delimiter="\t", deletechars="")
    bins_left = data["Rda_left"]
    bins_right = data["Rda_right"]
    # make sure that no bins were skipped and
    # ensures continuous binning(i.e. 0-10,10-20... and not 0-10, 20-30)
    assert np.all(np.isclose(bins_left[1:], bins_right[:-1]))
    # np.flatnonzero gives the indices of nonzero elements of bins_left array that are >= rda_min.
    # Then take the first element
    binStart = np.flatnonzero(bins_left >= rda_min)[0]
    # np.flatnonzero gives the indices of nonzero elements of bins_right array that are <= rda_max.
    # Then take the last element's index+1
    binEnd = np.flatnonzero(bins_right <= rda_max)[-1] + 1
    numBins = binEnd - binStart

    pExp = np.full((numBins, numPairs), -1.0)
    err = np.full((numBins, numPairs), -1.0)
    for pExpPath in pExpPathList:
        # In a list of strings: 'Lif_data/ucfret_20201210/137-251.txt' remove '.txt',
        # take the last string separated with /, replace - with _ -> '137_251'
        pair = pExpPath.replace(".txt", "").split("/")[-1].replace("-", "_")
        # ensure that pair name from exp. file exists in the header of "RDAMean.dat"
        assert pair in pairsSorted
        # index of the given FRET pair in pairsSorted(pairsStr), i.e. in the simulation files
        iP = pairsSorted.index(pair)
        data = np.genfromtxt(pExpPath, names=True, delimiter="\t", deletechars="")
        assert np.all(bins_left == data["Rda_left"])
        # sorts the pExp in the same order as in pairsSorted(pairsStr), given by iP
        pExp[:, iP] = data["prda"][binStart:binEnd]
        err[:, iP] = data["prda_error"][binStart:binEnd]
    assert np.all(pExp >= 0.0)
    assert np.all(err > 0.0)

    edges = np.append(bins_left[binStart:binEnd], bins_right[binEnd - 1])

    minErrFrac = 0.01 / numBins  # minimal allowed error for a bin
    if np.any(err < minErrFrac):
        print(f"WARNING: some experimental errors are below {minErrFrac:.1e}, this is unusual!")
    assert np.all(err > np.finfo(float).eps)
    assert np.all(pExp.sum(axis=0) < 1.03)

    return edges, pExp, err


def plot_pRda(path, rda_edges, model, exp, err, model_initial, pRmp, comm):
    rda = (rda_edges[:-1] + rda_edges[1:]) * 0.5
    bin_width = rda_edges[1:] - rda_edges[:-1]

    fig, ax = plt.subplots()
    ax.errorbar(rda, exp / bin_width, yerr=err / bin_width, color="black", label="experiment", fmt="o", markersize=2, elinewidth=1.0)
    ax.plot(rda, model / bin_width, "bo--", label="MD+weights", markersize=2, linewidth=1)
    ax.plot(rda, model_initial / bin_width, "ro--", label="MD initial", markersize=2, linewidth=1)

    axR = ax.twinx()
    axR.plot(rda, pRmp / bin_width, "go--", label="p(Rmp), MD+reweigting", markersize=2, linewidth=1)
    axR.set_ylabel(r'$p(R_{MP})$ per $\AA$ / $\AA^{-1}$', color="g")
    axR.set_ylim(0.0, None)

    ax.set_ylim(0.0, None)
    ax.set_xlabel(r'$R_{app}$ / $\AA$')
    ax.set_ylabel(r'$p(R_{DA})$ per $\AA$ / $\AA^{-1}$')
    ax.text(0.77,0.97, comm, transform=ax.transAxes, verticalalignment='top')
    fig.legend(loc="upper center", ncol=4, prop={"size": 8}, bbox_to_anchor=(0.5, 0.06))
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.15)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_pRdaTraj(savePrefix, gTraj, wTraj, w0, theta, rmp2pR, rmps, bin_edges, pExp, errExp):
    #usefull for debugging
    gCur = np.inf
    pModIni = rmp2pR(w0)
    for i in range(gTraj.shape[0]):
        if gTraj[i] >= gCur:
            continue
        gCur = gTraj[i]
        w = wTraj[i]
        pModel = rmp2pR(w)
        _, chi2r, S, _ = energyTerms(w, w0, pExp, errExp, rmp2pR(w), theta)
        for ip, pair in enumerate(rmps.dtype.names):
            pRmp = np.histogram(rmps[pair], bins=bin_edges, weights=w)[0]
            path = savePrefix + f"/{pair}_{i:05}.png"
            tpair = pair.replace('_','\_')
            chi2r_pair = np.square((pModel[:, ip] - pExp[:, ip]) / errExp[:, ip]).mean()
            comm = '\n'.join([f'$\chi^2_{{r,{tpair}}}={chi2r_pair:.1f}$',
                          f'$\chi^2_{{r,tot}}={chi2r:.1f}$',
                          f'$S={S:.1f}$'])
            plot_pRda(path, bin_edges, pModel[:, ip], pExp[:, ip], errExp[:, ip], pModIni[:, ip], pRmp, comm)
            

def save_pRda(savePath, rmpModel, weights, sigma):
    edges_full = np.linspace(0.0, 200.0, 201)
    pModel = distHist2D(rmpModel, sigma, edges_full)(weights)
    pModData = np.column_stack((edges_full[:-1], edges_full[1:], pModel))
    header = ["Rda_left", "Rda_right"] + ["p_" + p for p in rmpModel.dtype.names]
    header = "\t".join(header)
    np.savetxt(savePath, pModData, fmt="%.5e", header=header, delimiter="\t", comments="")


def save_energies(path, wTraj, gTraj, w0, theta, saveStride):
    e_traj = np.zeros((wTraj.shape[0], 5))
    for i, w in enumerate(wTraj):
        S = entropy(w, w0)
        Usum_w = normalisationPenalty(w)
        chi2r = gTraj[i] - Usum_w - S * theta
        e_traj[i] = i * saveStride, gTraj[i], chi2r, S, Usum_w
    header = "step\tG\tchi2r\tS\tEnergy_w.sum"
    np.savetxt(path + ".dat", e_traj, fmt=["%d"] + ["%.6e"] * 4, delimiter="\t", header=header)

    x, G, chi2r, S, resW = e_traj.T
    fig, ax = plt.subplots()
    ax.plot(x, G, "r-", label="G", linewidth=1)
    ax.plot(x, chi2r, "g-", label="chi2", linewidth=1)
    ax.plot(x, S, "b-", label="S", linewidth=1)
    ax.plot(x, resW, "m-", label="Esum_w", linewidth=1)
    y_max = min(e_traj[0].max() * 1.1, G.min() * 20.0)
    ax.set_ylim(0.0, y_max)
    ax.set_xlim(0, x[-1])
    ax.set_xlabel("# iteration")
    ax.set_ylabel("Optimization energy")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, prop={"size": 10})

    fig.tight_layout()
    fig.savefig(path + ".png", dpi=300)
    plt.close(fig)


def dictlist2arr(dicts):
    dt_tuples = []
    for key, value in dicts[0].items():
        if not isinstance(value, str):
            value_dtype = np.array([value]).dtype
        else:
            value_dtype = "|U{}".format(max([len(d.get(key, "")) for d in dicts]))
        dt_tuples.append((key, value_dtype))
    dt = np.dtype(dt_tuples)

    values = [tuple(d.get(name, "-1") for name in dt.names) for d in dicts]
    return np.array(values, dtype=dt)


def validMinimizers(jac, fakeBounds):
    # These methods converge the fastest if jacobian is enabled: 'L-BFGS-B', 'TNC', 'CG'
    # Without jacobian 'TNC' converges slowly
    # 'SLSQP', 'BFGS' are extremely slow if the ensemble is large and the jacobian is enabled
    # Powell takes more iterations to converge and can not use a jacobian
    # 'dogleg', 'trust-exact', 'trust-krylov', 'trust-ncg' require jacobian+hessian (not implemented)
    minimizers = ["L-BFGS-B"]
    if fakeBounds:
        minimizers += ["CG"]
    if jac:
        minimizers += ["TNC"]
        if fakeBounds:
            minimizers += ["Newton-CG"]
    else:
        minimizers += ["MH", "trust-constr", "Powell"]
        if fakeBounds:
            minimizers += ["BFGS", "SLSQP"]
    return minimizers


def main():
    all_minimizers = ["valid"] + list(set(validMinimizers(True, True) + validMinimizers(False, True)))

    ap = argparse.ArgumentParser(description='Use Maximum Entropy (MEM) to reweight a conformational emseble and improve agreement with inter-dye distance distributions obtained from FRET experimental data.')
    ap.add_argument('--exp', type=str, required=True, help='Path pattern for experimental data')
    ap.add_argument('--weights', type=str, required=True, help='Path to conformer weights')
    ap.add_argument('--rmps', type=str, required=True, help='Path to conformer distance (Rmp) data')
    ap.add_argument('--out', type=str, required=True, help='Path to the results directory')
    ap.add_argument('--theta', nargs='+', type=float, required=True, action='extend', 
                    help='Relative weights of χ2 and entropy (S)')
    ap.add_argument('--sigma', type=float, default=8.5, 
                    help='Widgth parameter of inter-dye distance distributions (Angstrom)')
    ap.add_argument('--seed', nargs='+', type=str, action='extend', default=[],
                    help="Initial weights, supported options are: 'reference', 'random' or path to a weights file")
    ap.add_argument('--steps', type=int, default=300, 
                    help='Number of iterations to take during minimization')
    ap.add_argument('--minimizer', nargs='+', type=str, action='extend', default=[], 
                    help='Local minimization algorithm: '+'|'.join(all_minimizers))
    ap.add_argument('--jac', type=bool, action=argparse.BooleanOptionalAction, default=True, 
                    help="Use jacobian during minimization. Some minimizers can't use jacobian.")
    ap.add_argument('--fake_bounds', type=bool, action=argparse.BooleanOptionalAction, default=True, 
                    help='Transform variables instead of using constraints. Recommended.')
    ap.add_argument('--timeout', type=float, default=np.inf, 
                    help="Stop minimization if it didn't finish in given time (seconds).")
    ap.add_argument('--threads', type=int, default=1, help='Number of parallel threads')
    args = ap.parse_args()

    memargs = {
        "expPattern": args.exp,
        "w0": loadWeights(args.weights),
        "rmps": loadRmps(args.rmps),
        "sigma": args.sigma,
    }

    unk = list(set(args.minimizer) - set(all_minimizers))
    if len(unk) > 0:
        print("Error! Unknown minimizers: " + ", ".join(unk))
        return
    minimizers = list(args.minimizer)
    if "valid" in args.minimizer:
        minimizers.extend(validMinimizers(args.jac, args.fake_bounds))
        minimizers.remove("valid")
    if not minimizers:
        minimizers = ["L-BFGS-B"]

    if not args.seed:
        args.seed = ["reference"]

    kwargsLst = []
    for minimizer, theta, seed in itertools.product(minimizers, args.theta, args.seed):
        savePath = f"{args.out}/th{theta:.2f}_{minimizer}_{seed[:3]}_{len(kwargsLst)}"
        minParams = MinimizerSettings(args.steps, minimizer, args.jac, args.fake_bounds, args.timeout)
        d = memargs.copy()
        for s in ("savePath", "seed", "theta", "minParams"):
            d[s] = locals()[s]
        kwargsLst.append(d)
    if len(kwargsLst) == 1:
        kwargsLst[0]["savePath"] = args.out
    if not os.path.exists(args.out):
        os.mkdir(args.out)
    
    results = []
    if args.threads == 1:
        for d in kwargsLst:
            results.append(memWrap(d))
    else:
        pool = Pool(args.threads)
        tbar = trange(len(kwargsLst))
        for d in pool.imap_unordered(memWrap, kwargsLst):
            results.append(d)
            tbar.set_description(f'{d.get("savePath","")}, G={d.get("G_min",-1.0):.3f}')
            tbar.update()

    restab = dictlist2arr(results)
    restab = restab[np.argsort(restab['S'])]
    header = "\t".join(restab.dtype.names)
    fmtMap = {np.int32: "%d", np.float64: "%.3f"}
    fmt = [fmtMap.get(np.dtype(t).type, "%s") for n, t in restab.dtype.descr]
    for out in [f"{args.out}/summary.dat", sys.stdout]:
        np.savetxt(out, restab, fmt=fmt, delimiter="\t", header=header, comments="")


if __name__ == "__main__":
    main()

# TODO: documentation, testing, PyPi, uncertainty of weights
