#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import sys

def main(inDir, outPrefix, confThreshold=0.68):
    rda=np.loadtxt(inDir+'/pRapp_density_rda_axis.txt',usecols=[0])
    p=np.loadtxt(inDir+'/pRapp_density.txt')
    rda_max = 2.0*rda[-1] - rda[-2]
    rda_step = (rda[1] - rda[0])*2.0
    edges = np.concatenate(([0.], np.arange(30,80,rda_step), [rda_max]))

    assert np.allclose(rda[1:]-rda[:-1],rda[1]-rda[0])

    y=np.zeros(rda.shape[0])
    ymax=np.zeros(rda.shape[0])
    ymin=np.zeros(rda.shape[0])
    nbins=p.shape[0]
    smax=(confThreshold+1.0)*0.5
    smin=(1-confThreshold)/2
    for i in range(rda.shape[0]):
        cs=np.cumsum(p[i])
        cs[0]=0.0
        ymax[i] = np.argmax(cs/cs[-1]>smax)/nbins
        ymin[i] = 1.0-np.argmax((cs[::-1]/cs[-1])<smin)/nbins
        y[i] = np.argmax(cs/cs[-1]>=0.5)/nbins
        
    ysum = y.sum()
    if np.abs(ysum-1.0)>0.1:
        print("Warning! Total fraction is not ~1, normalizing.")
        ymin/=ysum
        ymax/=ysum
        y/=ysum
    
    bin_y, bin_err_neg, bin_err_pos = histogram(rda, y, y-ymin, ymax-y, edges)
    data=np.column_stack([edges[:-1], edges[1:], bin_y, bin_err_neg, bin_err_pos])
    np.savetxt(outPrefix+'.txt',data,delimiter='\t',header='Rda_left	Rda_right	prda	err_neg	err_pos',fmt='%.5f', comments="")

    fig, ax = plt.subplots()
    heatmap=ax.imshow(p.T, origin='lower', extent=[rda[0],rda[-1],0,1.0/ysum], cmap='hot', aspect='auto')
    cbar = fig.colorbar(heatmap, ax=ax)
    cbar.ax.set_ylabel("p-value")
    
    ax.plot(rda,y, color='#55F', linewidth=1)
    ax.plot(rda,ymax, color='#55F', linewidth=1)
    ax.plot(rda,ymin, color='#55F', linewidth=1)
    ax.set_xlabel('Rda / A')
    ax.set_ylabel('species fraction')
    
    fig.savefig(outPrefix+'.png',dpi=300)

def histogram(x, y, err_neg, err_pos, edges):
    in_data=np.column_stack([x,y,err_neg,err_pos])
    
    bin_y = np.zeros(len(edges)-1)
    bin_err_neg = np.zeros_like(bin_y)
    bin_err_pos = np.zeros_like(bin_y)
    
    for i_bin in range(len(edges)-1):
        x_left = edges[i_bin]
        x_right = edges[i_bin+1]
        mask = (x>=x_left) & (x<x_right)
        in_slice = in_data[mask]
        bin_y[i_bin] = in_slice[:,1].sum()
        bin_err_neg[i_bin] = np.sqrt(np.square(in_slice[:,2]).sum())
        bin_err_pos[i_bin] = np.sqrt(np.square(in_slice[:,3]).sum())
    
    return bin_y, bin_err_neg, bin_err_pos

if __name__ == '__main__':
    main(sys.argv[1],sys.argv[2])
    #main('ucfret_137-215/pRda_analysis', 'p-range_approx/137-215')
