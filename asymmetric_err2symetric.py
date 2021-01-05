#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import sys

def main(inPath, outPath):
    data = np.genfromtxt(inPath,names=True,delimiter='\t',deletechars="")
    rda_l = data['Rda_left']
    rda_r = data['Rda_right']
    p_top = data['prda']
    err_neg = data['err_neg']
    err_pos = data['err_pos']
    
    p_cent = p_top + (err_pos - err_neg)*0.5
    err_mean = (err_pos + err_neg)*0.5
    
    res=np.column_stack([rda_l, rda_r, p_cent, err_mean])
    
    np.savetxt(outPath, res, delimiter='\t', header='Rda_left	Rda_right	prda	prda_error',fmt='%.5f', comments="")

if __name__ == '__main__':
    main(sys.argv[1],sys.argv[2])
