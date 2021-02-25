#!/bin/bash

../maxent.py \
--exp 'ucfret_20201210/*.txt' \
--weights MD/cluster_weights_ff99sb-disp.dat \
--rmps MD/Rmp_ff99sb-disp.dat \
--out result_ff99sb-disp \
--theta 0.31
