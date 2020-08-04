#!/usr/bin/env python3
import sys
def simBOLD(sub):
    import os

    import numpy as np
    import pandas as pd
    import scipy.io
    import sys
    sys.path.append("/brunhild/mcintosh_lab/jwang/git_Lab/WWD_Model/Model/")
    sys.path.append("/brunhild/mcintosh_lab/jwang/git_Lab/WWD_Model/ParamAnalysis/")
    import WWD_model
    import ParamAnalysis
    
    out_dir =  '/liberatrix/mcintosh_lab/jwang/ModelFitting/babyTVB/DecoAdam/'
    SC_dir_pre= '/alkeste/mcintosh_lab/lrokos/tvb_baby/test_dti/sub-'
    SC_dir_post = '/ses-1/dwi/results/RM/defaults/SCmats_defaults.mat'
    TS_dir_pre= '/alkeste/mcintosh_lab/aeasson/kspipe/babyTVB_0606/bold_prep/_subject_'
    TS_dir_post = '/computeFC/TS.txt'
    SC_file = SC_dir_pre + sub + SC_dir_post
    TS_file = TS_dir_pre + sub + TS_dir_post
    if os.path.isfile(SC_file) and os.path.isfile(TS_file):
        SC_data= scipy.io.loadmat(SC_file)
        SC = SC_data['probConnect']
        TS= np.loadtxt(TS_file)
        TS =(TS.T -TS.T.mean(axis= 0)).T ### demean across ROIs on ts
        SC = (SC+SC.T)*0.5

        #### threshold on your SC if needed
        SC1=SC[:48,:48].copy()
        SC2=SC[48:96,48:96].copy()
        SC3=SC[:48,48:96].copy()
        mask1 = (SC1-SC1.mean(axis=1)< .5*SC1.std(axis=1)) 
        SC1[mask1]=0
        SC2[(SC2-SC2.mean(axis=1) <  0.5*SC2.std(axis=1)) ]=0
        SC3[(SC3-SC3.mean(axis=1)<  0.5*SC3.std(axis=1)) ]=0
        SC[:48,:48] = SC1
        SC[48:96,48:96] = SC2
        SC[:48,48:96] = 1*SC3
        SC[48:96,:48] = 1*SC3.T
        """mask =  (SC-SC.mean(axis=1)< 0.0*SC.std(axis=1)) 
        SC[mask] = 0"""

        # We use the logarithm to decrease the values of the weights
        w = np.log1p(SC)/np.linalg.norm(np.log1p(SC))
        #ts = TS#ts -ts.min()
        ts =TS ### due to model limit you have to shrink you empirical ts to certain range
        exp_dt = 3.0              # time interval between two data points

        md_wwd = WWD_model.WongWangDeco(w, ts, exp_dt)
        total_t = exp_dt*len(ts)  # total time length of the time series



        batch_size = 15
        epoch_num = 40
        alpha = 0.001
    else:
        print("data is missing")

    # in order to see the stability of the fitted parameteres we model 4 times for each 
    num_chain = 4


    for i in range(num_chain):
        print('chain:'+ str(i))

        cost, theta, states, errors, e_ep = md_wwd.deco_train(batch_size, epoch_num, alpha)
        
        results_sim, Es_sim, Is_sim = md_wwd.deco_test(1200)
        np.savetxt(out_dir+'babyTVB_'+sub+'_chain_'+str(i)+'_parameters.txt', theta)
        np.savetxt(out_dir+'babyTVB_'+sub+'_chain_'+str(i)+'_sim_bold.txt', results_sim.T)
        np.savetxt(out_dir+'babyTVB_'+sub+'_chain_'+str(i)+'_sim_I.txt', Is_sim.T)
        np.savetxt(out_dir+'babyTVB_'+sub+'_chain_'+str(i)+'_sim_E.txt', Es_sim.T)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        simBOLD(sys.argv[1])
    else:
        raise SystemExit("usage:  python hello.py <name>")
