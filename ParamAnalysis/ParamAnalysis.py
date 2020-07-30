# Parameter analysis: R statistics on chains
# R< 1.1 density is unique

# author: Zheng Wang

import numpy as np
def R_stat(data):
    ###B step 1 variance of mean of m chains
    num_data = data.shape[1]
    num_param = data.shape[2]
    num_chain = data.shape[0]
    var_chs =  num_data* data.mean(1).std(0)**2
    
    ### step 2. the average of variance of m chains W

    m_var_chs = (data.std(1)**2).mean(0)
    
    ### step 3 target of mean: mean of 4*datapoints 
    m_target = data.mean(1).mean(0)
    
    #### step 4 estimate of target variance

    v_target = (num_data-1.0)/num_data*m_var_chs+ 1.0/num_data* var_chs
    #### step 5 
    V_hat = v_target +var_chs/num_chain/num_data
    
    v_var = (data.std(1)**2).std(0)**2
    
    v_V_hat = (num_data/(num_data- 1.0))**2/num_chain*v_var +((num_chain+1.0)/num_chain/num_data)**2*2/(num_chain -1.0)*var_chs**2+\
      2*(num_chain + 1.0 )*(num_data- 1.0)/num_chain/num_data/num_chain*(np.diag(np.cov((data.std(1)**2).T,\
                (data.mean(1)**2).T)[:num_param,:][:,num_param:])\
                -2*m_target*np.diag(np.cov((data.std(1)**2).T, data.mean(1).T)[:num_param,:][:,num_param:]))
    df = 2.0* V_hat**2/v_V_hat
    
    ### R
    R = V_hat/m_var_chs*df/(df-2.0)
    
    return R
