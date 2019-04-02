# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 11:45:16 2016

@author: tnan287
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 12:54:15 2016

@author: tnan287
"""

import numpy as np
import GPy
from binary_smgp_combine import BIN_SMGP
import scipy.io
import time
import random 




m = BIN_SMGP()


dataset='usps'
inducing_points=[50, 200]
for inducing_i in range(inducing_points.__len__()):
    num_inducing=inducing_points[inducing_i]
    repeat=10;
    ac_full=[];
    pred_full=[];
    log_pred_full=[];
    test_error_full=[];
    Yt_m_full=[];
    Yt_v_full=[];
    exec_time_full=[];
    
    for i in range(repeat):
        start_time = time.time()
        mat = scipy.io.loadmat('dataset/'+dataset+'.mat')
        X = mat['X']
        Y= mat['Y']
        Xt=mat['Xt']
        Yt=mat['Yt']
            
        
        trainN=X.shape[0]   
        model = m.inference(X, Y,numZ=num_inducing,num_local_Z=num_inducing,num_cluster=3, batchsize=X.shape[0])
        ac, pred,decv, test_error, Yt_m, Yt_v = m.prediction1(model, Xt,Yt)
        exec_time= (time.time() - start_time)
        print("--- %s seconds ---" % exec_time)
        ac_full.append(ac)
        pred_full.append(pred)
        log_pred_full.append(decv)
        test_error_full.append(test_error)
        Yt_m_full.append(Yt_m)
        Yt_v_full.append(Yt_v)
        exec_time_full.append(exec_time)
    
    overall_ac=sum(ac_full)/float(len(ac_full));
    std_ac =np.std(ac_full);
    overall_log_pred=sum(log_pred_full)/float(len(log_pred_full));
    std_log_pred=np.std(log_pred_full);
    print("--- accuracy: %s, std: %s ---" % (overall_ac, std_ac))
    print("--- error: %s ---" % (1-overall_ac))
    print("--- log pred: %s , std: %s ---" % (overall_log_pred,std_log_pred))
    print("--- exec: %s , std: %s ---" % (sum(exec_time_full)/float(len(exec_time_full)),np.std(exec_time_full)))