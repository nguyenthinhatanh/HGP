# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 16:08:19 2016

@author: tnan287
"""

from . import LatentFunctionInference
from ...util import linalg
from ...util import choleskies
import numpy as np
from .posterior import Posterior
from scipy.linalg.blas import dgemm, dsymm, dtrmm
from .svgp import SVGP as svgp_inf
#from local_svgp import Local_SVGP
from collections import Counter

class SMGP_L(LatentFunctionInference):

    def inference(self, q_u_mean, q_u_chol, kern, Z, likelihood,  mean_function,Y_metadata, l_svgps, X, Y, rho,  KL_scale=1.0, batch_scale=1.0):

        num_data, _ = Y.shape
        num_inducing, num_outputs = q_u_mean.shape
        g_log_marginal_likelihood=0;
        
        #g_posterior, g_log_marginal_likelihood, g_grad_dict, g_mu, g_v, g_A, g_L, g_S, g_Kmmi, g_Kmmim = self.global_inference(q_u_mean, q_u_chol, kern, X, Z, likelihood, Y, mean_function, Y_metadata, KL_scale=KL_scale, batch_scale=batch_scale)
        
#        g_grad_dict['dL_dchol'] = choleskies.flat_to_triang(g_grad_dict['dL_dchol'])
        l_posteriors=[];
        l_grad_dicts =[];
        
        for i in range(len(l_svgps)):
            l_posterior, l_log_marginal, l_grad_dict, index = self.local_inference(X, Y, rho[:, i].reshape(rho.shape[0],1), l_svgps[i].q_u_mean, l_svgps[i].q_u_chol, l_svgps[i].kern, l_svgps[i].Z, l_svgps[i].likelihood,  l_svgps[i].mean_function,   l_svgps[i].Y_metadata, KL_scale, batch_scale)
            l_svgps[i].set_data(X[index,:],Y[index,:])            
            l_posteriors.append(l_posterior)
            l_grad_dicts.append(l_grad_dict)            
            g_log_marginal_likelihood = g_log_marginal_likelihood + l_log_marginal
            #g_grad_dict = dict(Counter(g_grad_dict)+Counter(add_g_grad_dict));
#            g_grad_dict = self.addDicts(g_grad_dict, add_g_grad_dict)

#        g_grad_dict['dL_dchol'] = choleskies.triang_to_flat(g_grad_dict['dL_dchol'])
        
        return  l_posteriors, g_log_marginal_likelihood, l_grad_dicts

    def addDicts(self, a, b):
        sa = set(a.keys())
        sb = set(b.keys())
        sc= set()
        for k in sa&sb:
            if (a[k] is None) or (b[k] is None) :
                sc.add(k)
        c = dict( \
        [(k, a[k]+b[k]) for k in (sa&sb)-sc ] + \
        [(k, a[k]) for k in sa-sb-sc ] + \
        [(k, b[k]) for k in sb-sa-sc ] +\
        [(k, None) for k in sc ] \
        )
        return c
        


#    def local_inference(self,  X, Y, rho, q_u_mean, q_u_chol, kern, Z, likelihood,  mean_function,  g_q_u_mean, g_q_u_chol, g_kern, g_Z, g_likelihood, g_mean_function=None,Y_metadata=None, KL_scale=1.0, batch_scale=1.0):
#
#        num_data, _ = Y.shape
#        num_inducing, num_outputs = q_u_mean.shape
#
#        #expand cholesky representation
#        L = choleskies.flat_to_triang(q_u_chol)
#
#
#        S = np.empty((num_outputs, num_inducing, num_inducing))
#        [np.dot(L[i,:,:], L[i,:,:].T, S[i,:,:]) for i in range(num_outputs)]
#        #Si,_ = linalg.dpotri(np.asfortranarray(L), lower=1)
#        Si = choleskies.multiple_dpotri(L)
#        logdetS = np.array([2.*np.sum(np.log(np.abs(np.diag(L[i,:,:])))) for i in range(L.shape[0])])
#        
#        if np.any(np.isinf(Si)):
#            raise ValueError("Cholesky representation unstable")
#            
#        g_num_inducing, num_outputs = g_q_u_mean.shape
#        
#        #expand cholesky representation
#        g_L = choleskies.flat_to_triang(g_q_u_chol)
#
#
#        g_S = np.empty((num_outputs, g_num_inducing, g_num_inducing))
#        [np.dot(g_L[i,:,:], g_L[i,:,:].T, g_S[i,:,:]) for i in range(num_outputs)]
#        #Si,_ = linalg.dpotri(np.asfortranarray(L), lower=1)
#        g_Si = choleskies.multiple_dpotri(g_L)
#        #g_logdetS = np.array([2.*np.sum(np.log(np.abs(np.diag(g_L[i,:,:])))) for i in range(g_L.shape[0])])
#        
#
#        if np.any(np.isinf(g_Si)):
#            raise ValueError("Cholesky representation unstable")
#
#        #compute mean function stuff
##        if mean_function is not None:
##            prior_mean_u = mean_function.f(Z)
##            prior_mean_f = mean_function.f(X)
##        else:
##            prior_mean_u = np.zeros((num_inducing, num_outputs))
##            prior_mean_f = np.zeros((num_data, num_outputs))
#
#        prior_mean_u = np.zeros((num_inducing, num_outputs))
#        prior_mean_f = np.zeros((num_data, num_outputs))
#            
#        if g_mean_function is not None:
#            g_prior_mean_u = g_mean_function.f(g_Z)
#            g_prior_mean_f = g_mean_function.f(X)
#        else:
#            g_prior_mean_u = np.zeros((g_num_inducing, num_outputs))
#            g_prior_mean_f = np.zeros((num_data, num_outputs))
#            
#
#        #compute kernel related stuff
#        Kmm = kern.K(Z)
#        Kmn = kern.K(Z, X)
#        Knn_diag = kern.Kdiag(X)
#        Lm = linalg.jitchol(Kmm)
#        logdetKmm = 2.*np.sum(np.log(np.diag(Lm)))
#        Kmmi, _ = linalg.dpotri(Lm)
#        
#        g_Kmm = g_kern.K(g_Z)
#        g_Kmn = g_kern.K(g_Z, X)
#        #g_Knn_diag = g_kern.Kdiag(X)
#        g_Lm = linalg.jitchol(g_Kmm)
#        #g_logdetKmm = 2.*np.sum(np.log(np.diag(g_Lm)))
#        g_Kmmi, _ = linalg.dpotri(g_Lm)
#        
#
#        #compute the marginal means and variances of q(f)
#        A, _ = linalg.dpotrs(Lm, Kmn)
#        mu = prior_mean_f + np.dot(A.T, q_u_mean - prior_mean_u)
#        v = np.empty((num_data, num_outputs))
#        for i in range(num_outputs):
#            tmp = dtrmm(1.0,L[i].T, A, lower=0, trans_a=0)
#            v[:,i] = np.sum(np.square(tmp),0)
#        v += (Knn_diag - np.sum(A*Kmn,0))[:,None]
#        
#        
#        g_A, _ = linalg.dpotrs(g_Lm, g_Kmn)
#        g_mu = g_prior_mean_f + np.dot(g_A.T, g_q_u_mean - g_prior_mean_u)
#        g_v = np.empty((num_data, num_outputs))
#        for i in range(num_outputs):
#            tmp = dtrmm(1.0,g_L[i].T, g_A, lower=0, trans_a=0)
#            g_v[:,i] = np.sum(np.square(tmp),0)
#        #Anh v += (Knn_diag - np.sum(A*Kmn,0))[:,None]
#        
#        #final marginal means and variances of q(f)
#        mu += g_mu
#        v += g_v
#        
#        #compute the KL term
#        Kmmim = np.dot(Kmmi, q_u_mean)
#        KLs = -0.5*logdetS -0.5*num_inducing + 0.5*logdetKmm + 0.5*np.sum(Kmmi[None,:,:]*S,1).sum(1) + 0.5*np.sum(q_u_mean*Kmmim,0)
#        KL = KLs.sum()
#        #gradient of the KL term (assuming zero mean function)
#        dKL_dm = Kmmim.copy()
#        dKL_dS = 0.5*(Kmmi[None,:,:] - Si)
#        dKL_dKmm = 0.5*num_outputs*Kmmi - 0.5*Kmmi.dot(S.sum(0)).dot(Kmmi) - 0.5*Kmmim.dot(Kmmim.T)
#
##        if mean_function is not None:
##            #adjust KL term for mean function
##            Kmmi_mfZ = np.dot(Kmmi, prior_mean_u)
##            KL += -np.sum(q_u_mean*Kmmi_mfZ)
##            KL += 0.5*np.sum(Kmmi_mfZ*prior_mean_u)
##
##            #adjust gradient for mean fucntion
##            dKL_dm -= Kmmi_mfZ
##            dKL_dKmm += Kmmim.dot(Kmmi_mfZ.T)
##            dKL_dKmm -= 0.5*Kmmi_mfZ.dot(Kmmi_mfZ.T)
##
##            #compute gradients for mean_function
##            dKL_dmfZ = Kmmi_mfZ - Kmmim
#
#        #quadrature for the likelihood
#        F, dF_dmu, dF_dv, dF_dthetaL = likelihood.variational_expectations(Y, mu, v, Y_metadata=Y_metadata)
#        
#        #multiply with rho
#        F, dF_dmu, dF_dv =  F*rho, dF_dmu*rho, dF_dv*rho
#        
#        #rescale the F term if working on a batch
#        F, dF_dmu, dF_dv =  F*batch_scale, dF_dmu*batch_scale, dF_dv*batch_scale
#        if dF_dthetaL is not None:
#            dF_dthetaL =  dF_dthetaL.sum(1).sum(1)*batch_scale
#
#        #derivatives of expected likelihood w.r.t. global parameters, assuming zero mean function
#        g_Kmmim = np.dot(g_Kmmi, g_q_u_mean)
#        g_Adv = g_A[None,:,:]*dF_dv.T[:,None,:] # As if dF_Dv is diagonal, D, M, N
#        g_Admu = g_A.dot(dF_dmu)
#        g_Adv = np.ascontiguousarray(g_Adv) # makes for faster operations later...(inc dsymm)
#        g_AdvA = np.dot(g_Adv.reshape(-1, num_data),g_A.T).reshape(num_outputs, g_num_inducing, g_num_inducing )
#        tmp = np.sum([np.dot(a,s) for a, s in zip(g_AdvA, g_S)],0).dot(g_Kmmi)
#        #dF_dgKmm = -g_Admu.dot(g_Kmmim.T) + g_AdvA.sum(0) - tmp - tmp.T
#        dF_dgKmm = -g_Admu.dot(g_Kmmim.T)  - tmp - tmp.T
#        dF_dgKmm = 0.5*(dF_dgKmm + dF_dgKmm.T) # necessary? GPy bug?
#        tmp = g_S.reshape(-1, g_num_inducing).dot(g_Kmmi).reshape(num_outputs, g_num_inducing , g_num_inducing )
#        #tmp = 2.*(tmp - np.eye(g_num_inducing)[None, :,:])
#        tmp = 2.*tmp
#
#        dF_dgKmn = g_Kmmim.dot(dF_dmu.T)
#        for a,b in zip(tmp, g_Adv):
#            dF_dgKmn += np.dot(a.T, b)
#
#        dF_dgm = g_Admu
#        dF_dgS = g_AdvA
#        
#        #derivatives of expected likelihood w.r.t. local parameters, assuming zero mean function
#        Adv = A[None,:,:]*dF_dv.T[:,None,:] # As if dF_Dv is diagonal, D, M, N
#        Admu = A.dot(dF_dmu)
#        Adv = np.ascontiguousarray(Adv) # makes for faster operations later...(inc dsymm)
#        AdvA = np.dot(Adv.reshape(-1, num_data),A.T).reshape(num_outputs, num_inducing, num_inducing )
#        tmp = np.sum([np.dot(a,s) for a, s in zip(AdvA, S)],0).dot(Kmmi)
#        dF_dKmm = -Admu.dot(Kmmim.T) + AdvA.sum(0) - tmp - tmp.T
#        dF_dKmm = 0.5*(dF_dKmm + dF_dKmm.T) # necessary? GPy bug?
#        tmp = S.reshape(-1, num_inducing).dot(Kmmi).reshape(num_outputs, num_inducing , num_inducing )
#        tmp = 2.*(tmp - np.eye(num_inducing)[None, :,:])
#
#        dF_dKmn = Kmmim.dot(dF_dmu.T)
#        for a,b in zip(tmp, Adv):
#            dF_dKmn += np.dot(a.T, b)
#
#        dF_dm = Admu
#        dF_dS = AdvA
#
#        #adjust gradient to account for mean function
##        if mean_function is not None:
##            dF_dmfX = dF_dmu.copy()
##            dF_dmfZ = -Admu
##            dF_dKmn -= np.dot(Kmmi_mfZ, dF_dmu.T)
##            dF_dKmm += Admu.dot(Kmmi_mfZ.T)
#
#
#        #sum (gradients of) expected likelihood and KL part w.r.t local parameters
#        log_marginal = F.sum() - KL
#        dL_dm, dL_dS, dL_dKmm, dL_dKmn = dF_dm - dKL_dm, dF_dS- dKL_dS, dF_dKmm- dKL_dKmm, dF_dKmn
#    
#        dL_dchol = 2.*np.array([np.dot(a,b) for a, b in zip(dL_dS, L) ])
#        dL_dchol = choleskies.triang_to_flat(dL_dchol)
#        
#        #sum (gradients of) expected likelihood and KL part w.r.t global parameters    
#        dL_dgm, dL_dgS, dL_dgKmm, dL_dgKmn = dF_dgm, dF_dgS, dF_dgKmm, dF_dgKmn
#        
#        dL_dgchol = 2.*np.array([np.dot(a,b) for a, b in zip(dL_dgS, g_L) ])
#        #dL_dgchol = choleskies.triang_to_flat(dL_dgchol)
#
#
#        grad_dict = {'dL_dKmm':dL_dKmm, 'dL_dKmn':dL_dKmn, 'dL_dKdiag': dF_dv.sum(1), 'dL_dm':dL_dm, 'dL_dchol':dL_dchol, 'dL_dthetaL':dF_dthetaL}
#        g_grad_dict = {'dL_dKmm':dL_dgKmm, 'dL_dKmn':dL_dgKmn, 'dL_dKdiag': 0, 'dL_dm':dL_dgm, 'dL_dchol':dL_dgchol, 'dL_dthetaL':None}
##        if mean_function is not None:
##            grad_dict['dL_dmfZ'] = dF_dmfZ - dKL_dmfZ
##            grad_dict['dL_dmfX'] = dF_dmfX
#        return Posterior(mean=q_u_mean, cov=S.T, K=Kmm, prior_mean=prior_mean_u), log_marginal, grad_dict, g_grad_dict
        
    def global_inference(self, q_u_mean, q_u_chol, kern, X, Z, likelihood, Y, mean_function=None, Y_metadata=None, KL_scale=1.0, batch_scale=1.0):

        num_data, _ = Y.shape
        num_inducing, num_outputs = q_u_mean.shape

        #expand cholesky representation
        L = choleskies.flat_to_triang(q_u_chol)


        S = np.empty((num_outputs, num_inducing, num_inducing))
        [np.dot(L[i,:,:], L[i,:,:].T, S[i,:,:]) for i in range(num_outputs)]
        #Si,_ = linalg.dpotri(np.asfortranarray(L), lower=1)
        Si = choleskies.multiple_dpotri(L)
        logdetS = np.array([2.*np.sum(np.log(np.abs(np.diag(L[i,:,:])))) for i in range(L.shape[0])])

        if np.any(np.isinf(Si)):
            raise ValueError("Cholesky representation unstable")

        #compute mean function stuff
        if mean_function is not None:
            prior_mean_u = mean_function.f(Z)
            prior_mean_f = mean_function.f(X)
        else:
            prior_mean_u = np.zeros((num_inducing, num_outputs))
            prior_mean_f = np.zeros((num_data, num_outputs))

        #compute kernel related stuff
        Kmm = kern.K(Z)
        Kmn = kern.K(Z, X)
        Knn_diag = kern.Kdiag(X)
        Lm = linalg.jitchol(Kmm)
        logdetKmm = 2.*np.sum(np.log(np.diag(Lm)))
        Kmmi, _ = linalg.dpotri(Lm)

        #compute the marginal means and variances of q(f)
        A, _ = linalg.dpotrs(Lm, Kmn)
        mu = prior_mean_f + np.dot(A.T, q_u_mean - prior_mean_u)
        v = np.empty((num_data, num_outputs))
        for i in range(num_outputs):
            tmp = dtrmm(1.0,L[i].T, A, lower=0, trans_a=0)
            v[:,i] = np.sum(np.square(tmp),0)
        partial_v=v.copy()
        v += (Knn_diag - np.sum(A*Kmn,0))[:,None]

        #compute the KL term
        Kmmim = np.dot(Kmmi, q_u_mean)
        KLs = -0.5*logdetS -0.5*num_inducing + 0.5*logdetKmm + 0.5*np.sum(Kmmi[None,:,:]*S,1).sum(1) + 0.5*np.sum(q_u_mean*Kmmim,0)
        KL = KLs.sum()
        #gradient of the KL term (assuming zero mean function)
        dKL_dm = Kmmim.copy()
        dKL_dS = 0.5*(Kmmi[None,:,:] - Si)
        dKL_dKmm = 0.5*num_outputs*Kmmi - 0.5*Kmmi.dot(S.sum(0)).dot(Kmmi) - 0.5*Kmmim.dot(Kmmim.T)

        if mean_function is not None:
            #adjust KL term for mean function
            Kmmi_mfZ = np.dot(Kmmi, prior_mean_u)
            KL += -np.sum(q_u_mean*Kmmi_mfZ)
            KL += 0.5*np.sum(Kmmi_mfZ*prior_mean_u)

            #adjust gradient for mean fucntion
            dKL_dm -= Kmmi_mfZ
            dKL_dKmm += Kmmim.dot(Kmmi_mfZ.T)
            dKL_dKmm -= 0.5*Kmmi_mfZ.dot(Kmmi_mfZ.T)

            #compute gradients for mean_function
            dKL_dmfZ = Kmmi_mfZ - Kmmim

        #quadrature for the likelihood
        F, dF_dmu, dF_dv, dF_dthetaL = likelihood.variational_expectations(Y, mu, v, Y_metadata=Y_metadata)

        #rescale the F term if working on a batch
        F, dF_dmu, dF_dv =  F*batch_scale, dF_dmu*batch_scale, dF_dv*batch_scale
        if dF_dthetaL is not None:
            dF_dthetaL =  dF_dthetaL.sum(1).sum(1)*batch_scale

        #derivatives of expected likelihood, assuming zero mean function
        Adv = A[None,:,:]*dF_dv.T[:,None,:] # As if dF_Dv is diagonal, D, M, N
        Admu = A.dot(dF_dmu)
        Adv = np.ascontiguousarray(Adv) # makes for faster operations later...(inc dsymm)
        AdvA = np.dot(Adv.reshape(-1, num_data),A.T).reshape(num_outputs, num_inducing, num_inducing )
        tmp = np.sum([np.dot(a,s) for a, s in zip(AdvA, S)],0).dot(Kmmi)
        dF_dKmm = -Admu.dot(Kmmim.T) + AdvA.sum(0) - tmp - tmp.T
        dF_dKmm = 0.5*(dF_dKmm + dF_dKmm.T) # necessary? GPy bug?
        tmp = S.reshape(-1, num_inducing).dot(Kmmi).reshape(num_outputs, num_inducing , num_inducing )
        tmp = 2.*(tmp - np.eye(num_inducing)[None, :,:])

        dF_dKmn = Kmmim.dot(dF_dmu.T)
        for a,b in zip(tmp, Adv):
            dF_dKmn += np.dot(a.T, b)

        dF_dm = Admu
        dF_dS = AdvA

        #adjust gradient to account for mean function
        if mean_function is not None:
            dF_dmfX = dF_dmu.copy()
            dF_dmfZ = -Admu
            dF_dKmn -= np.dot(Kmmi_mfZ, dF_dmu.T)
            dF_dKmm += Admu.dot(Kmmi_mfZ.T)


        #sum (gradients of) expected likelihood and KL part
        log_marginal = F.sum() - KL
        dL_dm, dL_dS, dL_dKmm, dL_dKmn = dF_dm - dKL_dm, dF_dS- dKL_dS, dF_dKmm- dKL_dKmm, dF_dKmn

        dL_dchol = 2.*np.array([np.dot(a,b) for a, b in zip(dL_dS, L) ])
        dL_dchol = choleskies.triang_to_flat(dL_dchol)

        grad_dict = {'dL_dKmm':dL_dKmm, 'dL_dKmn':dL_dKmn, 'dL_dKdiag': dF_dv.sum(1), 'dL_dm':dL_dm, 'dL_dchol':dL_dchol, 'dL_dthetaL':dF_dthetaL}
        if mean_function is not None:
            grad_dict['dL_dmfZ'] = dF_dmfZ - dKL_dmfZ
            grad_dict['dL_dmfX'] = dF_dmfX
        return Posterior(mean=q_u_mean, cov=S.T, K=Kmm, prior_mean=prior_mean_u), log_marginal, grad_dict, mu,partial_v, A, L, S, Kmmi, Kmmim      
        
    def local_inference(self,  X_full, Y_full, rho_full, q_u_mean, q_u_chol, kern, Z, likelihood,  mean_function,  Y_metadata=None, KL_scale=1.0, batch_scale=1.0):
        
        index = np.where(rho_full>0)[0]   
        X = X_full[index, :]
        Y = Y_full[index, :]
        rho = rho_full [index, :]
#        g_mu= g_mu_full[index,:]
#        g_v= g_v_full[index,:]
#        g_A = g_A_full[:,index]
        
        num_data, _ = Y.shape
        num_data_full, _ = Y_full.shape
        
#        g_num_inducing = g_Kmmi.shape[0]
#        
#        #expand cholesky representation
#        g_L = choleskies.flat_to_triang(g_q_u_chol)
#
#
#        g_S = np.empty((num_outputs, g_num_inducing, g_num_inducing))
#        [np.dot(g_L[i,:,:], g_L[i,:,:].T, g_S[i,:,:]) for i in range(num_outputs)]
#        #Si,_ = linalg.dpotri(np.asfortranarray(L), lower=1)
#        g_Si = choleskies.multiple_dpotri(g_L)
#        #g_logdetS = np.array([2.*np.sum(np.log(np.abs(np.diag(g_L[i,:,:])))) for i in range(g_L.shape[0])])
#
#        if np.any(np.isinf(g_Si)):
#            raise ValueError("Cholesky representation unstable")
#            
#        if g_mean_function is not None:
#            g_prior_mean_u = g_mean_function.f(g_Z)
#            g_prior_mean_f = g_mean_function.f(X)
#        else:
#            g_prior_mean_u = np.zeros((g_num_inducing, num_outputs))
#            g_prior_mean_f = np.zeros((num_data, num_outputs))
#            
#        g_Kmm = g_kern.K(g_Z)
#        g_Kmn = g_kern.K(g_Z, X)
#        #g_Knn_diag = g_kern.Kdiag(X)
#        g_Lm = linalg.jitchol(g_Kmm)
#        #g_logdetKmm = 2.*np.sum(np.log(np.diag(g_Lm)))
#        g_Kmmi, _ = linalg.dpotri(g_Lm)
#                    
#        g_A, _ = linalg.dpotrs(g_Lm, g_Kmn)
#        g_mu = g_prior_mean_f + np.dot(g_A.T, g_q_u_mean - g_prior_mean_u)
#        g_v = np.empty((num_data, num_outputs))
#        for i in range(num_outputs):
#            tmp = dtrmm(1.0,g_L[i].T, g_A, lower=0, trans_a=0)
#            g_v[:,i] = np.sum(np.square(tmp),0)
#        #Anh v += (Knn_diag - np.sum(A*Kmn,0))[:,None]
#        
#        g_Kmmim = np.dot(g_Kmmi, g_q_u_mean);
            
        
        num_inducing, num_outputs = q_u_mean.shape

        #expand cholesky representation
        L = choleskies.flat_to_triang(q_u_chol)


        S = np.empty((num_outputs, num_inducing, num_inducing))
        [np.dot(L[i,:,:], L[i,:,:].T, S[i,:,:]) for i in range(num_outputs)]
        #Si,_ = linalg.dpotri(np.asfortranarray(L), lower=1)
        Si = choleskies.multiple_dpotri(L)
        logdetS = np.array([2.*np.sum(np.log(np.abs(np.diag(L[i,:,:])))) for i in range(L.shape[0])])
        
        if np.any(np.isinf(Si)):
            raise ValueError("Cholesky representation unstable")
            
        

        #compute mean function stuff
#        if mean_function is not None:
#            prior_mean_u = mean_function.f(Z)
#            prior_mean_f = mean_function.f(X)
#        else:
#            prior_mean_u = np.zeros((num_inducing, num_outputs))
#            prior_mean_f = np.zeros((num_data, num_outputs))

        prior_mean_u = np.zeros((num_inducing, num_outputs))
        prior_mean_f = np.zeros((num_data, num_outputs))
            
        #compute kernel related stuff
        Kmm = kern.K(Z)
        Kmn = kern.K(Z, X)
        Knn_diag = kern.Kdiag(X)
        Lm = linalg.jitchol(Kmm)
        logdetKmm = 2.*np.sum(np.log(np.diag(Lm)))
        Kmmi, _ = linalg.dpotri(Lm)
        


        #compute the marginal means and variances of q(f)
        A, _ = linalg.dpotrs(Lm, Kmn)
        mu = prior_mean_f + np.dot(A.T, q_u_mean - prior_mean_u)
        v = np.empty((num_data, num_outputs))
        for i in range(num_outputs):
            tmp = dtrmm(1.0,L[i].T, A, lower=0, trans_a=0)
            v[:,i] = np.sum(np.square(tmp),0)
        v += (Knn_diag - np.sum(A*Kmn,0))[:,None]
        
        

        
        #final marginal means and variances of q(f)
#        mu += g_mu
#        v += g_v
        
        #compute the KL term
        Kmmim = np.dot(Kmmi, q_u_mean)
        KLs = -0.5*logdetS -0.5*num_inducing + 0.5*logdetKmm + 0.5*np.sum(Kmmi[None,:,:]*S,1).sum(1) + 0.5*np.sum(q_u_mean*Kmmim,0)
        KL = KLs.sum()
        #gradient of the KL term (assuming zero mean function)
        dKL_dm = Kmmim.copy()
        dKL_dS = 0.5*(Kmmi[None,:,:] - Si)
        dKL_dKmm = 0.5*num_outputs*Kmmi - 0.5*Kmmi.dot(S.sum(0)).dot(Kmmi) - 0.5*Kmmim.dot(Kmmim.T)

#        if mean_function is not None:
#            #adjust KL term for mean function
#            Kmmi_mfZ = np.dot(Kmmi, prior_mean_u)
#            KL += -np.sum(q_u_mean*Kmmi_mfZ)
#            KL += 0.5*np.sum(Kmmi_mfZ*prior_mean_u)
#
#            #adjust gradient for mean fucntion
#            dKL_dm -= Kmmi_mfZ
#            dKL_dKmm += Kmmim.dot(Kmmi_mfZ.T)
#            dKL_dKmm -= 0.5*Kmmi_mfZ.dot(Kmmi_mfZ.T)
#
#            #compute gradients for mean_function
#            dKL_dmfZ = Kmmi_mfZ - Kmmim

        #quadrature for the likelihood
        F, dF_dmu, dF_dv, dF_dthetaL = likelihood.variational_expectations(Y, mu, v, Y_metadata=Y_metadata)
        
        #multiply with rho
        F, dF_dmu, dF_dv =  F*rho, dF_dmu*rho, dF_dv*rho
        
        #rescale the F term if working on a batch
        F, dF_dmu, dF_dv =  F*batch_scale, dF_dmu*batch_scale, dF_dv*batch_scale
        if dF_dthetaL is not None:
            dF_dthetaL =  dF_dthetaL.sum(1).sum(1)*batch_scale

        #derivatives of expected likelihood w.r.t. global parameters, assuming zero mean function
#        g_Adv = g_A[None,:,:]*dF_dv.T[:,None,:] # As if dF_Dv is diagonal, D, M, N
#        g_Admu = g_A.dot(dF_dmu)
#        g_Adv = np.ascontiguousarray(g_Adv) # makes for faster operations later...(inc dsymm)
#        g_AdvA = np.dot(g_Adv.reshape(-1, num_data),g_A.T).reshape(num_outputs, g_num_inducing, g_num_inducing )
#        tmp = np.sum([np.dot(a,s) for a, s in zip(g_AdvA, g_S)],0).dot(g_Kmmi)
#        #dF_dgKmm = -g_Admu.dot(g_Kmmim.T) + g_AdvA.sum(0) - tmp - tmp.T
#        dF_dgKmm = -g_Admu.dot(g_Kmmim.T)  - tmp - tmp.T
#        dF_dgKmm = 0.5*(dF_dgKmm + dF_dgKmm.T) # necessary? GPy bug?
#        tmp = g_S.reshape(-1, g_num_inducing).dot(g_Kmmi).reshape(num_outputs, g_num_inducing , g_num_inducing )
#        #tmp = 2.*(tmp - np.eye(g_num_inducing)[None, :,:])
#        tmp = 2.*tmp
#
#        dF_dgKmn = g_Kmmim.dot(dF_dmu.T)
#        for a,b in zip(tmp, g_Adv):
#            dF_dgKmn += np.dot(a.T, b)
#
#        dF_dgm = g_Admu
#        dF_dgS = g_AdvA
        
        #derivatives of expected likelihood w.r.t. local parameters, assuming zero mean function
        Adv = A[None,:,:]*dF_dv.T[:,None,:] # As if dF_Dv is diagonal, D, M, N
        Admu = A.dot(dF_dmu)
        Adv = np.ascontiguousarray(Adv) # makes for faster operations later...(inc dsymm)
        AdvA = np.dot(Adv.reshape(-1, num_data),A.T).reshape(num_outputs, num_inducing, num_inducing )
        tmp = np.sum([np.dot(a,s) for a, s in zip(AdvA, S)],0).dot(Kmmi)
        dF_dKmm = -Admu.dot(Kmmim.T) + AdvA.sum(0) - tmp - tmp.T
        dF_dKmm = 0.5*(dF_dKmm + dF_dKmm.T) # necessary? GPy bug?
        tmp = S.reshape(-1, num_inducing).dot(Kmmi).reshape(num_outputs, num_inducing , num_inducing )
        tmp = 2.*(tmp - np.eye(num_inducing)[None, :,:])

        dF_dKmn = Kmmim.dot(dF_dmu.T)
        for a,b in zip(tmp, Adv):
            dF_dKmn += np.dot(a.T, b)

        dF_dm = Admu
        dF_dS = AdvA

        #adjust gradient to account for mean function
#        if mean_function is not None:
#            dF_dmfX = dF_dmu.copy()
#            dF_dmfZ = -Admu
#            dF_dKmn -= np.dot(Kmmi_mfZ, dF_dmu.T)
#            dF_dKmm += Admu.dot(Kmmi_mfZ.T)


        #sum (gradients of) expected likelihood and KL part w.r.t local parameters
        log_marginal = F.sum() - KL
        dL_dm, dL_dS, dL_dKmm, dL_dKmn = dF_dm - dKL_dm, dF_dS- dKL_dS, dF_dKmm- dKL_dKmm, dF_dKmn
    
        dL_dchol = 2.*np.array([np.dot(a,b) for a, b in zip(dL_dS, L) ])
        dL_dchol = choleskies.triang_to_flat(dL_dchol)
        
        #sum (gradients of) expected likelihood and KL part w.r.t global parameters    
#        dL_dgm, dL_dgS, dL_dgKmm, dL_dgKmn = dF_dgm, dF_dgS, dF_dgKmm, dF_dgKmn
#        dL_dgKmn_full=np.zeros([g_num_inducing, num_data_full])
#        dL_dgKmn_full[:, index] = dL_dgKmn
#        
#        dL_dgchol = 2.*np.array([np.dot(a,b) for a, b in zip(dL_dgS, g_L) ])
        #dL_dgchol = choleskies.triang_to_flat(dL_dgchol)


        grad_dict = {'dL_dKmm':dL_dKmm, 'dL_dKmn':dL_dKmn, 'dL_dKdiag': dF_dv.sum(1), 'dL_dm':dL_dm, 'dL_dchol':dL_dchol, 'dL_dthetaL':dF_dthetaL}
#        g_grad_dict = {'dL_dKmm':dL_dgKmm, 'dL_dKmn':dL_dgKmn_full, 'dL_dKdiag': 0, 'dL_dm':dL_dgm, 'dL_dchol':dL_dgchol, 'dL_dthetaL':None}
#        if mean_function is not None:
#            grad_dict['dL_dmfZ'] = dF_dmfZ - dKL_dmfZ
#            grad_dict['dL_dmfX'] = dF_dmfX
        return Posterior(mean=q_u_mean, cov=S.T, K=Kmm, prior_mean=prior_mean_u), log_marginal, grad_dict,index
                
        
    def local_likelihood(self,  X, Y, l_svgps,  g_q_u_mean, g_q_u_chol, g_kern, g_Z, g_mean_function=None):

        num_data, _ = Y.shape
        F=np.zeros([num_data, len(l_svgps)])
        
        #global
        g_num_inducing, num_outputs = g_q_u_mean.shape
        
        #expand cholesky representation
        g_L = choleskies.flat_to_triang(g_q_u_chol)


        g_S = np.empty((num_outputs, g_num_inducing, g_num_inducing))
        [np.dot(g_L[i,:,:], g_L[i,:,:].T, g_S[i,:,:]) for i in range(num_outputs)]


        if g_mean_function is not None:
            g_prior_mean_u = g_mean_function.f(g_Z)
            g_prior_mean_f = g_mean_function.f(X)
        else:
            g_prior_mean_u = np.zeros((g_num_inducing, num_outputs))
            g_prior_mean_f = np.zeros((num_data, num_outputs))
        
        g_Kmm = g_kern.K(g_Z)
        g_Kmn = g_kern.K(g_Z, X)
        #g_Knn_diag = g_kern.Kdiag(X)
        g_Lm = linalg.jitchol(g_Kmm)        
        
        g_A, _ = linalg.dpotrs(g_Lm, g_Kmn)
        g_mu = g_prior_mean_f + np.dot(g_A.T, g_q_u_mean - g_prior_mean_u)
        g_v = np.empty((num_data, num_outputs))
        for i in range(num_outputs):
            tmp = dtrmm(1.0,g_L[i].T, g_A, lower=0, trans_a=0)
            g_v[:,i] = np.sum(np.square(tmp),0)
        #Anh v += (Knn_diag - np.sum(A*Kmn,0))[:,None]
            
        for i in range(len(l_svgps)):
            num_inducing, num_outputs = l_svgps[i].q_u_mean.shape
    
            #expand cholesky representation
            L = choleskies.flat_to_triang(l_svgps[i].q_u_chol)
    
    
            S = np.empty((num_outputs, num_inducing, num_inducing))
            [np.dot(L[i,:,:], L[i,:,:].T, S[i,:,:]) for i in range(num_outputs)]
               
            
            prior_mean_u = np.zeros((num_inducing, num_outputs))
            prior_mean_f = np.zeros((num_data, num_outputs))
                
    
                
    
            #compute kernel related stuff
            Kmm = l_svgps[i].kern.K(l_svgps[i].Z)
            Kmn = l_svgps[i].kern.K(l_svgps[i].Z, X)
            Knn_diag = l_svgps[i].kern.Kdiag(X)
            Lm = linalg.jitchol(Kmm)
            #Kmmi, _ = linalg.dpotri(Lm)
            
    
            
    
            #compute the marginal means and variances of q(f)
            A, _ = linalg.dpotrs(Lm, Kmn)
            mu = prior_mean_f + np.dot(A.T, l_svgps[i].q_u_mean - prior_mean_u)
            v = np.empty((num_data, num_outputs))
            for i in range(num_outputs):
                tmp = dtrmm(1.0,L[i].T, A, lower=0, trans_a=0)
                v[:,i] = np.sum(np.square(tmp),0)
            v += (Knn_diag - np.sum(A*Kmn,0))[:,None]
            
                    
            #final marginal means and variances of q(f)
            mu += g_mu
            v += g_v

            #quadrature for the likelihood
            F[:,i,None], dF_dmu, dF_dv, dF_dthetaL = l_svgps[i].likelihood.variational_expectations(Y, mu, v, Y_metadata=l_svgps[i].Y_metadata)
        
        return F# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 16:08:19 2016

@author: tnan287
"""

from . import LatentFunctionInference
from ...util import linalg
from ...util import choleskies
import numpy as np
from .posterior import Posterior
from scipy.linalg.blas import dgemm, dsymm, dtrmm
from .svgp import SVGP as svgp_inf
#from local_svgp import Local_SVGP
from collections import Counter

class SMGP_L(LatentFunctionInference):

    def inference(self, q_u_mean, q_u_chol, kern, Z, likelihood,  mean_function,Y_metadata, l_svgps, X, Y, rho,  KL_scale=1.0, batch_scale=1.0):

        num_data, _ = Y.shape
        num_inducing, num_outputs = q_u_mean.shape
        g_log_marginal_likelihood=0;
        
        #g_posterior, g_log_marginal_likelihood, g_grad_dict, g_mu, g_v, g_A, g_L, g_S, g_Kmmi, g_Kmmim = self.global_inference(q_u_mean, q_u_chol, kern, X, Z, likelihood, Y, mean_function, Y_metadata, KL_scale=KL_scale, batch_scale=batch_scale)
        
#        g_grad_dict['dL_dchol'] = choleskies.flat_to_triang(g_grad_dict['dL_dchol'])
        l_posteriors=[];
        l_grad_dicts =[];
        
        for i in range(len(l_svgps)):
            l_posterior, l_log_marginal, l_grad_dict, index = self.local_inference(X, Y, rho[:, i].reshape(rho.shape[0],1), l_svgps[i].q_u_mean, l_svgps[i].q_u_chol, l_svgps[i].kern, l_svgps[i].Z, l_svgps[i].likelihood,  l_svgps[i].mean_function,   l_svgps[i].Y_metadata, KL_scale, batch_scale)
            l_svgps[i].set_data(X[index,:],Y[index,:])            
            l_posteriors.append(l_posterior)
            l_grad_dicts.append(l_grad_dict)            
            g_log_marginal_likelihood = g_log_marginal_likelihood + l_log_marginal
            #g_grad_dict = dict(Counter(g_grad_dict)+Counter(add_g_grad_dict));
#            g_grad_dict = self.addDicts(g_grad_dict, add_g_grad_dict)

#        g_grad_dict['dL_dchol'] = choleskies.triang_to_flat(g_grad_dict['dL_dchol'])
        
        return  l_posteriors, g_log_marginal_likelihood, l_grad_dicts

    def addDicts(self, a, b):
        sa = set(a.keys())
        sb = set(b.keys())
        sc= set()
        for k in sa&sb:
            if (a[k] is None) or (b[k] is None) :
                sc.add(k)
        c = dict( \
        [(k, a[k]+b[k]) for k in (sa&sb)-sc ] + \
        [(k, a[k]) for k in sa-sb-sc ] + \
        [(k, b[k]) for k in sb-sa-sc ] +\
        [(k, None) for k in sc ] \
        )
        return c
        


#    def local_inference(self,  X, Y, rho, q_u_mean, q_u_chol, kern, Z, likelihood,  mean_function,  g_q_u_mean, g_q_u_chol, g_kern, g_Z, g_likelihood, g_mean_function=None,Y_metadata=None, KL_scale=1.0, batch_scale=1.0):
#
#        num_data, _ = Y.shape
#        num_inducing, num_outputs = q_u_mean.shape
#
#        #expand cholesky representation
#        L = choleskies.flat_to_triang(q_u_chol)
#
#
#        S = np.empty((num_outputs, num_inducing, num_inducing))
#        [np.dot(L[i,:,:], L[i,:,:].T, S[i,:,:]) for i in range(num_outputs)]
#        #Si,_ = linalg.dpotri(np.asfortranarray(L), lower=1)
#        Si = choleskies.multiple_dpotri(L)
#        logdetS = np.array([2.*np.sum(np.log(np.abs(np.diag(L[i,:,:])))) for i in range(L.shape[0])])
#        
#        if np.any(np.isinf(Si)):
#            raise ValueError("Cholesky representation unstable")
#            
#        g_num_inducing, num_outputs = g_q_u_mean.shape
#        
#        #expand cholesky representation
#        g_L = choleskies.flat_to_triang(g_q_u_chol)
#
#
#        g_S = np.empty((num_outputs, g_num_inducing, g_num_inducing))
#        [np.dot(g_L[i,:,:], g_L[i,:,:].T, g_S[i,:,:]) for i in range(num_outputs)]
#        #Si,_ = linalg.dpotri(np.asfortranarray(L), lower=1)
#        g_Si = choleskies.multiple_dpotri(g_L)
#        #g_logdetS = np.array([2.*np.sum(np.log(np.abs(np.diag(g_L[i,:,:])))) for i in range(g_L.shape[0])])
#        
#
#        if np.any(np.isinf(g_Si)):
#            raise ValueError("Cholesky representation unstable")
#
#        #compute mean function stuff
##        if mean_function is not None:
##            prior_mean_u = mean_function.f(Z)
##            prior_mean_f = mean_function.f(X)
##        else:
##            prior_mean_u = np.zeros((num_inducing, num_outputs))
##            prior_mean_f = np.zeros((num_data, num_outputs))
#
#        prior_mean_u = np.zeros((num_inducing, num_outputs))
#        prior_mean_f = np.zeros((num_data, num_outputs))
#            
#        if g_mean_function is not None:
#            g_prior_mean_u = g_mean_function.f(g_Z)
#            g_prior_mean_f = g_mean_function.f(X)
#        else:
#            g_prior_mean_u = np.zeros((g_num_inducing, num_outputs))
#            g_prior_mean_f = np.zeros((num_data, num_outputs))
#            
#
#        #compute kernel related stuff
#        Kmm = kern.K(Z)
#        Kmn = kern.K(Z, X)
#        Knn_diag = kern.Kdiag(X)
#        Lm = linalg.jitchol(Kmm)
#        logdetKmm = 2.*np.sum(np.log(np.diag(Lm)))
#        Kmmi, _ = linalg.dpotri(Lm)
#        
#        g_Kmm = g_kern.K(g_Z)
#        g_Kmn = g_kern.K(g_Z, X)
#        #g_Knn_diag = g_kern.Kdiag(X)
#        g_Lm = linalg.jitchol(g_Kmm)
#        #g_logdetKmm = 2.*np.sum(np.log(np.diag(g_Lm)))
#        g_Kmmi, _ = linalg.dpotri(g_Lm)
#        
#
#        #compute the marginal means and variances of q(f)
#        A, _ = linalg.dpotrs(Lm, Kmn)
#        mu = prior_mean_f + np.dot(A.T, q_u_mean - prior_mean_u)
#        v = np.empty((num_data, num_outputs))
#        for i in range(num_outputs):
#            tmp = dtrmm(1.0,L[i].T, A, lower=0, trans_a=0)
#            v[:,i] = np.sum(np.square(tmp),0)
#        v += (Knn_diag - np.sum(A*Kmn,0))[:,None]
#        
#        
#        g_A, _ = linalg.dpotrs(g_Lm, g_Kmn)
#        g_mu = g_prior_mean_f + np.dot(g_A.T, g_q_u_mean - g_prior_mean_u)
#        g_v = np.empty((num_data, num_outputs))
#        for i in range(num_outputs):
#            tmp = dtrmm(1.0,g_L[i].T, g_A, lower=0, trans_a=0)
#            g_v[:,i] = np.sum(np.square(tmp),0)
#        #Anh v += (Knn_diag - np.sum(A*Kmn,0))[:,None]
#        
#        #final marginal means and variances of q(f)
#        mu += g_mu
#        v += g_v
#        
#        #compute the KL term
#        Kmmim = np.dot(Kmmi, q_u_mean)
#        KLs = -0.5*logdetS -0.5*num_inducing + 0.5*logdetKmm + 0.5*np.sum(Kmmi[None,:,:]*S,1).sum(1) + 0.5*np.sum(q_u_mean*Kmmim,0)
#        KL = KLs.sum()
#        #gradient of the KL term (assuming zero mean function)
#        dKL_dm = Kmmim.copy()
#        dKL_dS = 0.5*(Kmmi[None,:,:] - Si)
#        dKL_dKmm = 0.5*num_outputs*Kmmi - 0.5*Kmmi.dot(S.sum(0)).dot(Kmmi) - 0.5*Kmmim.dot(Kmmim.T)
#
##        if mean_function is not None:
##            #adjust KL term for mean function
##            Kmmi_mfZ = np.dot(Kmmi, prior_mean_u)
##            KL += -np.sum(q_u_mean*Kmmi_mfZ)
##            KL += 0.5*np.sum(Kmmi_mfZ*prior_mean_u)
##
##            #adjust gradient for mean fucntion
##            dKL_dm -= Kmmi_mfZ
##            dKL_dKmm += Kmmim.dot(Kmmi_mfZ.T)
##            dKL_dKmm -= 0.5*Kmmi_mfZ.dot(Kmmi_mfZ.T)
##
##            #compute gradients for mean_function
##            dKL_dmfZ = Kmmi_mfZ - Kmmim
#
#        #quadrature for the likelihood
#        F, dF_dmu, dF_dv, dF_dthetaL = likelihood.variational_expectations(Y, mu, v, Y_metadata=Y_metadata)
#        
#        #multiply with rho
#        F, dF_dmu, dF_dv =  F*rho, dF_dmu*rho, dF_dv*rho
#        
#        #rescale the F term if working on a batch
#        F, dF_dmu, dF_dv =  F*batch_scale, dF_dmu*batch_scale, dF_dv*batch_scale
#        if dF_dthetaL is not None:
#            dF_dthetaL =  dF_dthetaL.sum(1).sum(1)*batch_scale
#
#        #derivatives of expected likelihood w.r.t. global parameters, assuming zero mean function
#        g_Kmmim = np.dot(g_Kmmi, g_q_u_mean)
#        g_Adv = g_A[None,:,:]*dF_dv.T[:,None,:] # As if dF_Dv is diagonal, D, M, N
#        g_Admu = g_A.dot(dF_dmu)
#        g_Adv = np.ascontiguousarray(g_Adv) # makes for faster operations later...(inc dsymm)
#        g_AdvA = np.dot(g_Adv.reshape(-1, num_data),g_A.T).reshape(num_outputs, g_num_inducing, g_num_inducing )
#        tmp = np.sum([np.dot(a,s) for a, s in zip(g_AdvA, g_S)],0).dot(g_Kmmi)
#        #dF_dgKmm = -g_Admu.dot(g_Kmmim.T) + g_AdvA.sum(0) - tmp - tmp.T
#        dF_dgKmm = -g_Admu.dot(g_Kmmim.T)  - tmp - tmp.T
#        dF_dgKmm = 0.5*(dF_dgKmm + dF_dgKmm.T) # necessary? GPy bug?
#        tmp = g_S.reshape(-1, g_num_inducing).dot(g_Kmmi).reshape(num_outputs, g_num_inducing , g_num_inducing )
#        #tmp = 2.*(tmp - np.eye(g_num_inducing)[None, :,:])
#        tmp = 2.*tmp
#
#        dF_dgKmn = g_Kmmim.dot(dF_dmu.T)
#        for a,b in zip(tmp, g_Adv):
#            dF_dgKmn += np.dot(a.T, b)
#
#        dF_dgm = g_Admu
#        dF_dgS = g_AdvA
#        
#        #derivatives of expected likelihood w.r.t. local parameters, assuming zero mean function
#        Adv = A[None,:,:]*dF_dv.T[:,None,:] # As if dF_Dv is diagonal, D, M, N
#        Admu = A.dot(dF_dmu)
#        Adv = np.ascontiguousarray(Adv) # makes for faster operations later...(inc dsymm)
#        AdvA = np.dot(Adv.reshape(-1, num_data),A.T).reshape(num_outputs, num_inducing, num_inducing )
#        tmp = np.sum([np.dot(a,s) for a, s in zip(AdvA, S)],0).dot(Kmmi)
#        dF_dKmm = -Admu.dot(Kmmim.T) + AdvA.sum(0) - tmp - tmp.T
#        dF_dKmm = 0.5*(dF_dKmm + dF_dKmm.T) # necessary? GPy bug?
#        tmp = S.reshape(-1, num_inducing).dot(Kmmi).reshape(num_outputs, num_inducing , num_inducing )
#        tmp = 2.*(tmp - np.eye(num_inducing)[None, :,:])
#
#        dF_dKmn = Kmmim.dot(dF_dmu.T)
#        for a,b in zip(tmp, Adv):
#            dF_dKmn += np.dot(a.T, b)
#
#        dF_dm = Admu
#        dF_dS = AdvA
#
#        #adjust gradient to account for mean function
##        if mean_function is not None:
##            dF_dmfX = dF_dmu.copy()
##            dF_dmfZ = -Admu
##            dF_dKmn -= np.dot(Kmmi_mfZ, dF_dmu.T)
##            dF_dKmm += Admu.dot(Kmmi_mfZ.T)
#
#
#        #sum (gradients of) expected likelihood and KL part w.r.t local parameters
#        log_marginal = F.sum() - KL
#        dL_dm, dL_dS, dL_dKmm, dL_dKmn = dF_dm - dKL_dm, dF_dS- dKL_dS, dF_dKmm- dKL_dKmm, dF_dKmn
#    
#        dL_dchol = 2.*np.array([np.dot(a,b) for a, b in zip(dL_dS, L) ])
#        dL_dchol = choleskies.triang_to_flat(dL_dchol)
#        
#        #sum (gradients of) expected likelihood and KL part w.r.t global parameters    
#        dL_dgm, dL_dgS, dL_dgKmm, dL_dgKmn = dF_dgm, dF_dgS, dF_dgKmm, dF_dgKmn
#        
#        dL_dgchol = 2.*np.array([np.dot(a,b) for a, b in zip(dL_dgS, g_L) ])
#        #dL_dgchol = choleskies.triang_to_flat(dL_dgchol)
#
#
#        grad_dict = {'dL_dKmm':dL_dKmm, 'dL_dKmn':dL_dKmn, 'dL_dKdiag': dF_dv.sum(1), 'dL_dm':dL_dm, 'dL_dchol':dL_dchol, 'dL_dthetaL':dF_dthetaL}
#        g_grad_dict = {'dL_dKmm':dL_dgKmm, 'dL_dKmn':dL_dgKmn, 'dL_dKdiag': 0, 'dL_dm':dL_dgm, 'dL_dchol':dL_dgchol, 'dL_dthetaL':None}
##        if mean_function is not None:
##            grad_dict['dL_dmfZ'] = dF_dmfZ - dKL_dmfZ
##            grad_dict['dL_dmfX'] = dF_dmfX
#        return Posterior(mean=q_u_mean, cov=S.T, K=Kmm, prior_mean=prior_mean_u), log_marginal, grad_dict, g_grad_dict
        
    def global_inference(self, q_u_mean, q_u_chol, kern, X, Z, likelihood, Y, mean_function=None, Y_metadata=None, KL_scale=1.0, batch_scale=1.0):

        num_data, _ = Y.shape
        num_inducing, num_outputs = q_u_mean.shape

        #expand cholesky representation
        L = choleskies.flat_to_triang(q_u_chol)


        S = np.empty((num_outputs, num_inducing, num_inducing))
        [np.dot(L[i,:,:], L[i,:,:].T, S[i,:,:]) for i in range(num_outputs)]
        #Si,_ = linalg.dpotri(np.asfortranarray(L), lower=1)
        Si = choleskies.multiple_dpotri(L)
        logdetS = np.array([2.*np.sum(np.log(np.abs(np.diag(L[i,:,:])))) for i in range(L.shape[0])])

        if np.any(np.isinf(Si)):
            raise ValueError("Cholesky representation unstable")

        #compute mean function stuff
        if mean_function is not None:
            prior_mean_u = mean_function.f(Z)
            prior_mean_f = mean_function.f(X)
        else:
            prior_mean_u = np.zeros((num_inducing, num_outputs))
            prior_mean_f = np.zeros((num_data, num_outputs))

        #compute kernel related stuff
        Kmm = kern.K(Z)
        Kmn = kern.K(Z, X)
        Knn_diag = kern.Kdiag(X)
        Lm = linalg.jitchol(Kmm)
        logdetKmm = 2.*np.sum(np.log(np.diag(Lm)))
        Kmmi, _ = linalg.dpotri(Lm)

        #compute the marginal means and variances of q(f)
        A, _ = linalg.dpotrs(Lm, Kmn)
        mu = prior_mean_f + np.dot(A.T, q_u_mean - prior_mean_u)
        v = np.empty((num_data, num_outputs))
        for i in range(num_outputs):
            tmp = dtrmm(1.0,L[i].T, A, lower=0, trans_a=0)
            v[:,i] = np.sum(np.square(tmp),0)
        partial_v=v.copy()
        v += (Knn_diag - np.sum(A*Kmn,0))[:,None]

        #compute the KL term
        Kmmim = np.dot(Kmmi, q_u_mean)
        KLs = -0.5*logdetS -0.5*num_inducing + 0.5*logdetKmm + 0.5*np.sum(Kmmi[None,:,:]*S,1).sum(1) + 0.5*np.sum(q_u_mean*Kmmim,0)
        KL = KLs.sum()
        #gradient of the KL term (assuming zero mean function)
        dKL_dm = Kmmim.copy()
        dKL_dS = 0.5*(Kmmi[None,:,:] - Si)
        dKL_dKmm = 0.5*num_outputs*Kmmi - 0.5*Kmmi.dot(S.sum(0)).dot(Kmmi) - 0.5*Kmmim.dot(Kmmim.T)

        if mean_function is not None:
            #adjust KL term for mean function
            Kmmi_mfZ = np.dot(Kmmi, prior_mean_u)
            KL += -np.sum(q_u_mean*Kmmi_mfZ)
            KL += 0.5*np.sum(Kmmi_mfZ*prior_mean_u)

            #adjust gradient for mean fucntion
            dKL_dm -= Kmmi_mfZ
            dKL_dKmm += Kmmim.dot(Kmmi_mfZ.T)
            dKL_dKmm -= 0.5*Kmmi_mfZ.dot(Kmmi_mfZ.T)

            #compute gradients for mean_function
            dKL_dmfZ = Kmmi_mfZ - Kmmim

        #quadrature for the likelihood
        F, dF_dmu, dF_dv, dF_dthetaL = likelihood.variational_expectations(Y, mu, v, Y_metadata=Y_metadata)

        #rescale the F term if working on a batch
        F, dF_dmu, dF_dv =  F*batch_scale, dF_dmu*batch_scale, dF_dv*batch_scale
        if dF_dthetaL is not None:
            dF_dthetaL =  dF_dthetaL.sum(1).sum(1)*batch_scale

        #derivatives of expected likelihood, assuming zero mean function
        Adv = A[None,:,:]*dF_dv.T[:,None,:] # As if dF_Dv is diagonal, D, M, N
        Admu = A.dot(dF_dmu)
        Adv = np.ascontiguousarray(Adv) # makes for faster operations later...(inc dsymm)
        AdvA = np.dot(Adv.reshape(-1, num_data),A.T).reshape(num_outputs, num_inducing, num_inducing )
        tmp = np.sum([np.dot(a,s) for a, s in zip(AdvA, S)],0).dot(Kmmi)
        dF_dKmm = -Admu.dot(Kmmim.T) + AdvA.sum(0) - tmp - tmp.T
        dF_dKmm = 0.5*(dF_dKmm + dF_dKmm.T) # necessary? GPy bug?
        tmp = S.reshape(-1, num_inducing).dot(Kmmi).reshape(num_outputs, num_inducing , num_inducing )
        tmp = 2.*(tmp - np.eye(num_inducing)[None, :,:])

        dF_dKmn = Kmmim.dot(dF_dmu.T)
        for a,b in zip(tmp, Adv):
            dF_dKmn += np.dot(a.T, b)

        dF_dm = Admu
        dF_dS = AdvA

        #adjust gradient to account for mean function
        if mean_function is not None:
            dF_dmfX = dF_dmu.copy()
            dF_dmfZ = -Admu
            dF_dKmn -= np.dot(Kmmi_mfZ, dF_dmu.T)
            dF_dKmm += Admu.dot(Kmmi_mfZ.T)


        #sum (gradients of) expected likelihood and KL part
        log_marginal = F.sum() - KL
        dL_dm, dL_dS, dL_dKmm, dL_dKmn = dF_dm - dKL_dm, dF_dS- dKL_dS, dF_dKmm- dKL_dKmm, dF_dKmn

        dL_dchol = 2.*np.array([np.dot(a,b) for a, b in zip(dL_dS, L) ])
        dL_dchol = choleskies.triang_to_flat(dL_dchol)

        grad_dict = {'dL_dKmm':dL_dKmm, 'dL_dKmn':dL_dKmn, 'dL_dKdiag': dF_dv.sum(1), 'dL_dm':dL_dm, 'dL_dchol':dL_dchol, 'dL_dthetaL':dF_dthetaL}
        if mean_function is not None:
            grad_dict['dL_dmfZ'] = dF_dmfZ - dKL_dmfZ
            grad_dict['dL_dmfX'] = dF_dmfX
        return Posterior(mean=q_u_mean, cov=S.T, K=Kmm, prior_mean=prior_mean_u), log_marginal, grad_dict, mu,partial_v, A, L, S, Kmmi, Kmmim      
        
    def local_inference(self,  X_full, Y_full, rho_full, q_u_mean, q_u_chol, kern, Z, likelihood,  mean_function,  Y_metadata=None, KL_scale=1.0, batch_scale=1.0):
        
        index = np.where(rho_full>0)[0]   
        X = X_full[index, :]
        Y = Y_full[index, :]
        rho = rho_full [index, :]
#        g_mu= g_mu_full[index,:]
#        g_v= g_v_full[index,:]
#        g_A = g_A_full[:,index]
        
        num_data, _ = Y.shape
        num_data_full, _ = Y_full.shape
        
#        g_num_inducing = g_Kmmi.shape[0]
#        
#        #expand cholesky representation
#        g_L = choleskies.flat_to_triang(g_q_u_chol)
#
#
#        g_S = np.empty((num_outputs, g_num_inducing, g_num_inducing))
#        [np.dot(g_L[i,:,:], g_L[i,:,:].T, g_S[i,:,:]) for i in range(num_outputs)]
#        #Si,_ = linalg.dpotri(np.asfortranarray(L), lower=1)
#        g_Si = choleskies.multiple_dpotri(g_L)
#        #g_logdetS = np.array([2.*np.sum(np.log(np.abs(np.diag(g_L[i,:,:])))) for i in range(g_L.shape[0])])
#
#        if np.any(np.isinf(g_Si)):
#            raise ValueError("Cholesky representation unstable")
#            
#        if g_mean_function is not None:
#            g_prior_mean_u = g_mean_function.f(g_Z)
#            g_prior_mean_f = g_mean_function.f(X)
#        else:
#            g_prior_mean_u = np.zeros((g_num_inducing, num_outputs))
#            g_prior_mean_f = np.zeros((num_data, num_outputs))
#            
#        g_Kmm = g_kern.K(g_Z)
#        g_Kmn = g_kern.K(g_Z, X)
#        #g_Knn_diag = g_kern.Kdiag(X)
#        g_Lm = linalg.jitchol(g_Kmm)
#        #g_logdetKmm = 2.*np.sum(np.log(np.diag(g_Lm)))
#        g_Kmmi, _ = linalg.dpotri(g_Lm)
#                    
#        g_A, _ = linalg.dpotrs(g_Lm, g_Kmn)
#        g_mu = g_prior_mean_f + np.dot(g_A.T, g_q_u_mean - g_prior_mean_u)
#        g_v = np.empty((num_data, num_outputs))
#        for i in range(num_outputs):
#            tmp = dtrmm(1.0,g_L[i].T, g_A, lower=0, trans_a=0)
#            g_v[:,i] = np.sum(np.square(tmp),0)
#        #Anh v += (Knn_diag - np.sum(A*Kmn,0))[:,None]
#        
#        g_Kmmim = np.dot(g_Kmmi, g_q_u_mean);
            
        
        num_inducing, num_outputs = q_u_mean.shape

        #expand cholesky representation
        L = choleskies.flat_to_triang(q_u_chol)


        S = np.empty((num_outputs, num_inducing, num_inducing))
        [np.dot(L[i,:,:], L[i,:,:].T, S[i,:,:]) for i in range(num_outputs)]
        #Si,_ = linalg.dpotri(np.asfortranarray(L), lower=1)
        Si = choleskies.multiple_dpotri(L)
        logdetS = np.array([2.*np.sum(np.log(np.abs(np.diag(L[i,:,:])))) for i in range(L.shape[0])])
        
        if np.any(np.isinf(Si)):
            raise ValueError("Cholesky representation unstable")
            
        

        #compute mean function stuff
#        if mean_function is not None:
#            prior_mean_u = mean_function.f(Z)
#            prior_mean_f = mean_function.f(X)
#        else:
#            prior_mean_u = np.zeros((num_inducing, num_outputs))
#            prior_mean_f = np.zeros((num_data, num_outputs))

        prior_mean_u = np.zeros((num_inducing, num_outputs))
        prior_mean_f = np.zeros((num_data, num_outputs))
            
        #compute kernel related stuff
        Kmm = kern.K(Z)
        Kmn = kern.K(Z, X)
        Knn_diag = kern.Kdiag(X)
        Lm = linalg.jitchol(Kmm)
        logdetKmm = 2.*np.sum(np.log(np.diag(Lm)))
        Kmmi, _ = linalg.dpotri(Lm)
        


        #compute the marginal means and variances of q(f)
        A, _ = linalg.dpotrs(Lm, Kmn)
        mu = prior_mean_f + np.dot(A.T, q_u_mean - prior_mean_u)
        v = np.empty((num_data, num_outputs))
        for i in range(num_outputs):
            tmp = dtrmm(1.0,L[i].T, A, lower=0, trans_a=0)
            v[:,i] = np.sum(np.square(tmp),0)
        v += (Knn_diag - np.sum(A*Kmn,0))[:,None]
        
        

        
        #final marginal means and variances of q(f)
#        mu += g_mu
#        v += g_v
        
        #compute the KL term
        Kmmim = np.dot(Kmmi, q_u_mean)
        KLs = -0.5*logdetS -0.5*num_inducing + 0.5*logdetKmm + 0.5*np.sum(Kmmi[None,:,:]*S,1).sum(1) + 0.5*np.sum(q_u_mean*Kmmim,0)
        KL = KLs.sum()
        #gradient of the KL term (assuming zero mean function)
        dKL_dm = Kmmim.copy()
        dKL_dS = 0.5*(Kmmi[None,:,:] - Si)
        dKL_dKmm = 0.5*num_outputs*Kmmi - 0.5*Kmmi.dot(S.sum(0)).dot(Kmmi) - 0.5*Kmmim.dot(Kmmim.T)

#        if mean_function is not None:
#            #adjust KL term for mean function
#            Kmmi_mfZ = np.dot(Kmmi, prior_mean_u)
#            KL += -np.sum(q_u_mean*Kmmi_mfZ)
#            KL += 0.5*np.sum(Kmmi_mfZ*prior_mean_u)
#
#            #adjust gradient for mean fucntion
#            dKL_dm -= Kmmi_mfZ
#            dKL_dKmm += Kmmim.dot(Kmmi_mfZ.T)
#            dKL_dKmm -= 0.5*Kmmi_mfZ.dot(Kmmi_mfZ.T)
#
#            #compute gradients for mean_function
#            dKL_dmfZ = Kmmi_mfZ - Kmmim

        #quadrature for the likelihood
        F, dF_dmu, dF_dv, dF_dthetaL = likelihood.variational_expectations(Y, mu, v, Y_metadata=Y_metadata)
        
        #multiply with rho
        F, dF_dmu, dF_dv =  F*rho, dF_dmu*rho, dF_dv*rho
        
        #rescale the F term if working on a batch
        F, dF_dmu, dF_dv =  F*batch_scale, dF_dmu*batch_scale, dF_dv*batch_scale
        if dF_dthetaL is not None:
            dF_dthetaL =  dF_dthetaL.sum(1).sum(1)*batch_scale

        #derivatives of expected likelihood w.r.t. global parameters, assuming zero mean function
#        g_Adv = g_A[None,:,:]*dF_dv.T[:,None,:] # As if dF_Dv is diagonal, D, M, N
#        g_Admu = g_A.dot(dF_dmu)
#        g_Adv = np.ascontiguousarray(g_Adv) # makes for faster operations later...(inc dsymm)
#        g_AdvA = np.dot(g_Adv.reshape(-1, num_data),g_A.T).reshape(num_outputs, g_num_inducing, g_num_inducing )
#        tmp = np.sum([np.dot(a,s) for a, s in zip(g_AdvA, g_S)],0).dot(g_Kmmi)
#        #dF_dgKmm = -g_Admu.dot(g_Kmmim.T) + g_AdvA.sum(0) - tmp - tmp.T
#        dF_dgKmm = -g_Admu.dot(g_Kmmim.T)  - tmp - tmp.T
#        dF_dgKmm = 0.5*(dF_dgKmm + dF_dgKmm.T) # necessary? GPy bug?
#        tmp = g_S.reshape(-1, g_num_inducing).dot(g_Kmmi).reshape(num_outputs, g_num_inducing , g_num_inducing )
#        #tmp = 2.*(tmp - np.eye(g_num_inducing)[None, :,:])
#        tmp = 2.*tmp
#
#        dF_dgKmn = g_Kmmim.dot(dF_dmu.T)
#        for a,b in zip(tmp, g_Adv):
#            dF_dgKmn += np.dot(a.T, b)
#
#        dF_dgm = g_Admu
#        dF_dgS = g_AdvA
        
        #derivatives of expected likelihood w.r.t. local parameters, assuming zero mean function
        Adv = A[None,:,:]*dF_dv.T[:,None,:] # As if dF_Dv is diagonal, D, M, N
        Admu = A.dot(dF_dmu)
        Adv = np.ascontiguousarray(Adv) # makes for faster operations later...(inc dsymm)
        AdvA = np.dot(Adv.reshape(-1, num_data),A.T).reshape(num_outputs, num_inducing, num_inducing )
        tmp = np.sum([np.dot(a,s) for a, s in zip(AdvA, S)],0).dot(Kmmi)
        dF_dKmm = -Admu.dot(Kmmim.T) + AdvA.sum(0) - tmp - tmp.T
        dF_dKmm = 0.5*(dF_dKmm + dF_dKmm.T) # necessary? GPy bug?
        tmp = S.reshape(-1, num_inducing).dot(Kmmi).reshape(num_outputs, num_inducing , num_inducing )
        tmp = 2.*(tmp - np.eye(num_inducing)[None, :,:])

        dF_dKmn = Kmmim.dot(dF_dmu.T)
        for a,b in zip(tmp, Adv):
            dF_dKmn += np.dot(a.T, b)

        dF_dm = Admu
        dF_dS = AdvA

        #adjust gradient to account for mean function
#        if mean_function is not None:
#            dF_dmfX = dF_dmu.copy()
#            dF_dmfZ = -Admu
#            dF_dKmn -= np.dot(Kmmi_mfZ, dF_dmu.T)
#            dF_dKmm += Admu.dot(Kmmi_mfZ.T)


        #sum (gradients of) expected likelihood and KL part w.r.t local parameters
        log_marginal = F.sum() - KL
        dL_dm, dL_dS, dL_dKmm, dL_dKmn = dF_dm - dKL_dm, dF_dS- dKL_dS, dF_dKmm- dKL_dKmm, dF_dKmn
    
        dL_dchol = 2.*np.array([np.dot(a,b) for a, b in zip(dL_dS, L) ])
        dL_dchol = choleskies.triang_to_flat(dL_dchol)
        
        #sum (gradients of) expected likelihood and KL part w.r.t global parameters    
#        dL_dgm, dL_dgS, dL_dgKmm, dL_dgKmn = dF_dgm, dF_dgS, dF_dgKmm, dF_dgKmn
#        dL_dgKmn_full=np.zeros([g_num_inducing, num_data_full])
#        dL_dgKmn_full[:, index] = dL_dgKmn
#        
#        dL_dgchol = 2.*np.array([np.dot(a,b) for a, b in zip(dL_dgS, g_L) ])
        #dL_dgchol = choleskies.triang_to_flat(dL_dgchol)


        grad_dict = {'dL_dKmm':dL_dKmm, 'dL_dKmn':dL_dKmn, 'dL_dKdiag': dF_dv.sum(1), 'dL_dm':dL_dm, 'dL_dchol':dL_dchol, 'dL_dthetaL':dF_dthetaL}
#        g_grad_dict = {'dL_dKmm':dL_dgKmm, 'dL_dKmn':dL_dgKmn_full, 'dL_dKdiag': 0, 'dL_dm':dL_dgm, 'dL_dchol':dL_dgchol, 'dL_dthetaL':None}
#        if mean_function is not None:
#            grad_dict['dL_dmfZ'] = dF_dmfZ - dKL_dmfZ
#            grad_dict['dL_dmfX'] = dF_dmfX
        return Posterior(mean=q_u_mean, cov=S.T, K=Kmm, prior_mean=prior_mean_u), log_marginal, grad_dict,index
                
        
    def local_likelihood(self,  X, Y, l_svgps,  g_q_u_mean, g_q_u_chol, g_kern, g_Z, g_mean_function=None):

        num_data, _ = Y.shape
        F=np.zeros([num_data, len(l_svgps)])
        
        #global
        g_num_inducing, num_outputs = g_q_u_mean.shape
        
        #expand cholesky representation
        g_L = choleskies.flat_to_triang(g_q_u_chol)


        g_S = np.empty((num_outputs, g_num_inducing, g_num_inducing))
        [np.dot(g_L[i,:,:], g_L[i,:,:].T, g_S[i,:,:]) for i in range(num_outputs)]


        if g_mean_function is not None:
            g_prior_mean_u = g_mean_function.f(g_Z)
            g_prior_mean_f = g_mean_function.f(X)
        else:
            g_prior_mean_u = np.zeros((g_num_inducing, num_outputs))
            g_prior_mean_f = np.zeros((num_data, num_outputs))
        
        g_Kmm = g_kern.K(g_Z)
        g_Kmn = g_kern.K(g_Z, X)
        #g_Knn_diag = g_kern.Kdiag(X)
        g_Lm = linalg.jitchol(g_Kmm)        
        
        g_A, _ = linalg.dpotrs(g_Lm, g_Kmn)
        g_mu = g_prior_mean_f + np.dot(g_A.T, g_q_u_mean - g_prior_mean_u)
        g_v = np.empty((num_data, num_outputs))
        for i in range(num_outputs):
            tmp = dtrmm(1.0,g_L[i].T, g_A, lower=0, trans_a=0)
            g_v[:,i] = np.sum(np.square(tmp),0)
        #Anh v += (Knn_diag - np.sum(A*Kmn,0))[:,None]
            
        for i in range(len(l_svgps)):
            num_inducing, num_outputs = l_svgps[i].q_u_mean.shape
    
            #expand cholesky representation
            L = choleskies.flat_to_triang(l_svgps[i].q_u_chol)
    
    
            S = np.empty((num_outputs, num_inducing, num_inducing))
            [np.dot(L[i,:,:], L[i,:,:].T, S[i,:,:]) for i in range(num_outputs)]
               
            
            prior_mean_u = np.zeros((num_inducing, num_outputs))
            prior_mean_f = np.zeros((num_data, num_outputs))
                
    
                
    
            #compute kernel related stuff
            Kmm = l_svgps[i].kern.K(l_svgps[i].Z)
            Kmn = l_svgps[i].kern.K(l_svgps[i].Z, X)
            Knn_diag = l_svgps[i].kern.Kdiag(X)
            Lm = linalg.jitchol(Kmm)
            #Kmmi, _ = linalg.dpotri(Lm)
            
    
            
    
            #compute the marginal means and variances of q(f)
            A, _ = linalg.dpotrs(Lm, Kmn)
            mu = prior_mean_f + np.dot(A.T, l_svgps[i].q_u_mean - prior_mean_u)
            v = np.empty((num_data, num_outputs))
            for i in range(num_outputs):
                tmp = dtrmm(1.0,L[i].T, A, lower=0, trans_a=0)
                v[:,i] = np.sum(np.square(tmp),0)
            v += (Knn_diag - np.sum(A*Kmn,0))[:,None]
            
                    
            #final marginal means and variances of q(f)
            mu += g_mu
            v += g_v

            #quadrature for the likelihood
            F[:,i,None], dF_dmu, dF_dv, dF_dthetaL = l_svgps[i].likelihood.variational_expectations(Y, mu, v, Y_metadata=l_svgps[i].Y_metadata)
        
   