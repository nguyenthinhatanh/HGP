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
import climin
from smgp import SMGP
from scipy import stats
import pickle
import time
from scipy import optimize

class BIN_SMGP():
	def inference(self, X, Y,numZ,num_local_Z,num_cluster, batchsize=1000,upperbound=-1, lowerbound_ratio=-1, optimizer=1, num_iters=1000):

		[Ntrain, d]=X.shape

		Yi=(Y==1).astype(int)
		pu=np.random.permutation(Ntrain)
		#numZ=300
		Z=X[pu[range(numZ)], :]

		#batchsize = 1000
		lik = GPy.likelihoods.Bernoulli()
		k = GPy.kern.RBF(d, lengthscale=5.,ARD=True) + GPy.kern.White(1, 1e-6)
		m = SMGP(X, Yi, Z, likelihood=lik, kernel=k, batchsize=batchsize, num_cluster=num_cluster,num_local_Z=num_local_Z,upperbound=upperbound, lowerbound_ratio=lowerbound_ratio)
		m.kern.white.variance = 1e-5
		m.kern.white.fix()


		from ipywidgets import Text
		from IPython.display import display

		t = Text(align='right')
		display(t)
		m.iter_no=0
		#import sys
		def callback_adadelta(i):
			t.value = str(m.log_likelihood())
			print(i['n_iter'])
			if i['n_iter'] > num_iters:
				return True
			return False
		
		def callback_lbfgsb(i):
			m.iter_no=m.iter_no+1
			t.value = str(m.log_likelihood())
			print(m.iter_no)    
		
		if optimizer==1: #Adadelta          
			opt = climin.Adadelta(m.optimizer_array, m.stochastic_grad, step_rate=0.2, momentum=0.9)          
			info = opt.minimize_until(callback_adadelta)
		elif optimizer==2: #l_bfgs_b    		
			x, f, d = optimize.fmin_l_bfgs_b(m._objective, m.optimizer_array, fprime=m.stochastic_grad, maxfun=1000, callback=callback_lbfgsb)
		else:
			print('optimizer not supported')

		return m

	#record performance and training time every 10 function calls 
	def inference_time(self, X, Y,Xt, Yt, numZ,num_local_Z,num_cluster, batchsize=1000,upperbound=-1, lowerbound_ratio=-1, optimizer=1, num_iters=1000):
		[Ntrain, d]=X.shape
		ac_array=[];
		decv_array=[];
		duration_array=[]
		self.start_time=time.time()
		self.total_pred_time=0;

		Yi=(Y==1).astype(int)
		pu=np.random.permutation(Ntrain)
	
		Z=X[pu[range(numZ)], :]


		lik = GPy.likelihoods.Bernoulli()
		k = GPy.kern.RBF(d, lengthscale=5.,ARD=True) + GPy.kern.White(1, 1e-6)
		m = SMGP(X, Yi, Z, likelihood=lik, kernel=k, batchsize=batchsize, num_cluster=num_cluster,num_local_Z=num_local_Z,upperbound=upperbound, lowerbound_ratio=lowerbound_ratio)
		m.kern.white.variance = 1e-5
		m.kern.white.fix()


		from ipywidgets import Text
		from IPython.display import display

		t = Text(align='right')
		display(t)

		m.iter_no=0
		def callback_adadelta(i):
			t.value = str(m.log_likelihood())
			print(i['n_iter'])
			if (i['n_iter'] %10==0):
				start_pred=time.time();
				ac, pred,decv, test_error, Yt_m, Yt_v = self.prediction1(m, Xt,Yt)
				ac_array.append(ac)
				decv_array.append(decv)
				pred_dur=time.time()-start_pred
				self.total_pred_time=self.total_pred_time+pred_dur
				duration_array.append(time.time()-self.start_time-self.total_pred_time)
			if i['n_iter'] > 1000:
				return True
			return False
		
		def callback_lbfgsb(i):
			m.iter_no=m.iter_no+1
			t.value = str(m.log_likelihood())
			print(m.iter_no)
			if (m.iter_no %10==0):
				start_pred=time.time();
				ac, pred,decv, test_error, Yt_m, Yt_v = self.prediction1(m, Xt,Yt)
				ac_array.append(ac)
				decv_array.append(decv)
				pred_dur=time.time()-start_pred
				self.total_pred_time=self.total_pred_time+pred_dur
				duration_array.append(time.time()-self.start_time-self.total_pred_time)

		if optimizer==1: #Adadelta          
			opt = climin.Adadelta(m.optimizer_array, m.stochastic_grad, step_rate=0.2, momentum=0.9)          
			info = opt.minimize_until(callback_adadelta)
		elif optimizer==2: #l_bfgs_b
			x, f, d = optimize.fmin_l_bfgs_b(m._objective, m.optimizer_array, fprime=m.stochastic_grad, maxfun=1000, callback=callback_lbfgsb)
		else:
			print('optimizer not supported')
			

		return m,ac_array, decv_array,duration_array

	def threshold(self,a1, threshmin=None, threshmax=None, newval=0):
		a = np.array(a1, copy=True)
		mask = np.zeros(a.shape, dtype=bool)
		if threshmax is None:
			mask = (a < threshmin)
		elif threshmin is None:
			mask = (a > threshmax)
		else:
			mask = (a<threshmin)|(a>threshmax)
		a[mask] = newval
		return a

	def prediction(self,m,Xt,Yt, upperbound=-1, lowerbound_ratio=-1):
		Yti=(Yt==1).astype(int)
		final_log_PD, final_mu, final_var = m.log_predictive_density( Xt, Yti, upperbound,lowerbound_ratio)
		pred=self.threshold(final_mu, 0.5)
		pred=self.threshold(pred, threshmax=0.5, newval=1)
		pred=pred*(pred!=0.5)
		test_error = 100*np.sum(pred!=Yti)/Yti.shape[0]
		ac = np.sum(pred==Yti)/Yti.shape[0];
		log_PD=np.sum(final_log_PD)/Yti.shape[0]


		return ac, pred, log_PD, test_error, final_mu, final_var

	def prediction1(self,m,Xt,Yt, upperbound=-1, lowerbound_ratio=-1):
		Yti=(Yt==1).astype(int)
		final_log_PD, final_mu, final_var = m.log_predictive_density1( Xt, Yti, upperbound,lowerbound_ratio)
		pred=self.threshold(final_mu, 0.5)
		pred=self.threshold(pred, threshmax=0.5, newval=1)
		pred=pred*(pred!=0.5)
		test_error = 100*np.sum(pred!=Yti)/Yti.shape[0]
		ac = np.sum(pred==Yti)/Yti.shape[0];
		log_PD=np.sum(final_log_PD)/Yti.shape[0]


		return ac, pred, log_PD, test_error, final_mu, final_var

