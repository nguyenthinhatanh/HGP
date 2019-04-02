import numpy as np
from GPy.core.svgp import SVGP
from GPy.inference.latent_function_inference.smgp import SMGP as smgp_inf

import sys, traceback
import GPy
from sklearn.cluster import KMeans
from GPy.core.svgp_modified import SVGPM
import random
from scipy.spatial import distance
from numpy import genfromtxt
import math
import pickle
class SMGP(SVGP):
    def __init__(self, X, Y, Z, kernel, likelihood, mean_function=None, name='SMGP', Y_metadata=None, batchsize=None, num_latent_functions=None, num_cluster=1, num_local_Z=0, upperbound=-1, lowerbound_ratio=-1):
        super(SMGP, self).__init__(X, Y, Z, kernel, likelihood, mean_function, Y_metadata=Y_metadata, batchsize=batchsize, num_latent_functions=num_latent_functions)
        self.rho_all = np.zeros((X.shape[0], num_cluster),dtype=np.bool)
        self.upperbound=upperbound
        self.lowerbound_ratio=lowerbound_ratio

        print('before kmean')
        yStd  = np.std(self.Y_all);
        self.cluster_model = KMeans(n_clusters=num_cluster).fit(np.concatenate((self.X_all,self.Y_all/yStd),axis=1))
        labels = self.cluster_model.predict(np.concatenate((self.X_all,self.Y_all/yStd),axis=1))
        print('after kmean')

        local_Z = np.zeros((num_local_Z, X.shape[1], num_cluster))

        for cc in range(0,num_cluster):
            ii = np.where(labels == cc)[0]
            self.rho_all[ii,cc]=1;
            if (ii.shape[0]>num_local_Z):
                pu=np.random.permutation(ii.shape[0])
                z_ii=ii[pu[range(num_local_Z)]]
                local_Z[:,:,cc] = X[z_ii,:]
            else:
                local_Z[range(ii.shape[0]),:,cc] = X[ii,:]
                jj = np.where(labels != cc)[0]
                pu=np.random.permutation(jj.shape[0])
                z_jj=jj[pu[range(num_local_Z-ii.shape[0])]]
                local_Z[range(ii.shape[0],num_local_Z),:,cc] = X[z_jj,:]

        if batchsize is None:
            X_batch,  Y_batch, rho_batch, index = X, Y, self.rho_all, range(X.shape[0])
        else:
            X_batch, Y_batch, rho_batch, index= self.new_batch_full()


        self.l_svgps = [SVGPM(X_batch, Y_batch, local_Z[:,:,i], likelihood=likelihood.copy(), kernel=kernel.copy(), name='SVGPM'+str(i)) for i in range(0,num_cluster)]
        self.link_parameters(*(self.l_svgps));
        self.X, self.Y= X_batch,Y_batch
        self.rho = rho_batch
        self.index=index

        #create the inference method
        self.inference_method=smgp_inf()

        self.update_cnt=1




    def save_params(self, output,num_iter,training_time_update):
        param_names=self.parameter_names()
        for i in range(param_names.__len__()):
            attr = self.__getitem__(param_names[i])
            pickle.dump(attr.__array__(), output)

    def load_params(self, output):
        param_names=self.parameter_names()
        for i in range(param_names.__len__()):
            attr=pickle.load(output)
            self.__setitem__(param_names[i], attr)


    def compute_inverse_distance(self,x):
        K = len(self.l_svgps)
        dim = x.shape[1]
        nu = self.l_svgps[0].Z.shape[0]
        centers = np.zeros([K, dim])
        Mu = np.zeros([nu*dim, K]);
        W = np.zeros([nu*dim, K]);
        for k in range(K):
            centers[k,:]=np.mean(self.l_svgps[k].Z, axis=0)
            matMu = np.tile(centers[None,k,:],(nu,1)); # nu x dim
            Mu[:,k,None] = matMu.reshape(nu*dim, 1); # (nu*dim)x1 as W(:,k)
            W[:,k,None] = self.l_svgps[k].Z.reshape(nu*dim, 1)

        # sum over partitions and then reshape
        diagCov = (np.sum((W-Mu)**2, axis=1)).reshape(nu,dim)
        # sum over xu and normalize (1xdim)
        diagCov = np.sum(diagCov,axis=0,keepdims=True)/(nu*K-K);
        diagCov = diagCov** 0.5;

        nx = (x / np.tile(diagCov,(x.shape[0],1)));
        nmu = (centers / np.tile(diagCov,(K,1)));

        rho = distance.cdist(nx, nmu, 'sqeuclidean')
        rho_map= np.zeros(rho.shape,dtype=np.bool)
        label = np.argmin(rho, axis=1);

        for k in range(K):
            rho_map[label==k,k]=1

        return rho_map
	
    def parameters_changed(self):
        if (self.update_cnt%50==0):
            self.rho_all = self.compute_inverse_distance(self.X_all)
            if self.upperbound>0:
                self.rho_all = self.qzi_trim(self.rho_all, self.upperbound, self.lowerbound_ratio)


            for i in range(len(self.l_svgps)):
                rho_full = self.rho_all[:, i]
                index = np.where(rho_full>0)[0]
                if (index.size==0):
                    pu=np.random.permutation(self.rho_all.shape[0])
                    z_ii=pu[0]
                    self.rho_all[z_ii,:]=0
                    self.rho_all[z_ii,i]=1


        self.set_data(*self.new_batch_full())

        self.posterior, l_posteriors, self._log_marginal_likelihood, g_grad_dict, l_grad_dicts = self.inference_method.inference(self.q_u_mean, self.q_u_chol, self.kern, self.Z, self.likelihood,  self.mean_function,self.Y_metadata, self.l_svgps, self.X, self.Y, self.rho,  KL_scale=1.0, batch_scale=float(self.X_all.shape[0])/float(self.X.shape[0]))
        self.update_gradients_helper(g_grad_dict);
        for i in range(len(self.l_svgps)):
             self.l_svgps[i].update_gradients_helper(l_grad_dicts[i])
             self.l_svgps[i].posterior=l_posteriors[i]

    def _raw_predict(self, Xnew, full_cov=False, kern=None):
        """
        For making predictions, does not account for normalization or likelihood

        full_cov is a boolean which defines whether the full covariance matrix
        of the prediction is computed. If full_cov is False (default), only the
        diagonal of the covariance is returned.

        .. math::
            p(f*|X*, X, Y) = \int^{\inf}_{\inf} p(f*|f,X*)p(f|X,Y) df
                        = N(f*| K_{x*x}(K_{xx} + \Sigma)^{-1}Y, K_{x*x*} - K_{xx*}(K_{xx} + \Sigma)^{-1}K_{xx*}
            \Sigma := \texttt{Likelihood.variance / Approximate likelihood covariance}
        """
        assert full_cov==False
        mu, var = self.posterior._raw_predict(kern=self.kern if kern is None else kern, Xnew=Xnew, pred_var=self._predictive_variable, full_cov=full_cov)
        if self.mean_function is not None:
            mu += self.mean_function.f(Xnew)

        l_mu=np.empty((Xnew.shape[0], len(self.l_svgps)))
        l_var=np.empty((Xnew.shape[0], len(self.l_svgps)))

        for i in range(len(self.l_svgps)):
            mu1, var1 = self.l_svgps[i].posterior._raw_predict(kern=self.l_svgps[i].kern, Xnew=Xnew, pred_var=self.l_svgps[i]._predictive_variable, full_cov=full_cov)
            mu1+=mu
            var1+=var
            l_mu[:,i,None], l_var[:,i,None]= mu1, var1

        return l_mu, l_var

    def _raw_predict1(self, Xnew, full_cov=False, kern=None):
        """
        For making predictions, does not account for normalization or likelihood

        full_cov is a boolean which defines whether the full covariance matrix
        of the prediction is computed. If full_cov is False (default), only the
        diagonal of the covariance is returned.

        .. math::
            p(f*|X*, X, Y) = \int^{\inf}_{\inf} p(f*|f,X*)p(f|X,Y) df
                        = N(f*| K_{x*x}(K_{xx} + \Sigma)^{-1}Y, K_{x*x*} - K_{xx*}(K_{xx} + \Sigma)^{-1}K_{xx*}
            \Sigma := \texttt{Likelihood.variance / Approximate likelihood covariance}
        """
        assert full_cov==False
        mu, var = self.posterior._raw_predict(kern=self.kern if kern is None else kern, Xnew=Xnew, pred_var=self._predictive_variable, full_cov=full_cov)
        if self.mean_function is not None:
            mu += self.mean_function.f(Xnew)

        l_mu=np.empty((Xnew.shape[0], len(self.l_svgps)))
        l_var=np.empty((Xnew.shape[0], len(self.l_svgps)))

        for i in range(len(self.l_svgps)):
            mu1, var1 = self.l_svgps[i].posterior._raw_predict(kern=self.l_svgps[i].kern, Xnew=Xnew, pred_var=self.l_svgps[i]._predictive_variable, full_cov=full_cov)
            mu1+=mu
            var1+=var
            l_mu[:,i,None], l_var[:,i,None]= mu1, var1
            print(i)

        return l_mu, l_var

    def predict(self, Xnew, full_cov=False, Y_metadata=None, kern=None, likelihood=None, include_likelihood=True):
        """
        Predict the function(s) at the new point(s) Xnew. This includes the likelihood
        variance added to the predicted underlying function (usually referred to as f).

        In order to predict without adding in the likelihood give
        `include_likelihood=False`, or refer to self.predict_noiseless().

        :param Xnew: The points at which to make a prediction
        :type Xnew: np.ndarray (Nnew x self.input_dim)
        :param full_cov: whether to return the full covariance matrix, or just
                         the diagonal
        :type full_cov: bool
        :param Y_metadata: metadata about the predicting point to pass to the likelihood
        :param kern: The kernel to use for prediction (defaults to the model
                     kern). this is useful for examining e.g. subprocesses.
        :param bool include_likelihood: Whether or not to add likelihood noise to the predicted underlying latent function f.

        :returns: (mean, var):
            mean: posterior mean, a Numpy array, Nnew x self.input_dim
            var: posterior variance, a Numpy array, Nnew x 1 if full_cov=False, Nnew x Nnew otherwise

           If full_cov and self.input_dim > 1, the return shape of var is Nnew x Nnew x self.input_dim. If self.input_dim == 1, the return shape is Nnew x Nnew.
           This is to allow for different normalizations of the output dimensions.

        Note: If you want the predictive quantiles (e.g. 95% confidence interval) use :py:func:"~GPy.core.gp.GP.predict_quantiles".
        """
        #predict the latent function values

        assert full_cov==False
        mu, var = self.posterior._raw_predict(kern=self.kern if kern is None else kern, Xnew=Xnew, pred_var=self._predictive_variable, full_cov=full_cov)
        if self.mean_function is not None:
            mu += self.mean_function.f(Xnew)

        l_mu=np.empty([Xnew.shape[0], len(self.l_svgps)])
        l_var=np.empty([Xnew.shape[0], len(self.l_svgps)])


        test_rho = self.compute_inverse_distance(Xnew)


        for i in range(len(self.l_svgps)):
            mu1, var1= self.l_svgps[i].posterior._raw_predict(kern=self.l_svgps[i].kern, Xnew=Xnew, pred_var=self.l_svgps[i]._predictive_variable, full_cov=full_cov)
            mu1+=mu

            if self.normalizer is not None:
                mu1, var1 = self.normalizer.inverse_mean( mu1), self.normalizer.inverse_variance(var1)
            if include_likelihood:
                # now push through likelihood
                if likelihood is None:
                    likelihood = self.l_svgps[i].likelihood
                mu1,var1 = likelihood.predictive_values(mu1, var1, full_cov, Y_metadata=Y_metadata)

            l_mu[:,i,None], l_var[:,i,None]= mu1, var1


        final_mu = np.sum(l_mu*test_rho,axis=1, keepdims=True)
        final_var = np.sum(l_var*test_rho,axis=1, keepdims=True)


        return final_mu, final_var

    def log_predictive_density1(self, x_test, y_test, upperbound=-1, lowerbound_ratio=-1, Y_metadata=None):
        """
        Calculation of the log predictive density

        .. math:
            p(y_{*}|D) = p(y_{*}|f_{*})p(f_{*}|\mu_{*}\\sigma^{2}_{*})

        :param x_test: test locations (x_{*})
        :type x_test: (Nx1) array
        :param y_test: test observations (y_{*})
        :type y_test: (Nx1) array
        :param Y_metadata: metadata associated with the test points
        """
        l_mu, l_var = self._raw_predict1(x_test)
        l_log_PD=np.empty([x_test.shape[0], len(self.l_svgps)])


        test_rho = self.compute_inverse_distance(x_test)
        if upperbound>0:
            test_rho = self.qzi_trim(test_rho, upperbound, lowerbound_ratio)


        for i in range(len(self.l_svgps)):
            mu1=l_mu[:,i,None]
            var1=l_var[:,i,None]
            likelihood = self.l_svgps[i].likelihood
            l_log_PD[:,i,None] = likelihood.log_predictive_density(y_test, mu1, var1, Y_metadata=Y_metadata)
            l_mu[:,i,None], l_var[:,i,None] = likelihood.predictive_values(mu1, var1, Y_metadata=Y_metadata)



        final_mu = np.sum(l_mu*test_rho,axis=1, keepdims=True)
        final_var = np.sum(l_var*test_rho,axis=1, keepdims=True)
        final_log_PD= np.sum(l_log_PD*test_rho,axis=1, keepdims=True)

        return final_log_PD, final_mu, final_var

    def log_predictive_density(self, x_test, y_test, upperbound=-1, lowerbound_ratio=-1, Y_metadata=None):
        """
        Calculation of the log predictive density

        .. math:
            p(y_{*}|D) = p(y_{*}|f_{*})p(f_{*}|\mu_{*}\\sigma^{2}_{*})

        :param x_test: test locations (x_{*})
        :type x_test: (Nx1) array
        :param y_test: test observations (y_{*})
        :type y_test: (Nx1) array
        :param Y_metadata: metadata associated with the test points
        """
        l_mu, l_var = self._raw_predict(x_test)
        l_log_PD=np.empty([x_test.shape[0], len(self.l_svgps)])

 
        test_rho = self.compute_inverse_distance(x_test)
        if upperbound>0:
            test_rho = self.qzi_trim(test_rho, upperbound, lowerbound_ratio)


        for i in range(len(self.l_svgps)):
            mu1=l_mu[:,i,None]
            var1=l_var[:,i,None]
            likelihood = self.l_svgps[i].likelihood
            l_log_PD[:,i,None] = likelihood.log_predictive_density(y_test, mu1, var1, Y_metadata=Y_metadata)
            l_mu[:,i,None], l_var[:,i,None] = likelihood.predictive_values(mu1, var1, Y_metadata=Y_metadata)


        final_mu = np.sum(l_mu*test_rho,axis=1, keepdims=True)
        final_var = np.sum(l_var*test_rho,axis=1, keepdims=True)
        final_log_PD= np.sum(l_log_PD*test_rho,axis=1, keepdims=True)

        return final_log_PD, final_mu, final_var

    def log_predictive_density2(self, x_test, y_test, upperbound=-1, lowerbound_ratio=-1, Y_metadata=None):
        """
        Calculation of the log predictive density

        .. math:
            p(y_{*}|D) = p(y_{*}|f_{*})p(f_{*}|\mu_{*}\\sigma^{2}_{*})

        :param x_test: test locations (x_{*})
        :type x_test: (Nx1) array
        :param y_test: test observations (y_{*})
        :type y_test: (Nx1) array
        :param Y_metadata: metadata associated with the test points
        """
        l_mu, l_var = self._raw_predict1(x_test)
        l_log_PD=np.empty([x_test.shape[0], len(self.l_svgps)])


        test_rho = self.compute_inverse_distance(x_test)
        if upperbound>0:
            test_rho = self.qzi_trim(test_rho, upperbound, lowerbound_ratio)


        for i in range(len(self.l_svgps)):
            mu1=l_mu[:,i,None]
            var1=l_var[:,i,None]
            likelihood = self.l_svgps[i].likelihood
            l_log_PD[:,i,None] = likelihood.log_predictive_density(y_test, mu1, var1, Y_metadata=Y_metadata)
            l_mu[:,i,None], l_var[:,i,None] = likelihood.predictive_values(mu1, var1, Y_metadata=Y_metadata)



        final_mu = np.sum(l_mu*test_rho,axis=1, keepdims=True)
        final_var = np.sum(l_var*test_rho,axis=1, keepdims=True)
        final_log_PD= np.sum(l_log_PD*test_rho,axis=1, keepdims=True)

        return final_log_PD, final_mu, final_var

    def predict_quantiles(self, X, quantiles=(2.5, 97.5), Y_metadata=None, kern=None, likelihood=None):
        """
        Get the predictive quantiles around the prediction at X

        :param X: The points at which to make a prediction
        :type X: np.ndarray (Xnew x self.input_dim)
        :param quantiles: tuple of quantiles, default is (2.5, 97.5) which is the 95% interval
        :type quantiles: tuple
        :param kern: optional kernel to use for prediction
        :type predict_kw: dict
        :returns: list of quantiles for each X and predictive quantiles for interval combination
        :rtype: [np.ndarray (Xnew x self.output_dim), np.ndarray (Xnew x self.output_dim)]
        """
        meanY, varY=self.predict(X, full_cov=False, Y_metadata=Y_metadata, kern=kern, likelihood=likelihood)
        return [np.asarray(meanY>(q/100.), dtype=np.int32) for q in quantiles]    #assume Bernoulli likelihood

    def set_data(self, X, Y, rho,index):
        """
        Set the data without calling parameters_changed to avoid wasted computation
        If this is called by the stochastic_grad function this will immediately update the gradients
        """
        assert X.shape[1]==self.Z.shape[1]
        self.X, self.Y, self.rho,self.index = X, Y, rho,index
        for i in range(len(self.l_svgps)):
             self.l_svgps[i].set_data(X,Y)

    def new_batch_full(self):
        """
        Return a new batch of X and Y by taking a chunk of data from the complete X and Y
        """
        #i = next(self.slicer)
        i=[];
        for ii in range(self.rho_all.shape[1]):
            rho_full = self.rho_all[:, ii]
            index = np.where(rho_full>0)[0]
            draw_size = max(math.floor(self.batchsize * index.shape[0]/ self.X_all.shape[0]),1)
            pu=np.random.permutation(index.shape[0])
            z_ii=index[pu[range(draw_size)]]   
            i = i+list(z_ii)
        return self.X_all[i,:], self.Y_all[i,:], self.rho_all[i,:], i


    def stochastic_grad(self, parameters):
        self.update_cnt=self.update_cnt+1
        #self.set_data(*self.new_batch_full())
        return self._grads(parameters)

    def optimizeWithFreezingZ(self):
        self.Z.fix()
        self.kern.fix()
        self.optimize('bfgs')
        self.Z.unfix()
        self.kern.constrain_positive()
        self.optimize('bfgs')

    def qzi_trim(self, rho, upperbound, lowerbound_ratio, set_max_to_one=0):
       # Assign probability to 1 if it is larger than upperbound
       # Assign probability to 0 if it is smaller than
       # max_prob/lowerbound_ratio.
       qzi=rho.copy()
       K=qzi.shape[1]
       half_max_prob = np.amax(qzi,axis=1);
       half_max_prob = half_max_prob/lowerbound_ratio;
       for k in range(K):
           qzi_k = qzi[:,k]
           ii = np.where(qzi_k<half_max_prob)[0]
           qzi[ii,k] =0;

       qzi[np.where(half_max_prob==0)[0],:] = 1/K;
       norm = np.sum(qzi,axis=1, keepdims=True);
       qzi = qzi/np.tile(norm,(1,K));
       for k in range(K):
           qzi_k = qzi[:,k];
           idx = np.where(qzi_k>upperbound)[0]
           qzi[idx,:]=0;
           qzi[idx,k] =1;

       #set_max_to_one=0;
       if(set_max_to_one==1):
          label = np.argmax(qzi,axis=1);
          for i in range(qzi.shape[0]):
             qzi[i,label[i]]=1;

       return qzi



