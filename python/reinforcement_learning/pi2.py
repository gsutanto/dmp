#!/usr/bin/env python
"""
This file defines code for PI2-based parameters optimization 
(such parameters are used in a trajectory roll-out, and 
 costs are measured at each timestep in the roll-out).

Optimization of parameters with PI2 and a REPS-like KL-divergence constraint.
References:
[1] E. Theodorou, J. Buchli, and S. Schaal. A generalized path integral control 
    approach to reinforcement learning. JMLR, 11, 2010.
[2] F. Stulp and O. Sigaud. Path integral policy improvement with covariance 
    matrix adaptation. In ICML, 2012.
[3] J. Peters, K. Mulling, and Y. Altun. Relative entropy policy search. 
    In AAAI, 2010.
[4] Y. Chebotar, M. Kalakrishnan, A. Yahya, A. Li, S. Schaal, S. Levine.
    Path Integral Guided Policy Search. In ICRA, 2017.
"""
import copy
import numpy as np
import scipy as sp

from numpy.linalg import LinAlgError
from scipy.optimize import minimize

class Pi2():
    """ PI2 parameter optimization with (REPS) KL-bound.
    Hyperparameters:
        kl_threshold: KL-divergence threshold between old and new policies.
        covariance_damping: If greater than zero, covariance is computed as a
            multiple of the old covariance. Multiplier is taken to the power
            (1 / covariance_damping). If greater than one, slows down 
            convergence and keeps exploration noise high for more iterations.
        min_temperature: Minimum bound of the temperature optimization for the 
            soft-max probabilities of the policy samples.
    """
    def __init__(self, kl_threshold = 1.0, covariance_damping = 0.0,
                       min_temperature = 0.001, 
                       is_computing_eta_per_timestep = True, 
                       is_printing_min_eta_warning = False):
        self._kl_threshold = kl_threshold
        self._covariance_damping = covariance_damping
        self._min_temperature = min_temperature
        self._is_computing_eta_per_timestep = is_computing_eta_per_timestep
        self._is_printing_min_eta_warning = is_printing_min_eta_warning
    
    def update(self, samples, costs, mean_old, cov_old):        
        """
        Perform optimization with PI2. Computes new mean and covariance matrix
        of the policy parameters given policy samples and their costs.
        Args:
            samples: Matrix of policy samples with dimensions: 
                     [N_samples x D_params].
            costs: Matrix of roll-out costs with dimensions:
                   [N_samples x num_timesteps].
            mean_old: Old policy mean.
            cov_old: Old policy covariance.
        Returns:
            mean_new: New policy mean.
            cov_new: New policy covariance.
            inv_cov_new: Inverse of the new policy covariance.
            chol_cov_new: Cholesky decomposition of the new policy covariance.
        """
        T = costs.shape[1] # num_timesteps
        N_samples = samples.shape[0]
        D_params = samples.shape[1]
        assert (costs.shape[0] == N_samples)
        assert (mean_old.shape == (D_params,))
        assert (cov_old.shape  == (D_params,D_params))
        
        epsilon = 1.0e-38
        
        mean_new_per_timestep = np.zeros((T,D_params))
        cov_new_per_timestep  = np.zeros((T,D_params,D_params))
        
        # Compute cost-to-go for each time step for each sample.
        cost_to_go_per_timestep = np.zeros((N_samples,T))
        cost_to_go_per_timestep[:,T-1] = costs[:,T-1]
        for t in xrange(T-2,-1,-1):
            cost_to_go_per_timestep[:,t] = costs[:,t] + cost_to_go_per_timestep[:,t+1]
        
        if (not self._is_computing_eta_per_timestep): # NOT quite working well...
            [normalized_cost_to_go_per_timestep_vectorized, eta
             ] = self.normalizeCostAndComputeTemperatureParameterEta(np.reshape(cost_to_go_per_timestep, 
                                                                                (N_samples*T,)), epsilon)
            normalized_cost_to_go_per_timestep = np.reshape(normalized_cost_to_go_per_timestep_vectorized, 
                                                            (N_samples,T))
        
        # Iterate over time steps.
        for t in xrange(T):
            if (self._is_computing_eta_per_timestep):
                cost_to_go = cost_to_go_per_timestep[:,t]
                [normalized_cost_to_go, eta
                 ] = self.normalizeCostAndComputeTemperatureParameterEta(cost_to_go, epsilon)
            else: # NOT quite working well...
                normalized_cost_to_go = normalized_cost_to_go_per_timestep[:,t]

            # Compute probabilities of each sample.
            exp_cost = np.exp(-normalized_cost_to_go / eta)
            prob = exp_cost / (np.sum(exp_cost) + epsilon)

            # Update policy mean with weighted max-likelihood.
            mean_new_per_timestep[t] = np.sum(prob[:, np.newaxis] * samples, axis=0)

            # Update policy covariance with weighted max-likelihood.
            for i in xrange(N_samples):
                mean_diff = samples[i] - mean_new_per_timestep[t]
                mean_diff = np.reshape(mean_diff, (D_params, 1))                
                cov_new_per_timestep[t] += prob[i] * np.dot(mean_diff, mean_diff.T)
        
        weight_per_timestep = np.array(range(T, 0, -1))
        sum_weight_per_timestep = np.sum(weight_per_timestep)
        mean_new = (np.sum(weight_per_timestep[:, np.newaxis] * 
                           mean_new_per_timestep, axis=0) / 
                    sum_weight_per_timestep)
        cov_new  = (np.sum(weight_per_timestep[:, np.newaxis, np.newaxis] * 
                           cov_new_per_timestep, axis=0) / 
                    sum_weight_per_timestep)
            
        # If covariance damping is enabled, compute covariance as multiple
        # of the old covariance. The multiplier is first fitted using 
        # max-likelihood and then taken to the power (1/covariance_damping).
        if(self._covariance_damping is not None 
           and self._covariance_damping > 0.0):
            
            mult = np.trace(np.dot(sp.linalg.inv(cov_old), cov_new)) / D_params
            mult = np.power(mult, 1 / self._covariance_damping)
            cov_new = mult * cov_old

        # Compute covariance inverse and cholesky decomposition.
        inv_cov_new  = sp.linalg.inv(cov_new)
        chol_cov_new = sp.linalg.cholesky(cov_new)

        return mean_new, cov_new, inv_cov_new, chol_cov_new
    
    def normalizeCostAndComputeTemperatureParameterEta(self, costs, epsilon=1.0e-10):
        # Normalize costs.
        min_cost = np.min(costs)
        max_cost = np.max(costs)
        normalized_costs = ((costs - min_cost) / (max_cost - min_cost + epsilon))

        # Perform REPS-like optimization of the temperature eta.
        res = minimize(self.KL_dual, 1.0, bounds=((self._min_temperature, 
                                                   None),), 
                       args=(self._kl_threshold, normalized_costs))
        eta = res.x
        if (self._is_printing_min_eta_warning and (eta <= self._min_temperature)):
            print("WARNING: minimum temperature parameter eta is attained; eta = %f" % eta)
        return normalized_costs, eta
    
    def KL_dual(self, eta, kl_threshold, costs):
        """
        Dual function for optimizing the temperature eta according to the given
        KL-divergence constraint.
        
        Args:
            eta: Temperature that has to be optimized.
            kl_threshold: Max. KL-divergence constraint.
            costs: Roll-out costs.            
        Returns:
            Value of the dual function.
        """
        return eta * kl_threshold + eta * np.log((1.0/len(costs)) * 
               np.sum(np.exp(-costs/eta)))