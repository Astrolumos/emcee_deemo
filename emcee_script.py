# This file defines general log-likelihood class for MCMC fitting
# and functions to setup and call emcee to conduct the fitting.

import numpy as np
import emcee
import yaml


class lnlikelihood:

    '''This class defines the logarithmic likelihood used for MCMC;
    fitting.  Inputs: data_x, data_y, cov: arrays; x, y and covmat of data; 
    model: function, the model to be fit;
    params_range: range of flat prior
    paramsdict_free: dictionary of free parameters;
    paramsfixed: values of fixed parameters;
    paramsdict: dictionary of fixed parameters

    '''

    def __init__(self, data_x, data_y, cov, model, params_range,
                 paramsdict_free, params_fixed=None, paramsdict_fixed=None):

        self.data_x = data_x
        self.data_y = data_y
        self.model = model
        self.cov = cov
        self.params_range = params_range
        self.nparams = params_range.shape[0]
        self.params_fixed = params_fixed
        self.paramsdict_fixed = paramsdict_fixed
        self.paramsdict_free = paramsdict_free
        self.invcov = np.linalg.inv(cov)

    def lnprior(self, params):
        for i, p in enumerate(params):
            if not(self.params_range[i][0] <= p <= self.params_range[i][1]):
                return -np.inf
            else:
                continue
        return 0

    def lnposterior(self, params, **kwargs):

        prior = self.lnprior(params)
        if prior == -np.inf:
            return prior


        #model_y = self.model(self.data_x, params, self.paramsdict_free,
        #                     self.params_fixed, self.paramsdict_fixed)
        model_y = self.model(self.data_x, params)

        if (model_y is None) or (self.cov is None):
            return -np.inf
        elif  (np.isnan(model_y).sum() != 0) or (np.isnan(self.cov).sum() != 0):
            return -np.inf
        else:

            diff = model_y - self.data_y
            chi2 = np.dot(np.dot(diff.T, self.invcov), diff)*0.5
            if np.isnan(chi2):
                return -np.inf                
            if chi2 < 0:
                return -np.inf
            #print(chi2 / (self.data_x.size-len(params)))
            return -chi2# - 0.5*np.linalg.slogdet(cov)[1]


def mcmc_setup(filename):

    '''
    Reading the mcmc setup from a yaml file
    '''
    
    paramsdict_free = np.array([])
    params_range = np.array([])
    params_0 = np.array([])
    lsteps = np.array([])
    params_range = np.array([0, 0])
    paramsdict_fixed = np.array([])
    params_fixed = np.array([])

    with open(filename) as file:
        documents = yaml.load(file)

    for i, item in enumerate(documents['params']):
        if item['vary'] is False:
            paramsdict_fixed = np.append(paramsdict_fixed, item['name'])
            params_fixed = np.append(params_fixed, item['value'])
            continue
        paramsdict_free = np.append(paramsdict_free, item['name'])
        params_range = np.vstack([params_range, np.asarray(item['prior']['values'])])
        params_0 = np.append(params_0, item['value'])
        lsteps = np.append(lsteps, item['lsteps'])

    params_range = params_range[1:]

    print('Free parameters:' + str(paramsdict_free))
    nsteps = documents['mcmc']['n_steps']
    nwalkers = documents['mcmc']['n_walkers']
    burnin = documents['mcmc']['burnin']
    return nsteps, nwalkers, lsteps, burnin, paramsdict_free, params_0, params_range,\
        paramsdict_fixed, params_fixed


def runmcmc(params0, nstep, nwalkers, lstep, lnposterior, pool=None, burnin=None, thread=1, **kwargs):
    ndim = params0.size
    #sampler = emcee.EnsembleSampler(nwalkers, ndim, lnposterior, threads=60)
    if pool is None:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnposterior, threads=thread)
    else:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnposterior, pool=pool)
    pos = [np.asarray(params0) + lstep * np.random.rand(ndim) for i in range(nwalkers)]

    if burnin is None:
        pos, prob, state = sampler.run_mcmc(pos, nstep, progress=True)
        return sampler.flatchain, sampler.flatlnprobability
    else:
        pos, prob, stata = sampler.run_mcmc(pos, burnin, progress=True)
        sampler.reset()
        pos, prob, state = sampler.run_mcmc(pos, nstep, progress=True)
        return sampler.flatchain, sampler.flatlnprobability
