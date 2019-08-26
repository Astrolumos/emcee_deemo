import numpy as np
import sys
import emcee
import corner
import matplotlib as plt


def model(x, params):
    '''
    This is an example model 'y = a*x + b' where a and b are parameters.
    '''
    return params[0] * x + params[1]


def lnprior_flat(params, params_range=[0, 1]):
    '''
    The prior function is defined as flat whose boundary is defined with 
    params_range.
    '''
    params_range_ = params_range.copy()
    params_range_[1] = -params_range[1]
    if ((params - params_range[0])>0).sum() == params.size \
       and ((params - params_range[1])<0).sum() == params.size:
        return 0
    else:
        return -np.inf


def lnlikelihood(params, data_x, data_y, data_cov, params_range, model):

    prior = lnprior_flat(params, params_range)
    if prior == -np.inf:
        return -np.inf
    model = model(data_x, params)
    inv_data_cov = np.linalg.inv(data_cov)
    chi2 = np.dot(np.dot((data_y - model).T, inv_data_cov), (data_y - model))
    sys.stdout.write(str(chi2))
    sys.stdout.write('\n')
    sys.stdout.flush()
    return -chi2 * 0.5


def emceeMCMC(params0, nstep, lstep, data_x, data_y, data_cov, params_range,
              model, burnin=None):
    ndim = len(params0)
    nwalkers = 2 * ndim + 2
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnlikelihood,
                                    args=[data_x, data_y, data_cov,
                                          params_range, model], threads=40)
    pos = [np.asarray(params0) + lstep * np.random.rand(ndim) for i in range(nwalkers)]

    if burnin is None:
        pos, prob, state = sampler.run_mcmc(pos, nstep)
        return sampler.flatchain, sampler.flatlnprobability
    else:
        pos, prob, stata = sampler.run_mcmc(pos, burnin)
        sampler.reset()
        pos, prob, state = sampler.run_mcmc(pos, nstep)
        return sampler.flatchain, sampler.flatlnprobability


data_x = np.linspace(0, 10, 20)
data_y_sample = np.zeros((50, 20))
for i in range(50):
    data_y_sample[i] = 2 * data_x + 3 + np.random.normal(size=20) * 2

data_y = np.mean(data_y_sample, axis=0)
data_cov = np.cov(data_y_sample.T)

params_0 = [1, 1]
params_range = np.array([[-0, -0], [10, 10]])
nstep = 500
lstep = 2

param_sample, lnlklhd_chain = emceeMCMC(params_0, nstep, lstep, data_x, data_y,
                                        data_cov, params_range, model)
corner.corner(param_sample)
plt.show()
