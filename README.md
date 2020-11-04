# emcee_demo
This is a script to use emcee to evaluate parameters in given model.
MCMC setup (step length, fixed parameters, priors, etc) is set in an yaml file (so pyyaml package is needed); see https://pypi.org/project/PyYAML/
Users need to change the 'model()' function to fit their own tasks.
The posterior can be plot with package 'getdist', see https://getdist.readthedocs.io/en/latest/
