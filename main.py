import sub.regression as rg
import sub.optimizer as opt
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=3)
np.random.seed(900)
#load and preprocess data
X_tr_norm, y_tr_norm, X_te_norm, y_te_norm, X_va_norm, y_va_norm, mean, std=rg.preprocessing()


#%%execute regression with LLS
LLS = opt.SolveLLS(y_tr_norm, X_tr_norm)
LLS.run()
LLS.print_result('LLS')
LLS.plot_w('Weight vector computed by LLS')

#test model
wLLS = LLS.sol 
rg.test_regression('LLS', wLLS, X_tr_norm, y_tr_norm, X_te_norm, y_te_norm, X_va_norm, y_va_norm, mean, std)

#%%ridge regression
ridgeRegr = opt.SolveRidgeReg(y_tr_norm, X_tr_norm)

# cross-validation
l_trials=1000
lamb = np.linspace(1, 100, l_trials)
opt.cross_validation(ridgeRegr, lamb, X_va_norm, y_va_norm, l_trials)

#from cross validation a good lambda value is around 28
print('\n')
ridgeRegr.run(28)
ridgeRegr.print_result('Ridge regression')
ridgeRegr.plot_w('Weight vector computed by ridge regression')
 
#test model
wRidge = ridgeRegr.sol 
rg.test_regression('ridge regression', wRidge, X_tr_norm, y_tr_norm, X_te_norm, y_te_norm, X_va_norm, y_va_norm, mean, std)

#%% regression with conjugate vectors
print('\n')
conjGrad = opt.SolveConjuGrad(y_tr_norm, X_tr_norm)
conjGrad.run(1)
conjGrad.print_result('Conjugate gradient')
conjGrad.plot_w('Weight vector computed by conjugate gradient')

# test model
wConj = conjGrad.sol  
rg.test_regression('conjugate gradient', wConj, X_tr_norm, y_tr_norm, X_te_norm, y_te_norm, X_va_norm, y_va_norm, mean,
                std)

#%% regression with stochastic grad descent
print('\n')
gamma = 1e-3 
Nit=60000 
stochG = opt.SolveStochGrad(y_tr_norm, X_tr_norm)
stochG.run(gamma, Nit, 1)
stochG.print_result('Stochastic Gradient_algorithm:')
logx = 0
logy = 0
stochG.plot_err('Stochastic Gradient_algorithm: mean square error', logy, logx)
stochG.plot_w('Weight vector computed by Stochastic Gradient algorithm with Adam optimizer')

#test model
wStoc = stochG.sol 
rg.test_regression('Stochastic gradient with Adam',wStoc,X_tr_norm, y_tr_norm, X_te_norm, y_te_norm, X_va_norm, y_va_norm, mean, std)

#%%final w comparison
plt.figure(figsize=(10, 10))
ax = plt.subplot(111)
ax.plot(wStoc, label="Stochastic gradient")
ax.plot(wLLS, label="LLS")
ax.plot(wRidge, label="Ridge regression")
ax.plot(wConj, label="Conjugate gradient")
chartBox = ax.get_position()
ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.6, chartBox.height])
ax.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), shadow=True, ncol=1)
plt.xlabel('Feature n')
plt.grid()
plt.ylabel('W(n)')
plt.title('Comparison of weight vectors from different methods')
plt.show()
x=['age','sex','motorUPDRS','Jitt(%)','Jitt(Abs)','Jitt:RAP','Jitt:PPQ5','Jitt:DDP','Shimm','Shimm(dB)','Shimm:APQ3','Shim:APQ5','Shim:APQ11','Shim:DDA','NHR','HNR','RPDE','DFA','PPE']
ax.set_xticks(np.linspace(0,19,19))
ax.tick_params(axis='x', labelsize=6)
ax.set_xticklabels(x, rotation='vertical')

