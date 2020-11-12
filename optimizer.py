import numpy as np
import matplotlib.pyplot as plt

class SolveMinProb:
    def __init__(self, y=np.ones((3,1)), A=np.eye(3)):
        self.matr=A
        self.Np=y.shape[0]
        self.Nf=A.shape[1]
        self.vect=y
        self.sol=np.zeros((self.Nf,1), dtype=float)
        return

    def plot_w(self, title='Solution'):
        w=self.sol
        n=np.linspace(1,19,19)
        x=['age','sex','motorUPDRS','Jitt(%)','Jitt(Abs)','Jitt:RAP','Jitt:PPQ5','Jitt:DDP','Shimm','Shimm(dB)','Shimm:APQ3','Shim:APQ5','Shim:APQ11','Shim:DDA','NHR','HNR','RPDE','DFA','PPE']
        f, ax = plt.subplots(figsize=(10, 10))
        ax.plot(n,w)
        ax.set_xticks(n)
        ax.set_xticklabels(x, rotation='vertical')
        ax.tick_params(axis='x', labelsize=8)
        plt.xlabel('feature n')
        plt.ylabel('w(n)')
        plt.title(title)
        plt.grid()
        plt.show()
        return
    
    def adam(self, cur_grad, miu, miu2, it):
        beta=0.9
        beta2=0.999
        eps=np.full((self.Nf,1), 1e-8)
        miu=beta*miu+(1-beta)*cur_grad
        miu_corr=miu/(1-(beta**(it+1)))
        miu2=beta2*miu2+(1-beta2)*np.square(cur_grad)
        miu2_corr=miu2/(1-(beta2**(it+1)))
        adam_res=np.divide(miu_corr,np.sqrt(miu2_corr+eps))
        return adam_res, miu, miu2

  
    
    def print_result(self, title):
        print(title, ' :')
        print('the optmimum weight vector is: ')
        print(self.sol)
        return

    def plot_err(self,title='Square Error', logy=0, logx=0):
        err=self.err
        plt.figure()
        if(logy==0) & (logx==0):
            plt.plot(err[:,0], err[:,1])
        if(logy==1) & (logx==0):
            plt.semilogy(err[:,0], err[:, 1])
        if(logy==0) & (logx==1):
            plt.semilogx(err[:,0], err[:, 1])
        if (logy == 1) & (logx == 1):
            plt.loglog(err[:, 0], err[:, 1])
        plt.xlabel('n')
        plt.ylabel('mean square error')
        plt.title(title)
        plt.margins(0.01, 0.1)
        plt.grid()
        plt.show()
        return
    
        
    
class SolveLLS(SolveMinProb):
    def run(self):
        A=self.matr
        y = self.vect
        w=np.dot(np.dot(np.linalg.inv(np.dot(A.T, A)), A.T), y)
        self.sol = w
        self.min=np.linalg.norm(np.dot(A,w)-y)

class SolveGrad(SolveMinProb):
    def run(self, gamma=1e-3, Nit=100): #default values are te
        self.err=np.zeros((Nit,2), dtype=float)
        A=self.matr
        y=self.vect.reshape(self.Np,1)
        w=np.random.rand(self.Nf, 1)
        for it in range(Nit):
            grad=2*np.dot(A.T, (np.dot(A, w)-y))
            w=w-gamma*grad
            self.err[it,0]=it
            self.err[it,1]=np.linalg.norm(np.dot(A,w)-y)
        self.sol=w
        self.min=self.err[it,1]

class SolveSteep(SolveMinProb):
    def run(self, gamma=1e-3, Nit=100): #default values are te
        self.err=np.zeros((Nit,2), dtype=float)
        A=self.matr
        y=self.vect
        w=np.random.rand(self.Nf, 1)
        for it in range(Nit):
            grad=2*np.dot(A.T, (np.dot(A, w)-y))
            hess=2*np.dot(A.T, A)
            gamma=np.linalg.norm(grad)/(np.dot(grad.T, np.dot(hess, grad)))
            w=w-gamma*grad
            self.err[it,0]=it
            self.err[it,1]=np.linalg.norm(np.dot(A,w)-y)
        self.sol=w
        self.min=self.err[it,1]

class SolveStochGrad(SolveMinProb):
    def run(self, gamma=1e-3, Nit=100, adam=0):
        self.err = np.zeros((Nit, 2), dtype=float)
        A = self.matr
        Np=self.Np
        Nf=self.Nf
        miu=np.zeros((Nf,1), dtype=float)
        miu2=np.zeros((Nf,1), dtype=float)
        y = self.vect
        w = np.random.rand(self.Nf, 1)
        for it in range(Nit):
            patient=it % Np
            x_p=A[patient, :].reshape(1,Nf)
            y_p=y[patient].reshape(1,1)
            grad = 2 *np.dot(x_p.T, (np.dot(x_p, w)-y_p)) 
            if(adam):
                update, miu, miu2=self.adam(grad, miu, miu2, it)
            else:
                update=grad
            w = w - gamma * update
            self.err[it, 0] = it
            self.err[it, 1] = np.mean(np.square(np.dot(A, w) - y))
        self.sol = w
        self.min = self.err[it, 1]
        
class SolveConjuGrad(SolveMinProb):
    def run(self, imp=1): #imp=1 means imp without eigenvectors
        A = self.matr
        y=self.vect
        Nf=self.Nf
        w = np.zeros((Nf, 1))
        Q=np.dot(A.T, A)
        b=np.dot(A.T, y).reshape(Nf,1)
        if(imp):
            d=b
            g=-b
            for k in range(Nf):
                alpha=-(np.dot(d.T, g)/(np.dot(np.dot(d.T, Q),d)))
                w=w+alpha*d
                g=g+alpha*np.dot(Q, d)
                beta=np.dot(np.dot(g.T,Q), d)/(np.dot(np.dot(d.T,Q), d))
                d=-g+beta*d
        else:
            lmbda,U=np.linalg.eig(Q)
            w=np.dot(U.T,b)/lmbda.reshape(Nf,1)
            what=np.dot(U,w)
            w=what
        self.sol=w
        
class SolveRidgeReg(SolveMinProb):
    def run(self, lamb):
        A = self.matr
        y=self.vect
        Nf=self.Nf
        w=np.dot(np.dot(np.linalg.inv(np.dot(A.T, A)+lamb*np.eye(Nf)),A.T),y)
        self.sol=w


def MSE(y_hat, y): #generic function for MSE
    error=(y_hat-y)**2
    return error.mean()

def cross_validation(model, parameter, A_val, y_val, size):
    train_MSE=np.zeros((size, 1))
    val_MSE=np.zeros((size, 1))
    it=0
    A = model.matr
    y=model.vect
    for p in parameter:
        model.run(p)
        y_hat=np.dot(A, model.sol)
        y_hat_val=np.dot(A_val, model.sol)
        train_MSE[it]=MSE(y_hat, y)
        val_MSE[it]=MSE(y_hat_val, y_val)
        it+=1
    plt.figure()
    ax = plt.subplot(111)
    ax.plot(parameter, train_MSE, label="train")
    ax.plot(parameter, val_MSE, label="validation")
    chartBox = ax.get_position()
    ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.6, chartBox.height])
    ax.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), shadow=True, ncol=1)
    plt.xlabel('lambda')
    plt.ylabel('MSE')
    plt.title("Cross validation for lambda")
    plt.grid()
    plt.show()