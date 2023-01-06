import numpy as np
import util

from linear_model import LinearModel
import matplotlib.pyplot as plt

def plot(theta,x,y):
    X = np.linspace(0,5,100)
    Y = -1*(theta[0][0]+theta[1][0]*X)/theta[2][0]
    x0 = []
    y0 = []
    x1 = []
    y1 = []
    for i in range(np.shape(x)[0]):
        if y[i]==0:
            x0.append(x[i][1])
            y0.append(x[i][2])
        else:
            x1.append(x[i][1])
            y1.append(x[i][2])
            
    x0 = np.array(x0)
    y0 = np.array(y0)
    x1 = np.array(x1)
    y1 = np.array(y1)
    
    plt.plot(X,Y)
    plt.scatter(x0,y0,color="yellow")
    plt.scatter(x1,y1,color="pink")
    plt.xlim(0, 5)
    plt.ylim(-400, 400)
    plt.show()


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    Model = LogisticRegression(theta_0= np.array([[0],[0],[0]]))
    Model.fit(x_train, y_train)
    
    x_valid, y_valid = util.load_dataset(eval_path, add_intercept=True)
    pridicted = Model.predict(x_valid)
    print(Model.accuracy(pridicted, y_valid))
    plot(Model.theta,x_valid,y_valid)
    
    
    

    # *** START CODE HERE ***
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    

    
    def h_theta(self, x, a1):
        X = np.zeros((np.size(x[a1]),1))
        for i in range(np.size(x[a1])):
            X[i][0] = x[a1][i]
        values =0

        values = np.exp(-1*np.dot(self.theta.T,X))
        
        return 1/(1+ values)
    
    
    def hessian(self,x,y):
        m = np.shape(x)[0]
        ans = np.zeros((np.size(x[0]),np.size(x[0])))
        
        for a1 in range(m):
            const_term = self.h_theta(x,a1)*(1-self.h_theta(x,a1))
            
            for i in range(np.size(x[0])):
                for j in range(np.size(x[0])):
                    ans[i][j] += (1/m)*const_term*x[a1][i]*x[a1][j]
        
        return ans
    
    def first_derivative(self,x,y):
        m = np.shape(x)[0]
        ans = np.zeros((np.size(x[0]),1))
        for i in range(np.size(x[0])):
            for j in range(m):
                ans[i][0]+=(y[j]*x[j][i]- self.h_theta(x,j)*x[j][i])
        return ans
       
    def mod(self,y):
        m = np.size(y)
        ans =0
        for i in range(m):
            ans+= abs(y[i][0])
        return ans
         
        
    def newton(self,x,y):
        previos_theta = self.theta
        # self.theta = self.theta - np.dot(np.linalg.inv( self.hessian(x,y)), self.first_derivative(x,y))
        # print(self.mod(np.subtract(self.theta,previos_theta)),self.eps)
        

        try :
            self.theta = self.theta - np.dot(np.linalg.inv( self.hessian(x,y)), self.first_derivative(x,y))
        except :
            pass
        print(self.mod(np.subtract(self.theta,previos_theta)),self.eps)
        if(self.mod(np.subtract(self.theta,previos_theta)) > self.eps) :
            self.newton(x,y)

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        self.newton(x,y)
        # *** END CODE HERE ***
        
    def accuracy(self,pred,ans):
        vari = 0
        for i in range(np.size(pred)):
            x=0
            if pred[i][0]>0.5 :
                x=1
            else:
                x=0
            vari += (abs(x -ans[i])/np.size(pred))
        return vari

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        m = np.shape(x)[0]
        ans = np.zeros((m,1))
        for i in range(m):
            ans[i] = self.h_theta(x,i)
            
        return ans
            
        # *** END CODE HERE ***

main("../data/ds1_train.csv","../data/ds1_valid.csv","/")
