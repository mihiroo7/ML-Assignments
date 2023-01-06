import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

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
        
        return 1/(1+ np.exp(np.multiply(np.transpose(x[a1]),self.theta)))
    
    def J_theta(self,x,y):
        m = np.size(x)
        ansj = 0
        for a1 in range(m):
            ansj+= (-1/m)*(y[a1][0]*np.log(self.h_theta(x[a1]))+(1-y[a1][0])*np.log(1-self.h_theta(x[a1])))
            
        return ansj
    
    def hessian(self,x,y):
        m = np.size(x)
        ans = np.zeros((np.size(x[0]),np.size(x[0])))
        
        
        for a1 in range(m):
            const_term = self.h_theta(x,a1)*(1-self.h_theta(x,a1))
            for i in range(np.size(x[0])):
                for j in range(np.size(x[0])):
                    ans[i][j] += (1/m)*const_term*x[a1][i]*x[a1][j]
        
        return ans
    
    def first_derivative(self,x,y):
        m = np.size(x)
        ans = np.zeros((np.size(x[0]),1))
        for i in range(np.size(x[0])):
            for j in range(m):
                ans[i][0]+=(y[j][0]*x[j][i]- self.h_theta(x,i)*x[j][i])
        return ans
        
        
    def newton(self,x,y):
        previos_theta = self.theta
        theata = self.theta - np.multiply(np.linalg.inv( self.hessian(x,y)), self.first_derivative(x,y))
        if(np.mod(np.subtract(theata-previos_theta)) > self.eps) :
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

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        m = np.size(x)
        ans = np.zeros((m,1))
        for i in range(m):
            ans[i] = self.h_theta(x,i)
            
        return ans
            
        # *** END CODE HERE ***
