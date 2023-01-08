import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self,varia=None,musal =None, *args, **kwargs):
        self.varia= varia
        self.musal = musal
        self.musal = kwargs.pop('mus')
        
        super(GDA, self).__init__(*args, **kwargs)
    
    def theta(self,y):
        ans =0
        for i in y:
            if(y==1):
                ans+=1
        self.theta = ans/np.size(y)
    
    def mu(self,x,y) :
        self.musal = []
        for k in range(np.size(x)[1]):        
            ans  = np.zeros((np.size(x)[1],1))
            divi = 0;
            for i in y:
                if int(i)==k:
                    divi+=1
                    
            for i in range(np.size(x)[0]):
                temp = np.array([x[i]]).T
                if(int(y[i])==k):
                    ans+=temp
                    
            self.musal.append(ans/divi)
    
    def sigma(self,x,y):
        m = np.size(y)
        ans = np.zeros((np.size(x)[1],np.size(x)[1]))            
        for i in range(m):
            mus = self.musal[int(y[i])]
            ans += np.dot(np.array([x[i]]).T-mus,np.array([x[i]]- mus.T))
            
        self.sigma = ans/m
            
                
            
                
        

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        self.theta()
        self.mu()
        self.sigma()
        
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        # *** END CODE HERE
