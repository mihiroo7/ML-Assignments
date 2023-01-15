import matplotlib.pyplot as plt
import numpy as np
import util

from linear_model import LinearModel


def main(tau, train_path, eval_path):
    """Problem 5(b): Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    model = LocallyWeightedLinearRegression(tau= tau)
    model.fit(x_train,y_train)
    x_valid, y_valid = util.load_dataset(eval_path, add_intercept=True)
    pred = model.predict(x_valid)
    
    x_train = np.delete(x_train,0,1)
    x_train = x_train.flatten()
    x_valid = np.delete(x_valid,0,1)
    x_valid = x_valid.flatten()
    plt.scatter(x_train,y_train,color='red')
    plt.scatter(x_valid,y_valid,color='blue')
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.show()
 

    # *** START CODE HERE ***
    # Fit a LWR model
    # Get MSE value on the validation set
    # Plot validation predictions on top of training set
    # No need to save predictions
    # Plot data
    # *** END CODE HERE ***


class LocallyWeightedLinearRegression(LinearModel):
    """Locally Weighted Regression (LWR).

    Example usage:
        > clf = LocallyWeightedLinearRegression(tau)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Fit LWR by saving the training set.

        """
        # *** START CODE HERE ***
        if self.theta == None :
            self.theta = np.zeros((np.shape(x)[1],1))
            self.x = x
            self.y = y
            

        # *** END CODE HERE ***

    def predict(self, x):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        Y = np.array([self.y])
        Y = Y.T  
        n = np.shape(x)[0]
        X = self.x
        pred = []
        for i in range(n):
            w = []
            for j in range(np.shape(self.x)[0]):
                l = []
                for k in range(np.shape(self.x)[0]):
                    if j==k: l.append(np.exp(-1*(np.linalg.norm(x[i]- x[k]))/(2*self.tau**2)))
                    else : l.append(0)
            
            w = np.array(w)
            self.theta = 0.5*np.dot(np.linalg.inv(np.dot(X.T,np.dot(w,X))),np.dot(np.dot(Y.T,np.dot(w,X)))+np.dot(np.dot(X.T,np.dot(w,Y))))
            pred.append(np.dot(self.theta,x[i]))
        
        # *** END CODE HERE ***

main(0.5, "../data/ds5_train.csv","../data/ds5_valid.csv")