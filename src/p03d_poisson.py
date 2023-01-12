import numpy as np
import util
from scipy.stats import poisson
from linear_model import LinearModel
import matplotlib.pyplot as plt


def main():
    """Problem 3(d): Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset("../data/ds4_train.csv", add_intercept=True)
    poimodel = PoissonRegression(theta_0=np.array([[0],[0],[0],[0],[0]]),max_iter=100000,step_size=0.000000001)
    poimodel.fit(x_train,y_train)
    x_valid, y_valid = util.load_dataset("../data/ds4_valid.csv",add_intercept=True)
    pred = poimodel.predict(x_valid)
    x_line = np.linspace(1e6,1e7,10)
    y_line = 1*x_line
    plt.plot(x_line,y_line)
    plt.scatter(pred,y_valid,color='red')
    plt.xlim(1e6, 1e7)
    plt.ylim(1e6, 1e7)
    plt.show()

    # *** START CODE HERE ***
    # Fit a Poisson Regression model
    # Run on the validation set, and use np.savetxt to save outputs to pred_path
    # *** END CODE HERE ***


class PoissonRegression(LinearModel):
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    
    
    def next(self,x,y,num):
        # print(np.exp(float(np.dot(self.theta.T,np.array([x[num]]).T))))
        self.theta = self.theta+self.step_size*(y[num]-np.exp(float(np.dot(self.theta.T,np.array([x[num]]).T))))*np.array([x[num]]).T
    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        count = 0
        num = 0
        while count<self.max_iter:
            self.next(x,y,num)
            num+=1
            num%=np.shape(x)[0]
            count+=1
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Floating-point prediction for each input, shape (m,).
        """
        # *** START CODE HERE ***
        pred = []
        for i in range(np.shape(x)[0]):
            pred.append(np.exp(float(np.dot(self.theta.T,np.array([x[i]]).T))))
            
        return pred
        # *** END CODE HERE ***

main()