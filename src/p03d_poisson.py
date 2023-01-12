import numpy as np
import util
from scipy.stats import poisson
from linear_model import LinearModel


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
    poimodel = PoissonRegression(theta_0=np.array([[0],[0],[0],[0],[0]]))
    poimodel.fit(x_train,y_train)
    x_valid, y_valid = util.load_dataset("../data/ds4_valid.csv",add_intercept=True)
    pred = poimodel.predict(x_valid)
    util.plot(np.array(pred),y_valid,np.array([0,1,1]))

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
    
    
    def next(self,x,y):
        l = self.make_list(x,y)
        addtheta = np.zeros((np.shape(x)[1],1))
        for j in range(np.shape(x)[1]):
            for i in range(np.shape(x)[0]):
                print(l[i],x[i][j])
                addtheta[j]+=((l[i])*x[i][j])
        self.theta = self.theta+self.step_size*addtheta
        
    
    
        

    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        prevtheta = self.theta
        count = 0
        while count<self.max_iter:
            self.next(x,y)
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
        for i in x:
            pred.append(np.exp(np.dot(self.theta.T,np.array([i]).T)))
            
        return pred
        # *** END CODE HERE ***

main()