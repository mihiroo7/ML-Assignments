import numpy as np
import util

from linear_model import LinearModel


def main():
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset("../data/ds2_train.csv", add_intercept=False)
    x_valid, y_valid = util.load_dataset("../data/ds2_valid.csv", add_intercept=False)
    model = GDA()
    model.fit(x_train,y_train)
    pred = model.predict(x_valid)
    print(model.accuracy(pred,y_valid))


    # *** START CODE HERE ***
    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self,phi = None,varia=None,musal =None,thetazero = None, *args, **kwargs):
        self.varia= varia
        self.musal = musal
        self.phi= phi
        self.thetazero = None
        super(GDA, self).__init__(*args, **kwargs)
    
    def phif(self,y):
        ans =0
        for i in y:
            if(int(i)==1):
                ans+=1
        self.phi = ans/np.size(y)
    
    def mu(self,x,y) :
        self.musal = []
        for k in range(np.shape(x)[1]):        
            ans  = np.zeros((np.shape(x)[1],1))
            divi = 0;
            for i in y:
                if int(i)==k:
                    divi+=1
                    
            for i in range(np.shape(x)[0]):
                temp = np.array([x[i]]).T
                if(int(y[i])==k):
                    ans+=temp
                    
            self.musal.append(ans/divi)
    
    def sigma(self,x,y):
        m = np.size(y)
        ans = np.zeros((np.shape(x)[1],np.shape(x)[1]))            
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
        self.phif(y)
        self.mu(x,y)
        self.sigma(x,y)
        
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        sigmainv = np.linalg.inv(self.sigma)
        self.theta = np.dot(sigmainv,(self.musal[1]-self.musal[0]))
        self.thetazero = np.dot(np.dot(self.musal[0].T,sigmainv),self.musal[1]) - np.dot(np.dot(self.musal[1].T,sigmainv),self.musal[0])-np.log((1-self.phi)/self.phi)
        y = np.zeros((np.shape(x)[0],1))
        for i in range(np.shape(x)[0]):
            y[i][0] = 1/(1+np.exp(-(np.dot(self.theta.T,np.array([x[i]]).T)+self.thetazero)))
            
        return y
    
    def accuracy(self,pred,y):
        ans = 0 
        for i in range(np.size(y)):
            ans+= abs(pred[i][0]-y[i])
        return 1-(ans)/np.size(y)
            
        # *** END CODE HERE


main()