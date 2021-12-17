import numpy as np
import util


def main(train_path, valid_path, save_path):
    """Problem: Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Train a logistic regression classifier
    clf = LogisticRegression()
    clf.fit(x_train,y_train)

    # Plot decision boundary on top of validation set
    x_eval,y_eval = util.load_dataset(valid_path,add_intercept=True)
    plot_path = save_path.replace('txt','jpg')
    util.plot(x_eval,y_eval,clf.theta,plot_path)

    # Use np.savetxt to save predictions on eval set to save_path
    p_eval = clf.predict(x_eval)
    y_hat = p_eval > 0.5
    accuracy = np.mean((y_hat == 1) == (y_eval == 1))
    print(f'Acuuracy {accuracy}') 
    np.savetxt(save_path,p_eval)

    #Plot decision boundary on top of training dataset
    save_path_train = save_path.strip('.txt')
    save_path_train = save_path_train+'_train.jpg'
    util.plot(x_train,y_train,clf.theta,save_path_train)
    
    # *** END CODE HERE ***


class LogisticRegression:
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=10000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        (m,n) = x.shape
        if self.theta == None:
            self.theta = np.zeros(n,dtype=np.float32)

        for i in range(self.max_iter):
            grad = self._gradient(x,y)
            hess = self._hessian(x)
            prev_theta = np.copy(self.theta)
            self.theta -= self.step_size*np.matmul(np.linalg.inv(hess),grad)
            loss = self._loss(x,y)
            if self.verbose:
                print(f'iteration {i}, loss {loss}')
            if np.sum(np.abs(prev_theta-self.theta)) < self.eps:
                break

        if self.verbose:
            print(f'final theta (logreg): {self.theta}')


    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        y_hat = self._sigmoid(x.dot(self.theta))
        return y_hat

    def _hessian(self,x):
        (m,n) = x.shape
        probs = self._sigmoid(np.dot(x,self.theta))
        diags = np.diag(probs*(1.-probs))
        hess = (1/m)*np.matmul(np.matmul(x.T,diags),x)
        return hess

    def _gradient(self,x,y):
        (m,n) = x.shape
        probs = self._sigmoid(np.dot(x,self.theta))
        grad = (1/m)*np.matmul(x.T,(probs-y))
        return grad

    def _loss(self,x,y):
        hx = self._sigmoid(np.dot(x,self.theta))
        loss = -np.mean(y*np.log(hx+self.eps)+(1-y)*np.log(1-hx+self.eps))  # Added self.eps so that log is not computed for a 0 value. For numerical stability.
        return loss

    def _sigmoid(self,z):
        return 1/(1+np.exp(-z))

if __name__ == '__main__':
    main(train_path='ds1_train.csv',valid_path='ds1_valid.csv',save_path='logreg_pred_1.txt')

    main(train_path='ds2_train.csv',valid_path='ds2_valid.csv',save_path='logreg_pred_2.txt')
