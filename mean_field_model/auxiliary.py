import numpy as np

class AuxiliaryFunctions:
    @staticmethod
    def indicator(x):
        """
        Indicator function that returns an array where elements are 1 if corresponding element in x is positive, 0 otherwise.

        Parameters
        ----------
        x : array-like
            Input array.

        Returns
        -------
        result : array-like
            An array of the same shape as x with 1s where x > 0 and 0s elsewhere.
        """
        positive = x > 0
        result = np.zeros_like(x)
        result[positive] = 1
        return result

    @staticmethod
    def logit(x):
        """
        Computes the logit (log-odds) function.

        Parameters
        ----------
        x : array-like
            Input array where 0 < x < 1.

        Returns
        -------
        array-like
            The logit of the input array.
        """
        return np.log(x / (1 - x))

    @staticmethod
    def sigmoid(x):
        """
        Computes the sigmoid function.

        Parameters
        ----------
        x : array-like
            Input array.

        Returns
        -------
        array-like
            The sigmoid of the input array.
        """
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def dif_sigmoid(x):
        """
        Computes the derivative of the sigmoid function.

        Parameters
        ----------
        x : array-like
            Input array.

        Returns
        -------
        array-like
            The derivative of the sigmoid of the input array.
        """
        return np.exp(-x) / (1 + np.exp(-x)) ** 2

    @staticmethod
    def gauss_density(x):
        """
        Computes the Gaussian density function.

        Parameters
        ----------
        x : array-like
            Input array.

        Returns
        -------
        array-like
            The Gaussian density of the input array.
        """
        return np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi)

    @staticmethod
    def tilde_t_delta(x,delta):
        """
        Computes the smooth approximation of tilde T.

        Parameters
        ----------
        x : array-like
            Input array.

        delta: float.

        Returns
        -------
        array-like
            Smooth approximation of tilde T evaluated at x.
        """
        return 0.5*(np.tanh(x/delta)+1.0)

    @staticmethod
    def dif_tilde_t_delta(x,delta):
        """
        Computes the derivative of the smooth approximation of tilde T.

        Parameters
        ----------
        x : array-like
            Input array.

        delta: float.

        Returns
        -------
        array-like
            Derivative of smooth approximation of tilde T evaluated at x.
        """
        return 0.5*(1-np.tanh(x/delta)**2)/delta