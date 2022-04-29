#/usr/bin/env python3

"""
Author:
- Aditya Bhardwaj <adityabhardwaj727@gmail.com>

This code is part of the project Merton Jump Diffusion
Process for Stock Price Modelling. This work was done in
partial fulfilment of the course MATH F424 (Applied Stochastic Process),
Mathematics Department, BITS Pilani, India

Data:
Stock Market Data gathered from Yahoo Finance
More details can be found at the link provided below
https://github.com/SneakyRain/Jump-Diffusion

"""

import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.optimize as optimize
import matplotlib.pyplot as plt
global epsilon 

epsilon = 1e-10

class ModelCalibration():
    """
    Model Calibration class
    """
    def __init__(self, prices, dt) -> None:
        """
        Init ModelCalibration

        Inputs
        ------
        `prices`: daily stock prices
        `dt`: `1 day` (default) time interval
        """
        self.prices = prices
        self.dt = dt
        self.sampling_freq = "1D"
        self.prices = self.data_preprocessing(self.prices, self.sampling_freq)
        self.returns = self.calculate_log_returns(self.prices.close)

    @staticmethod
    def data_preprocessing(m_data, sampling_freq):
        """
        Pre process the data before analyzing it. This function changes the sampling frequency of ohlc data to given
        sampling frequency.

        Inputs
        ------
        `m_data`: raw market data in ohlc (open-high-low-close) format
        `sampling_freq`: sampling frequency

        Outputs
        -------
        `m_data`: preprocessed market data
        """
        conversion = {'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum' 
        }
        resampled_m_data = m_data.resample(sampling_freq).agg(conversion).dropna()
        return resampled_m_data

    @staticmethod
    def calculate_log_returns(prices):
        """
        Calculate log returns of prices

        Inputs
        ------
        `prices`: variable containing prices

        Outputs
        -------
        `lr`: log return of prices
        """
        lr = np.log(prices/prices.shift(1)).dropna()
        return lr

    def jump_pdf(self, _lambda, mu_d, sig_d, mu_j, sig_j):
        """
        Returns pdf function for returns of stock following `Merton Jump Diffusion Model`

        Inputs
        ------
        `_lambda`:  mean rate of occurence of random `poisson` jumps
        `mu_d`: mean of drift
        `sig_d`: standard deviation of drift
        `mu_j`: mean of normally distributed jumps
        `sig_j`: standard deviation of normally distributed jumps

        Outputs
        -------
        `f`: pdf for given jump diffusion characteristics

        References
        ----------
        1. `Poisson Process`: https://en.wikipedia.org/wiki/Poisson_point_process
        2. `Jump Diffusion Process`: https://en.wikipedia.org/wiki/Jump_diffusion
        """
        def f(x):
            """
            Calculates pdf for given x

            Inputs
            ------
            `x`: x

            Outputs
            -------
            `ans`: pdf value for given input (`x`)
            """
            k = ans = 0
            increment = 1
            while increment >  epsilon:
                pk = stats.distributions.poisson.pmf(k, _lambda*self.dt)
                mean = (mu_d - (sig_d**2)/2)*self.dt + mu_j*k
                std = (sig_d**2)*self.dt + (sig_j**2)*k
                phi = stats.distributions.norm.pdf(x, mean, std)
                increment = pk * phi
                ans = ans + increment
                k = k+1
            if ans == 0:
                ans = epsilon
            return ans
        return f

    def log_likelihood(self, args):
        """
        Calculate negative log likelihood for given args and data

        Inputs
        ------
        `args`: a `List` containing the following parameters
                `_lambda`:  mean rate of occurence of random `poisson` jumps
                `mu_d`: mean of drift
                `sig_d`: standard deviation of drift
                `mu_j`: mean of normally distributed jumps
                `sig_j`: standard deviation of normally distributed jumps

        Outputs
        -------
        `sum`: negative log likelyhood of the jump pdf of daily log returns
        """
        _lambda = args[0]
        mu_d = args[1]
        sig_d = args[2]
        mu_j = args[3]
        sig_j = args[4]
        sum = 0
        self.f = self.jump_pdf(_lambda, mu_d, sig_d, mu_j, sig_j)
        for r in self.returns:
            sum = sum + np.math.log(self.f(r))
        return -sum

if __name__ == "__main__":
    prices = pd.read_csv('data/zomato.csv', parse_dates=True, index_col="timestamp")
    dt = 1
    calibration = ModelCalibration(prices, dt)
    l = calibration.calculate_log_returns(prices.close)
    l = l.reset_index(drop=True)
    plt.plot(l)
    plt.ylabel("Log Returns")
    plt.xlabel("Time")
    plt.show()
    x0 = [1, 1, 1, 1, 1]

    calibration.cons = [
        {'type': 'ineq', 'fun': lambda x: x[0]},
        {'type': 'ineq', 'fun': lambda x: x[2]},
        {'type': 'ineq', 'fun': lambda x: x[4]}
    ]
    calibration.res = optimize.minimize(calibration.log_likelihood, x0, constraints=calibration.cons)
    print(calibration.res.x)
    print(calibration.res.message)
    print(calibration.log_likelihood(calibration.res.x))
    print(calibration.returns.describe())
