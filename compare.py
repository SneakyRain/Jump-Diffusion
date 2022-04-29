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
from parameters import ModelCalibration

if __name__ == "__main__":
    prices = pd.read_csv('data/infy.csv', parse_dates=True, index_col="timestamp")
    dt = 1
    mc = ModelCalibration(prices, dt)
    
    x0 = [1, 1, 1, 1, 1]

    mc.cons = [
        {'type': 'ineq', 'fun': lambda x: x[0]},
        {'type': 'ineq', 'fun': lambda x: x[2]},
        {'type': 'ineq', 'fun': lambda x: x[4]}
    ]
    mc.res = optimize.minimize(mc.log_likelihood, x0, constraints=mc.cons)
    res = mc.res.x
    f = mc.jump_pdf(res[0], res[1], res[2], res[3], res[4])

    x = np.linspace(-1, 1, 10000)
    lr = mc.calculate_log_returns(prices.close)

    mu = mc.returns.mean()
    sig = mc.returns.std()
    bs = stats.distributions.norm.pdf(x, mu, sig)

    mjd = [f(i) for i in x]
    
    mc.returns.plot.kde(label="Empirical")
    plt.plot(x, mjd, "--", label="Jump Diffusion")
    plt.plot(x, bs, "--", label="Black-Scholes")
    plt.xlim(-0.5, 0.5)
    plt.legend()
    # plt.title("Log Returns Distribution")
    plt.ylabel("f(x)")
    plt.xlabel("x")
    plt.show()