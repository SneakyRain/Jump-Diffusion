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

import bisect
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

class MonteCarlo():
    """
    Monte Carlo class
    """
    def __init__(self, T, num_steps, num_sim) -> None:
        """
        Init MonteCarlo

        Inputs
        ------
        `T`: total time
        `num_steps`: number of total steps
        `num_sim`: number of simulation to run
        """
        self.num_steps = num_steps
        self.T = T
        self.num_sim = num_sim
        self.dt = self.T/self.num_steps

    def poisson_process(self, _lambda):
        """
        Simulate Poisson Process

        Inputs
        ------
        `_lambda`:  mean rate of occurence of random `poisson` jumps

        Outputs
        -------
        `values`: simulated values of poisson process

        References
        ----------
        1. `Poisson Process`: https://en.wikipedia.org/wiki/Poisson_point_process
        """
        event_times = []
        events = []
        t = count = 0
        while True:
            tau = stats.distributions.expon(scale=1/_lambda)
            t = t + tau.rvs()
            count = count + 1
            if t <= self.T:
                event_times.append(t)
                events.append(count)
            else:
                break
        
        self.tgrid = np.linspace(0, self.T, self.num_steps+1)
        values = []
        for t_i in self.tgrid:
            id = bisect.bisect(event_times, t_i)
            if id == 0:
                v = 0
            else:
                v = events[id-1]
            values.append(v)
        values = np.array(values)
        return values

    def compound_poisson_process(self, _lambda, f):
        """
        Simulate Compound Poisson Process

        Inputs
        ------
        `_lambda`:  mean rate of occurence of random `poisson` jumps
        `f`: law for jump intensity

        Outputs
        -------
        `values`: simulated values of compound poisson process

        References
        ----------
        1. `Compound Poisson Process`: https://en.wikipedia.org/wiki/Compound_Poisson_process
        """
        event_times = []
        events = []
        t = count = 0
        while True:
            tau = stats.distributions.expon(scale=1/_lambda)
            t = t + tau.rvs()
            jump = f.rvs()
            count = count + jump
            if t <= self.T:
                event_times.append(t)
                events.append(count)
            else:
                break
        
        self.tgrid = np.linspace(0, self.T, self.num_steps+1)
        values = []
        for t_i in self.tgrid:
            id = bisect.bisect(event_times, t_i)
            if id == 0:
                v = 0
            else:
                v = events[id-1]
            values.append(v)
        values = np.array(values)
        return values

    def geometric_brownian_motion(self, mu, sigma, S0):
        """
        Simulate Geoetric Brownian Motion

        Inputs
        ------
        `mu`: mean of drift
        `sigma`: std of diffusion
        `S0`: initial stock price

        Outputs
        -------
        `values`: simulated values of brownian motion

        References
        ----------
        1. Geometric Brownian Motion: https://en.wikipedia.org/wiki/Geometric_Brownian_motion
        """
        self.tgrid  = np.linspace(0, self.T, self.num_steps+1)
        values = np.exp(
            (mu - 0.5*(sigma**2)) * self.dt
            + sigma * np.random.normal(0, np.sqrt(self.dt), size=(self.num_sim, self.num_steps)).T
        )
        values = np.vstack([np.ones(self.num_sim), values])
        values = S0 * values.cumprod(axis=0)
        return values

    def get_gbm_log_increments(self, mu, sigma):
        """
        Simulate Geoetric Brownian Motion log-returns

        Inputs
        ------
        `mu`: mean of drift
        `sigma`: std of diffusion

        Outputs
        -------
        `ts`: time steps
        `ys`: simulated values of brownian motion

        References
        ----------
        1. Geometric Brownian Motion: https://en.wikipedia.org/wiki/Geometric_Brownian_motion
        """
        # simulation using numpy arrays for the geometiric brownian motion
        ys = [0]
        inc = 0

        for i in range(self.num_steps):
            inc += (mu - sigma ** 2 / 2) * self.dt + sigma * np.random.normal(0, np.sqrt(self.dt))
            ys.append(inc)

        ts = np.linspace(0, self.T, self.num_steps)
        return ts, ys


    def jump_process(self, _lambda, mu_d, sig_d, mu_j, sig_j, S0):
        """
        Simulate Jump Diffusion Process
        
        Inputs
        ------
        `_lambda`:  mean rate of occurence of random `poisson` jumps
        `mu_d`: mean of drift
        `sig_d`: standard deviation of drift
        `mu_j`: mean of normally distributed jumps
        `sig_j`: standard deviation of normally distributed jumps
        `S0`: initial stock price
        
        Outputs
        -------
        `values`: simulated values of jump diffusion

        References
        ----------
        1. `Jump Diffusion Process`: https://en.wikipedia.org/wiki/Jump_diffusion
        """
        
        self.tgrid = np.linspace(0, self.T, self.num_steps+1)
        t_gbm, y_gbm = self.get_gbm_log_increments(mu_d, sig_d)
        
        f = stats.distributions.norm(loc=mu_j, scale=sig_j)
        compound = self.compound_poisson_process(_lambda, f)
        compound = np.array(compound)
        # print(compound[1:])
        # print(compound[:-1])
        increments = compound[1:] - compound[:-1]
        increments = np.insert(increments, 0, 0)

        log_returns = []
        for i, _ in enumerate(self.tgrid):
            r = y_gbm[i] + increments[i]
            log_returns.append(r)

        values = S0 * np.exp(log_returns)
        return values

if __name__ == "__main__":
    mc = MonteCarlo(10, 1000, 1)
    f = stats.distributions.norm(0, 1)
    val = mc.poisson_process(1)
    plt.plot(mc.tgrid, val)
    val = mc.poisson_process(1)
    plt.plot(mc.tgrid, val)
    val = mc.poisson_process(1)
    plt.plot(mc.tgrid, val)
    plt.show()
    
