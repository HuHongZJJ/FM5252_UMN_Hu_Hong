import numpy as np
from scipy.stats import binom
from scipy.stats import norm


def eurooptions(kind, S0, T, r, sig, M, K):
    # kind: type of option
    # S0: underlying price at T0
    # T: days to maturity
    # r: risk-free rate
    # sig: volatility (sigma)
    # M: number of steps
    # K: strike price
    if kind == "European Call":
        dt = T / M
        u = np.exp(sig * np.sqrt(dt))
        k = np.arange(0, M + 1)
        svals = S0 * u ** (2 * k - int(M))
        pstar = (u * np.exp(r * dt) - 1) / (u ** 2 - 1)
        payoffs = np.maximum(svals - K, 0)
        probs = binom.pmf(k, n=M, p=pstar)
        eurocallvalue = (probs @ payoffs) * np.exp(-M * r * dt)
        return eurocallvalue
    elif kind == "European Put":
        dt = T / M
        u = np.exp(sig * np.sqrt(dt))
        k = np.arange(0, M + 1)
        svals = S0 * u ** (2 * k - int(M))
        pstar = (u * np.exp(r * dt) - 1) / (u ** 2 - 1)
        payoffs = np.maximum(K - svals, 0)
        probs = binom.pmf(k, n=M, p=pstar)
        europutvalue = (probs @ payoffs) * np.exp(-M * r * dt)
        return europutvalue


def amerioptions(kind, S0, T, r, sig, M, K):
    # kind: type of option
    # S0: underlying price at T0
    # T: days to maturity
    # r: risk-free rate
    # sig: volatility (sigma)
    # M: number of steps
    # K: strike price
    dt = T / M
    u = np.exp(sig * np.sqrt(dt))
    d = 1 / u
    disc = np.exp(-r * dt)
    pstar = (u * np.exp(r * dt) - 1) / (u ** 2 - 1)
    Svals = S0 * d ** (np.arange(M, -1, -1)) * u ** (np.arange(0, M + 1, 1))
    if kind == "American Call":
        payoff = np.maximum(Svals - K, 0)
    elif kind == "American Put":
        payoff = np.maximum(K - Svals, 0)
    N = M

    def iter(kind, N, disc, K, payoff):
        if kind == "American Call":
            if N > 1:
                N -= 1
                Svals = S0 * d ** (np.arange(N, -1, -1)) * u ** (np.arange(0, N + 1, 1))
                payoff[:N + 1] = disc * (pstar * payoff[1:N + 2] + (1 - pstar) * payoff[0:N + 1])
                payoff = payoff[:-1]
                payoff = np.maximum(payoff, Svals - K)
                return iter("American Call", N, disc, K, payoff)
            elif N == 1:
                Svals = S0 * d ** (np.arange(N, -1, -1)) * u ** (np.arange(0, N + 1, 1))
                payoff[:N + 1] = disc * (pstar * payoff[1:N + 2] + (1 - pstar) * payoff[0:N + 1])
                payoff = payoff[:-1]
                payoff = np.maximum(payoff, Svals - K)
                return payoff[0]
        if kind == "American Put":
            if N > 1:
                N -= 1
                Svals = S0 * d ** (np.arange(N, -1, -1)) * u ** (np.arange(0, N + 1, 1))
                payoff[:N + 1] = disc * (pstar * payoff[1:N + 2] + (1 - pstar) * payoff[0:N + 1])
                payoff = payoff[:-1]
                payoff = np.maximum(payoff, K - Svals)
                return iter("American Put", N, disc, K, payoff)
            elif N == 1:
                Svals = S0 * d ** (np.arange(N, -1, -1)) * u ** (np.arange(0, N + 1, 1))
                payoff[:N + 1] = disc * (pstar * payoff[1:N + 2] + (1 - pstar) * payoff[0:N + 1])
                payoff = payoff[:-1]
                payoff = np.maximum(payoff, K - Svals)
                return payoff[0]

    result = iter(kind, N, disc, K, payoff)
    return result


def greeks(kind, S, K, T, r, sig):
    d1 = lambda S, K, T, r, sig: (np.log(S / K) + (r + sig ** 2 / 2) * T) / (sig * np.sqrt(T))
    d2 = lambda S, K, T, r, sig: d1(S, K, T, r, sig) - sig * np.sqrt(T)
    if kind == "American Call" or kind == "European Call":
        call_delta = lambda S, K, T, r, sig: norm.cdf(d1(S, K, T, r, sig))
        gamma = lambda S, K, T, r, sig: norm.pdf(d1(S, K, T, r, sig)) / (S * sig * np.sqrt(T))
        vega = lambda S, K, T, r, sig: S * norm.pdf(d1(S, K, T, r, sig)) * np.sqrt(T)
        call_theta = lambda S, K, T, r, sig: -(S * norm.pdf(d1(S, K, T, r, sig)) * sig) / (
                    2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2(S, K, T, r, sig))
        call_rho = lambda S, K, T, r, sig: K * T * np.exp(-r * T) * norm.cdf(d2(S, K, T, r, sig))
        return call_delta(S, K, T, r, sig), gamma(S, K, T, r, sig), vega(S, K, T, r, sig), call_theta(S, K, T, r, sig), call_rho(S, K, T, r, sig)
    elif kind == "American Put" or kind == "European Put":
        put_delta = lambda S, K, T, r, sig: -norm.cdf(-d1(S, K, T, r, sig))
        gamma = lambda S, K, T, r, sig: norm.pdf(d1(S, K, T, r, sig)) / (S * sig * np.sqrt(T))
        vega = lambda S, K, T, r, sig: S * norm.pdf(d1(S, K, T, r, sig)) * np.sqrt(T)
        put_theta = lambda S, K, T, r, sig: -(S * norm.pdf(d1(S, K, T, r, sig)) * sig) / (
                    2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2(S, K, T, r, sig))
        put_rho = lambda S, K, T, r, sig: -K * T * np.exp(-r * T) * norm.cdf(-d2(S, K, T, r, sig))
        return put_delta(S, K, T, r, sig), gamma(S, K, T, r, sig), vega(S, K, T, r, sig), put_theta(S, K, T, r, sig), put_rho(S, K, T, r, sig)


def runfunc(kind, S0, T, r, sig, M, K):
    if kind == "American Call" or kind == "American Put":
        return amerioptions(kind, S0, T, r, sig, M, K), greeks(kind, S0, K, T, r, sig)
    elif kind == "European Call" or kind == "European Put":
        return eurooptions(kind, S0, T, r, sig, M, K), greeks(kind, S0, K, T, r, sig)


print("When input option type please type excatly one of following strings:")
print("American Call")
print("American Put")
print("European Call")
print("European Put")
kind = input("What is the kind option price you would like to check?   ")
S0 = float(input("What is the current underlying price?   "))
K = float(input("What is the strike price?   "))
T = float(input("What days to maturity?   "))
r = float(input("What is the risk-free rate?(range from 0 to 1)   "))
sig = float(input("What is the volatility?   "))
M = int(input("How many steps of binomial simulation that you what?   "))
finaldata = list(runfunc(kind, S0, T, r, sig, M, K))
print(f"{kind} of underlying at ${S0} with strike price ${K} is priced at ${finaldata[0]}.")
print(f"{kind}'s delta is {list(finaldata[1])[0]}, gamma is {list(finaldata[1])[1]}, vega is {list(finaldata[1])[2]}, theta is {list(finaldata[1])[3]} and rho is {list(finaldata[1])[4]}.")

# for the european option part, I use some of code from FM 5151 Class in last semester
# for the amerivan option part, some of the idea are from this wedsite https://asxportfolio.com/options-binomial-trees-american-option-pricing
