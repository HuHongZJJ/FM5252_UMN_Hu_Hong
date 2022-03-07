import pandas as pd
from scipy.stats import norm
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import scipy.optimize
# section 1

def CallPrice(S, T, K, r, sig):
    d1 = (np.log(S / K) + (r + 0.5 * sig ** 2) * (T)) / (sig * np.sqrt(T))
    d2 = d1 - sig * np.sqrt(T)
    cprice = S * norm.cdf(d1) - np.exp(-r * (T)) * K * norm.cdf(d2)
    return cprice

def PutPrice(S, T, K, r, sig):
    d1 = (np.log(S / K) + (r + 0.5 * sig ** 2) * (T)) / (sig * np.sqrt(T))
    d2 = d1 - sig * np.sqrt(T)
    pprice = np.exp(-r * (T)) * K * norm.cdf(-d2) - S * norm.cdf(-d1)
    return pprice

def bisection_method(type, S, T, K, r, sig_lowbound, sig_highbound, optionprice):
    error_allowed = 0.00005
    sigma_low = sig_lowbound
    sigma_high = sig_highbound
    if type == "put":

        while sigma_high - sigma_low > error_allowed:
            if PutPrice(S, T, K, r, (sigma_high + sigma_low) / 2) - optionprice > 0:
                sigma_high = (sigma_high + sigma_low) / 2
            else:
                sigma_low = (sigma_high + sigma_low) / 2

    elif type == "call":
        while sigma_high - sigma_low > error_allowed:
            if CallPrice(S, T, K, r, (sigma_high + sigma_low) / 2) - optionprice > 0:
                sigma_high = (sigma_high + sigma_low) / 2
            else:
                sigma_low = (sigma_high + sigma_low) / 2

    implied_vol = sigma_low
    return implied_vol

def newton_method(type, S, T, K, r, sig, optionprice):
    error_allowed = 0.00005
    sigma = sig
    d1 = (np.log(S / K) + (r + sig ** 2 / 2) * T) / (sig * np.sqrt(T))
    if type == "put":
        while np.abs(PutPrice(S, T, K, r, sigma) - optionprice) > error_allowed:
            vega = K * np.exp(-r * T) * norm.pdf(d1) * np.sqrt(T)
            sigma = sigma - ((PutPrice(S, T, K, r, sigma) - optionprice) / vega)

    elif type == "call":
        while np.abs(CallPrice(S, T, K, r, sigma) - optionprice) > error_allowed:
            vega = K * np.exp(-r * T) * norm.pdf(d1) * np.sqrt(T)
            sigma = sigma - ((CallPrice(S, T, K, r, sigma) - optionprice) / vega)

    implied_vol = sigma
    return implied_vol

print("section 1")

def run_func_1(type, S, T, K, r, sig_lowbound, sig_highbound, optionprice):
    sig = (sig_lowbound + sig_highbound)/2
    bisection = bisection_method(type, S, T, K, r, sig_lowbound, sig_highbound, optionprice)
    newton = newton_method(type, S, T, K, r, sig, optionprice)
    print(f"{type} option with Strike Price ${K} with T = {T} have implied volatility using bisection method {bisection} and using newton method {newton}")
run_func_1("put", 25, 1, 20, 0.05, 0, 2, 8)
# section 2
aapl = yf.Ticker("AAPL")
opt = aapl.option_chain('2022-06-17')
call = opt.calls

def get_vols_skew(ticker, date):
    underly = yf.Ticker(ticker)
    opt = underly.option_chain(date)
    call_data = opt.calls
    put_data = opt.puts
    effective_vol_call = []
    strike_call = []
    effective_vol_put = []
    strike_put = []
    for i in call_data.iterrows():
        if i[1][11] == True and i[1][10] > 0.00005 and i[1][10] < 1.5:

            effective_vol_call.append(i[1][10])
            strike_call.append(i[1][2])

    for i in put_data.iterrows():
        if i[1][11] == True and i[1][10] > 0.00005 and i[1][10] < 1.5:

            effective_vol_put.append(i[1][10])
            strike_put.append(i[1][2])
    current_price = (np.min(strike_put) + np.max(strike_call))/2
    effective_vol = effective_vol_put + effective_vol_call
    strike = strike_put + strike_call
    return effective_vol, strike, current_price

vol = get_vols_skew("AAPL","2022-06-17")

strike = vol[1]
impliedvol = vol[0]
underlyprice = vol[2]

def sviCurve (x, a, b, rho, m,sigma):

    #if b > 0 and np.abs(rho) < 0 and sigma>0:
    result = a+b*(rho*(x-m)+np.sqrt((x-m)**2 + sigma**2))
    
    return result

def sviFit (impliedvol,strikes):
    o = scipy.optimize.curve_fit(sviCurve, strikes, impliedvol, maxfev= 1000000)
    return o
print("section 2")

def runfunc_2(ticker,date):
    vol = get_vols_skew(ticker, date)

    strike = vol[1]
    impliedvol = vol[0]
    underlyprice = vol[2]
    param = sviFit(impliedvol, strike)[0]
    print(f"The a is {param[0]}, b is {param[1]}, rho is {param[2]}, m is {param[3]}, sigma is {param[4]}")


runfunc_2("AAPL", "2022-06-17")
