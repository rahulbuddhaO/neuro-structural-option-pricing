# Neuro-structural option pricing: learning the volatility smile inside Black-Scholes

This is a small neural network written from scratch in NumPy, which learns the implied volatility smile of 5 stocks **from option prices alone**. The network is placed inside the Black-Scholes model, so it never sees any volatilities, only prices.

The idea is inspired by neuro-structural models, where a neural network learns the hidden parameters of a financial model (here, volatility inside Black-Scholes).

## How it works

```
(moneyness, maturity, which stock)  ->  neural network  ->  sigma  ->  Black-Scholes  ->  price  ->  compare with market price
```

1. The network takes 7 inputs for each option: the 2 option facts (moneyness k and maturity T) and 5 on/off switches for which stock it is (one-hot encoding).
2. It has 1 hidden layer of 16 neurons with tanh, and the output goes through softplus so that sigma is always positive.
3. Black-Scholes turns sigma into a price, and the error is measured against the market price.
4. To train the network, the gradient has to flow back **through Black-Scholes**. This is done by hand using **vega** (how much the option price changes when volatility changes): dPrice/dsigma = S * pdf(d1) * sqrt(T).

## What is written by hand (no deep-learning libraries)

- The forward pass of the network
- Backpropagation through the network **and** through Black-Scholes (the chain rule, using vega)
- The Adam optimiser
- A gradient check, to prove the backpropagation maths is correct

## The data

The data is simulated, so that we know the true answer and can check whether the network finds it.

- 5 hypothetical stocks (Stock A to Stock E) with different prices (30 to 250) and different volatility smiles: each smile has a base level, a skew (higher vol for low strikes) and a curve (higher vol at both ends).
- 5,000 call options with strikes from 70% to 130% of the stock price and maturities from about 1 month to 2 years.
- Prices come from Black-Scholes using each stock's true smile, plus a small amount of noise so they look like real market prices.
- Risk-free rate of 4%, close to the current US rate.
- 80% of the options are used for training and 20% are kept aside for testing.

## Results

**Gradient check** (backpropagation compared with numerical gradients, relative error):

| W1 | b1 | W2 | b2 |
|---|---|---|---|
| 1.5e-10 | 2.0e-10 | 5.4e-11 | 8.6e-11 |

Errors around 1e-10 mean the hand-written gradients are correct, including the step through Black-Scholes.

**Recovering the smile** (error between the network's volatility and the true volatility, on test options it never trained on):

| Stock A | Stock B | Stock C | Stock D | Stock E |
|---|---|---|---|---|
| 0.0016 | 0.0013 | 0.0021 | 0.0016 | 0.0020 |

That is about **0.1 to 0.2 volatility points**, so one network learns all 5 smiles, from prices only. The test error is close to the training error, so the network is not overfitting.

`smiles.png` shows the learned smiles (solid lines) against the true smiles (dashed lines) for 1-year options. The lines sit almost on top of each other for every stock.

## Things to note

- The network learns least about volatility for short-dated options far from the money. Their vega is almost zero, so their prices barely change with volatility and carry very little information about it.
- The data is simulated. Applying the same method to real option prices is the natural next step.

## Next steps

- Train on real option data (for example, option chains from Yahoo Finance), adding dividends to Black-Scholes.
- Compare with a standard neural network that predicts prices directly, without Black-Scholes inside.
- Replace Black-Scholes with the Merton model, so the network learns asset volatility for credit risk and default prediction.

## How to run

```
pip install numpy scipy matplotlib
python neuro_bs.py
```

It runs the gradient check, trains the network (about 10 seconds), prints the results and saves `smiles.png`.
