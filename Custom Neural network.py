# These are the libraries which will be used for the project: numpy for arrays and matrices,
# scipy.stats for the normal distribution and matplotlib for generating plots and graphs.
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

R = 0.04  # This is the risk-free rate, which we have considered as 4%, close to the current US risk-free rate

STOCKS = ["Stock A", "Stock B", "Stock C", "Stock D", "Stock E"]  # These are the stocks we are considering for this project (hypothetical)
S0 = np.array([100.0, 50.0, 250.0, 30.0, 150.0])  # This is an array constructed using numpy; it holds each stock's current price respectively
BASE = np.array([0.18, 0.30, 0.22, 0.40, 0.15])  # This is also an array, which holds the base volatility of each stock respectively
SKEW = np.array([-0.12, -0.20, -0.05, -0.25, -0.10])  # This is an array containing the skew of each stock's smile (higher vol for low strikes)
# CURVE: bend of the smile. Because k^2 is positive on both sides of the money, this term raises vol
# for both deep low and deep high strikes, reflecting fat tails: extreme moves are more likely than Black-Scholes assumes.
CURVE = np.array([0.35, 0.50, 0.20, 0.60, 0.30])
N_STOCKS = len(STOCKS)  # Number of stocks


def bs_call(S, K, T, sigma):
    # Function for Black-Scholes call option pricing, using the variables
    # S - price of the stock, K - strike price, T - maturity in years, sigma - volatility
    d1 = (np.log(S / K) + (R + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))  # distance of the stock from the strike, measured in units of volatility
    d2 = d1 - sigma * np.sqrt(T)  # d2 is d1 shifted down by sigma * sqrt(T)
    price = S * norm.cdf(d1) - K * np.exp(-R * T) * norm.cdf(d2)  # Black-Scholes call price formula
    return price, d1  # we also return d1 because vega needs it


def bs_vega(S, T, d1):
    # Calculating vega - how much the option price changes when the volatility of the stock changes.
    # This is the link that lets the training signal flow back through Black-Scholes into the network.
    return S * norm.pdf(d1) * np.sqrt(T)


def init_params(n_inputs=2 + N_STOCKS, hidden=16, seed=0):
    # This is a function where we set the defaults: inputs are 2 + N_STOCKS, because there are
    # 2 inputs for the two option facts (moneyness k and maturity T) and 5 more inputs, one for each stock;
    # 16 hidden neurons; and the seed is set to 0.
    rng = np.random.default_rng(seed)  # This is a random number generator created using numpy, so we can give random starting weights to the 16 hidden neurons
    return {
        # This is the matrix of starting weights for each neuron, generated from the random number generator.
        # We use Xavier initialisation to keep the weights in a middle range. The size is 7 rows and 16 columns.
        "W1": rng.normal(0, np.sqrt(1 / n_inputs), size=(n_inputs, hidden)),
        # This vector holds the bias for each hidden neuron; all the biases start at 0.
        "b1": np.zeros(hidden),
        # Matrix of starting weights connecting the 16 hidden neurons to the 1 output (sigma), drawn from the
        # random number generator. Xavier initialisation (spread = 1/sqrt(16)) keeps the weights in a middle range.
        # Each weight sets how much one neuron counts towards the final volatility. Shape: 16 rows (neurons) x 1 column (output).
        "W2": rng.normal(0, np.sqrt(1 / hidden), size=(hidden, 1)),
        # Output bias (1 value). Set to -1.5 so that softplus(-1.5) ≈ 0.20,
        # i.e. the untrained network starts with a realistic volatility guess of about 20%.
        "b2": np.full(1, -1.5),
    }


def softplus(z):
    # This function makes any number positive, because volatility can't be negative.
    # It is written this way so the computer never deals with numbers that are too big.
    return np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0)


def forward(params, X):
    # This function passes the option inputs through the network to get a volatility (sigma) for each option.
    Z1 = X @ params["W1"] + params["b1"]  # each of the 16 neurons takes a weighted sum of the 7 inputs and adds its bias -> 16 scores per option
    H1 = np.tanh(Z1)  # tanh squashes each score between -1 and 1; this bend lets the network learn curves like the smile
    Z2 = H1 @ params["W2"] + params["b2"]  # the 16 neuron values are combined into one number per option
    sigma = softplus(Z2[:, 0]) + 0.01  # softplus makes it a positive volatility, and 0.01 is a 1% minimum so sigma is never zero
    return sigma, (X, Z1, H1, Z2)  # we return sigma and also the in-between values, which backpropagation needs later


def true_vol(k, T, stock):
    # This is the hidden "true" volatility smile for each stock. It is only used to create the data and to check
    # the answer at the end; the network never sees it.
    # base level + extra vol for short maturities + skew (tilt) + curve (bend)
    return BASE[stock] + 0.03 * np.exp(-T) + SKEW[stock] * k + CURVE[stock] * k**2


def make_inputs(k, T, stock):
    # This function turns each option into the 7 network inputs: moneyness k, maturity T and 5 stock switches.
    onehot = np.eye(N_STOCKS)[stock]  # one-hot encoding: each stock gets its own on/off switch, e.g. Stock D = [0, 0, 0, 1, 0]
    return np.column_stack([k, T, onehot])  # joins the columns side by side -> 7 inputs per option


def make_data(n=5000, seed=1):
    # This function creates the fake market: 5,000 options across the 5 stocks, priced from each stock's true smile.
    rng = np.random.default_rng(seed)  # random number generator for the data (a different seed from the weights)
    stock = rng.integers(0, N_STOCKS, n)  # assigns each option to a random stock (0 to 4)
    S = S0[stock]  # looks up each option's stock price
    K = S * rng.uniform(0.7, 1.3, n)  # strike prices between 70% and 130% of the stock price
    T = rng.uniform(0.1, 2.0, n)  # maturities between about 1 month and 2 years
    k = np.log(K / S)  # moneyness: 0 = at the money, negative = low strike, positive = high strike
    sigma_true = true_vol(k, T, stock)  # the true volatility of each option
    price, _ = bs_call(S, K, T, sigma_true)  # the true Black-Scholes price of each option
    price = price + S * rng.normal(0, 0.0002, n)  # adds a little noise so the prices look like real market prices
    X = make_inputs(k, T, stock)  # the 7 network inputs for each option
    return X, S, K, T, stock, price, sigma_true


def loss_and_grads(params, X, S, K, T, price_mkt):
    # This function measures how wrong the network is (the loss) and works out how to fix each weight (the gradients).
    sigma, (_, Z1, H1, Z2) = forward(params, X)  # the network's volatility guesses, plus the saved in-between values
    price, d1 = bs_call(S, K, T, sigma)  # turns the guesses into prices with Black-Scholes

    err = (price - price_mkt) / S  # pricing error as a fraction of the stock price, so every stock counts equally
    loss = np.mean(err**2)  # mean squared error: the average squared mistake
    N = len(price)  # number of options

    # Backpropagation: the chain rule, one step at a time, going backwards
    dP = 2 * err / N / S  # step 1: how the loss changes with each price
    dsigma = dP * bs_vega(S, T, d1)  # step 2: how the loss changes with sigma (through vega - the custom step)
    dZ2 = (dsigma / (1 + np.exp(-Z2[:, 0])))[:, None]  # step 3: through softplus (its slope is the sigmoid)

    grads = {}  # dictionary to store the gradient for each weight
    grads["W2"] = H1.T @ dZ2  # step 4: gradient for W2
    grads["b2"] = dZ2.sum(axis=0)  # gradient for b2 (summed over all options)
    dH1 = dZ2 @ params["W2"].T  # step 5: pass the gradient back to the hidden layer
    dZ1 = dH1 * (1 - H1**2)  # step 6: through tanh (its slope is 1 - tanh^2)
    grads["W1"] = X.T @ dZ1  # step 7: gradient for W1
    grads["b1"] = dZ1.sum(axis=0)  # gradient for b1 (summed over all options)
    return loss, grads


def grad_check(params, X, S, K, T, price_mkt, eps=1e-6):
    # This function proves the backpropagation maths is correct. It nudges each weight up and down by a tiny amount,
    # measures how the loss changes, and compares that with the gradient from backpropagation.
    _, grads = loss_and_grads(params, X, S, K, T, price_mkt)  # gradients from backpropagation (the ones being tested)
    for name in params:  # goes through W1, b1, W2 and b2
        P = params[name]
        num = np.zeros_like(P)  # empty array to hold the measured gradients
        for i in np.ndindex(P.shape):  # loops over every single weight
            old = P[i]  # remembers the original value
            P[i] = old + eps  # nudges the weight up
            loss_plus, _ = loss_and_grads(params, X, S, K, T, price_mkt)
            P[i] = old - eps  # nudges the weight down
            loss_minus, _ = loss_and_grads(params, X, S, K, T, price_mkt)
            P[i] = old  # puts the weight back
            num[i] = (loss_plus - loss_minus) / (2 * eps)  # the measured gradient (slope)
        rel = np.linalg.norm(num - grads[name]) / (np.linalg.norm(num) + np.linalg.norm(grads[name]))  # relative difference
        print(f"{name}: relative error = {rel:.2e}")  # around 1e-10 means the backpropagation is correct


def train(epochs=4000, lr=0.005, seed=0):
    # This function teaches the network: it repeats (measure error -> work out the fix -> adjust the weights) 4,000 times.
    X, S, K, T, stock, price, sigma_true = make_data()  # creates the market data
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))  # shuffles the options
    cut = int(0.8 * len(X))
    tr, te = idx[:cut], idx[cut:]  # 80% for training, 20% kept aside for testing

    params = init_params()  # the starting weights
    # Adam optimiser memory, one array per weight: m = running average of gradients, v = running average of squared gradients
    m = {name: np.zeros_like(p) for name, p in params.items()}
    v = {name: np.zeros_like(p) for name, p in params.items()}
    b1, b2, eps = 0.9, 0.999, 1e-8  # standard Adam settings

    for t in range(1, epochs + 1):
        loss, grads = loss_and_grads(params, X[tr], S[tr], K[tr], T[tr], price[tr])  # error and gradients on the training options
        for name in params:
            m[name] = b1 * m[name] + (1 - b1) * grads[name]  # updates the running average of gradients (momentum)
            v[name] = b2 * v[name] + (1 - b2) * grads[name] ** 2  # updates the running average of squared gradients
            m_hat = m[name] / (1 - b1 ** t)  # bias correction, because m and v start at zero
            v_hat = v[name] / (1 - b2 ** t)
            params[name] -= lr * m_hat / (np.sqrt(v_hat) + eps)  # moves each weight a small step in the direction that reduces the error
        if t % 500 == 0:  # every 500 steps, prints the training and test error
            test_loss, _ = loss_and_grads(params, X[te], S[te], K[te], T[te], price[te])
            print(f"epoch {t:4d}  train MSE {loss:.3e}  test MSE {test_loss:.3e}")

    # After training: compares the network's volatility with the true volatility on the test options, for each stock
    sigma_te, _ = forward(params, X[te])
    print("test vol RMSE vs true smile, by stock:")
    for j in range(N_STOCKS):
        mask = stock[te] == j  # picks out the test options for stock j
        err = np.sqrt(np.mean((sigma_te[mask] - sigma_true[te][mask]) ** 2))
        print(f"  {STOCKS[j]}: {err:.4f}")
    return params  # the trained weights


def plot_smiles(params, T_val=1.0):
    # This function draws each stock's learned smile (solid line) against the true smile (dashed line) and saves the chart.
    moneyness = np.linspace(0.7, 1.3, 100)  # 100 points from 70% to 130% of the stock price
    k_grid = np.log(moneyness)
    T_grid = np.full_like(k_grid, T_val)  # the same maturity (1 year) for every point
    plt.figure(figsize=(7, 4.5))
    colours = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    for j in range(N_STOCKS):
        stock = np.full(len(k_grid), j)  # all 100 points belong to stock j
        sigma_learned, _ = forward(params, make_inputs(k_grid, T_grid, stock))  # the network's volatility at each point
        plt.plot(moneyness, true_vol(k_grid, T_grid, stock), "--", color=colours[j])  # true smile (dashed)
        plt.plot(moneyness, sigma_learned, "-", color=colours[j], label=STOCKS[j])  # learned smile (solid)
    plt.xlabel("Moneyness (K / S)")
    plt.ylabel("Implied volatility")
    plt.title(f"Learned (solid) vs true (dashed) smiles, T = {T_val} year")
    plt.legend()
    plt.tight_layout()
    plt.savefig("smiles.png", dpi=150)  # saves the chart in the same folder as this file
    print("saved smiles.png")


if __name__ == "__main__":
    # This runs everything in order when the file is run directly.
    np.set_printoptions(suppress=True)  # prints numbers normally instead of in scientific notation

    print("=== Gradient check ===")
    X, S, K, T, stock, price, sigma_true = make_data(n=500)  # a smaller dataset so the check runs quickly
    grad_check(init_params(), X, S, K, T, price)  # step 1: prove the maths is correct

    print("\n=== Training ===")
    params = train()  # step 2: train the network
    plot_smiles(params)  # step 3: draw the chart
s
