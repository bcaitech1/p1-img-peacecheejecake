from scipy.stats import norm


def age_norm(x, mu, sigma=15):
    z = (x - mu) / sigma
    return norm.pdf(z) / norm.pdf(0)