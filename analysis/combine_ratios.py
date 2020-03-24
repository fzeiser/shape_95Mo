import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize
from scipy.interpolate import interp1d


def group_by(a):
    """ Groups array by first collumn, assuming it increases
    Args:
        a: array to group
    Returns:
        list of Ex, list of grouped arrays
    """
    group, length = np.unique(a[:, 0], return_counts=True)
    return group, np.split(a[:, 1:], np.cumsum(length)[:-1])


def fit_func(x, p0, p1):
    return p0 * np.exp(p1*x)


def scale_gsf(gsf, fit_values_x, fit_values_y):
    fgsf = log_interp1d(gsf[:, 0], gsf[:, 1])

    def err(scale):
        diff = (fgsf(fit_values_x) - fit_values_y * scale)
        return np.sum(diff**2)*1e10

    res = minimize(err, x0=0.5)
    return res.x


def log_interp1d(xx, yy, **kwargs):
    """ Interpolate a 1-D function.logarithmically """
    logy = np.log(yy)
    lin_interp = interp1d(xx, logy, kind='linear', **kwargs)
    log_interp = lambda zz: np.exp(lin_interp(zz))  # noqa
    return log_interp


if __name__ == "__main__":
    # format: Ex    x   y
    ratios = np.loadtxt("ratios.txt")

    ratios = ratios[20:, :]

    Ex, groups = group_by(ratios)
    Ngroups = len(groups)
    fits = [curve_fit(fit_func, group[:, 0], group[:, 1])
            for group in groups]

    xs = [group[:, 0] for group in groups]

    scaling = np.ones(Ngroups)
    for i, x in enumerate(xs):
        if i == Ngroups-1:
            break
        xmean = np.mean(x)
        popt_first = fits[i][0]
        popt_next = fits[i+1][0]

        yfirst = fit_func(xmean, *popt_first) * scaling[i]
        ynext = fit_func(xmean, *popt_next) * scaling[i+1]

        scaling[i+1] = yfirst/ynext

    def plot(gsf_scaling=0.05):
        fig, ax = plt.subplots()

        gsf_true = np.loadtxt("../gsf.txt")
        ax.plot(gsf_true[:, 0], gsf_true[:, 1]*gsf_scaling, "k--", alpha=0.7, label="input")

        fit_values_x = np.array([])
        fit_values_y = np.array([])
        for Ex_tmp, x, (popt, pcov), scale in zip(Ex, xs, fits, scaling):
            fit_values_x = np.append(fit_values_x, x)
            fit_values_y = np.append(fit_values_y, fit_func(x, *popt) * scale)
            ax.plot(x, fit_func(x, *popt) * scale, alpha=0.5, label=f"{Ex_tmp}")
            # print(np.mean(x))


        fig.legend(ncol=5)
        ax.set_yscale("log")
    plot()
    plt.show()
