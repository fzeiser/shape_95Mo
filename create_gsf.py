import numpy as np
import matplotlib.pyplot as plt
from numpy import pi


def writeRAINIERgsfTable(fname, Eg, E1, M1=None, E2=None):
    '''
    Takes gamma-ray strength function f and write it in a format readable for RAINIER

    input:
    fname: outputfilename
    x: Gamma-ray energy in MeV
    E1: E1 component [in 1/MeV^3]
    M1: M1 component [in 1/MeV^3]
    E2: E2 component, default set to 0
    '''
    if M1 is None:
        M1 = np.zeros(len(Eg))

    if E2 is None:
        E2 = np.zeros(len(Eg))

    fh = open(fname, "w")

    def write_arr(arr):
        # write array to file that resembles the CERN ROOT arrays
        for i, entry in enumerate(arr):
            if i != (len(arr) - 1):
                fh.write(str(entry) + ",\n")
            else:
                fh.write(str(entry) + "\n};\n")

    fh.write("#include \"TGraph.h\"\n\n")
    fh.write("double adGSF_ETable[] = {\n")
    write_arr(Eg)
    fh.write("const int nGSF_ETable = sizeof(adGSF_ETable)/sizeof(double);\n\n")

    fh.write("double adGSF_E1[] = {\n")
    write_arr(E1)
    fh.write("TGraph *grGSF_E1 = new TGraph(nGSF_ETable,adGSF_ETable,adGSF_E1);\n\n")

    fh.write("double adGSF_M1[] = {\n")
    write_arr(M1)
    fh.write("TGraph *grGSF_M1 = new TGraph(nGSF_ETable,adGSF_ETable,adGSF_M1);\n\n")

    fh.write("double adGSF_E2[] = {\n")
    write_arr(E2)
    fh.write("TGraph *grGSF_E2 = new TGraph(nGSF_ETable,adGSF_ETable,adGSF_E2);\n\n")

    print("Wrote gSF fro RAINIER to {}".format(fname))

    fh.close()


# commonly used const. strength_factor, convert in mb^(-1) MeV^(-2)
strength_factor = 8.6737E-08 # (1 / (3π²ħ²c²)  ?)


def SLO(E, E0, sigma0, Gamma0):
    # Standard Lorentzian,
    # adapted from Kopecky & Uhl (1989) eq. (2.1)
    f = (strength_factor * sigma0 * E * Gamma0**2
         / ((E**2 - E0**2)**2 + E**2 * Gamma0**2))
    return f


def GLO(E, E0, sigma0, Gamma0, T):
    # Generalized Lorentzian,
    # adapted from Kopecky & Uhl (1989) eq. (2.3-2.4)
    Gamma = Gamma0 * (E**2 + 4 * pi**2 * T**2) / E0**2
    f1 = (E * Gamma) / ((E**2 - E0**2)**2 + E**2 * Gamma**2)
    f2 = 0.7 * Gamma0 * 4 * pi**2 * T**2 / E0**5

    f = strength_factor * sigma0 * Gamma0 * (f1 + f2)

    return f

def KMF(E, E0, sigma0, Gamma0, T):
    """ Kadmenskij, Markushev and Furman model, Yad. Fiz. 37, 277-283 (1983)

    Note: here with constant temperature
    """
    Gamma_k = Gamma0*Gamma0*E**2 + (2*np.pi*T)**2 / E0
    denominator_rest = (E**2. - E0**2.)**2

    f_KMF = strength_factor*0.7*sigma0*Gamma_k / denominator_rest
    return f_KMF

def softpole(E, A, b):
    return strength_factor * A * E**(-b)


if __name__ == "__main__":
    # Parameters from Renstrom2018 article
    #               [omega, sigma, gamma]
    E1  = np.array([16.28, 185.0, 5.76])  # noqa
    T = 0.82
    M1   = np.array([8.99, 1.38, 4.0]) # noqa
    E2  = np.array([13.81, 2.22, 4.97])  # noqa

    A = 0.48
    b = 2.6
    kappa = 0.39 * 0.75  # reduced with regards to Guttormsen2005

    E = np.linspace(0.1, 10, num=200)
    fE1 = GLO(E, *E1, T=T)
    fM1 = SLO(E, *M1) + softpole(E, A, b)
    fE2 = 3/(5*E**2) * SLO(E, *E2)

    fE1 *= kappa
    fM1 *= kappa

    writeRAINIERgsfTable("gsf.dat", Eg=E, E1=fE1, M1=fM1, E2=fE2)
    np.savetxt("gsf.txt", np.c_[E, fE1+fM1+fE2, fE1, fM1, fE2],
               header="E fE1+fM1 fE1 fM1 fE2")

    fig, ax = plt.subplots()
    ax.semilogy(E, fE1+fM1+fE2, label="E1 + M1 + E2")

    gsf_oslo = np.loadtxt("oslo-gsf.txt", usecols=[1, 2, 3])

    ax.errorbar(gsf_oslo[:, 0], gsf_oslo[:, 1],
                yerr=gsf_oslo[:, 2], fmt="o", label="oslo-data")
    ax.semilogy(E, fE1, "--", label="E1")
    ax.semilogy(E, fM1, "--", label="M1")
    ax.semilogy(E, fE2, "--", label="E2")
    ax.legend()
    fig.savefig("gsf.png")
    plt.show()
