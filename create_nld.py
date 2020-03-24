import numpy as np
import matplotlib.pyplot as plt
import ompy as om
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit


def writeRAINIERnldTable(fname, Ex, nld, sigma, a=None):
    '''
    Takes nld and write it in a format readable for RAINIER

    input:
    fname: outputfilename
    x: Excitation energy in MeV
    nld: nld for above discretes [in 1/MeV]
    sigma: spin-cut
    a: level density parameter a
    '''
    if a is None:
        a = np.zeros(len(Ex))

    fh = open(fname, "w")

    def write_arr(arr):
        # write array to file that resembles the CERN ROOT arrays
        for i, entry in enumerate(arr):
            if i != (len(arr) - 1):
                fh.write(str(entry) + ",\n")
            else:
                fh.write(str(entry) + "\n};\n")

    fh.write("#include \"TGraph.h\"\n\n")
    fh.write("double adETable[] = {\n")
    write_arr(Ex)
    fh.write("const int nETable = sizeof(adETable)/sizeof(double);\n\n")

    fh.write("double adRho[] = {")
    write_arr(nld)
    fh.write("TGraph *grRho = new TGraph(nETable,adETable,adRho);\n\n")

    fh.write("double adLDa[] = {\n")
    write_arr(a)
    fh.write("TGraph *grLDa = new TGraph(nETable,adETable,adLDa);\n\n")

    fh.write("double adJCut[] = {\n")
    write_arr(sigma)
    fh.write("TGraph *grJCut = new TGraph(nETable,adETable,adJCut);\n\n")

    print("Wrote nld fro RAINIER to {}".format(fname))

    fh.close()


def pwrite(fout, array):
    for row in array:
        head = str(int(row[0]))
        tail = " ".join(map(str, row[1:].tolist()))
        sout = head + " " + tail + "\n"
        # print sout
        fout.write(sout)


def WriteRAINIERTotPar(nJs, nStartbin, spinpar_hist, Ex,
                       fname="Js2RAINER.txt"):
    ##############################################
    # Some test -- write files to use with RAINIER
    # print population distribution for RAINIER
    # bin    Ex    Popul.    J= 0.0    J= 1.0  [...] J=9.0

    Jstart = 0
    Js = np.array(range(nJs)) + Jstart
    nRowsOffset = 0  # cut away the first x Rows
    nRows = len(spinpar_hist)
    nCollumsExtra = 3  # 3 rows with "bin  Ex  Popul." added extra"
    nCollums = nJs + nCollumsExtra

    spins_RAINIER = np.zeros((nRows, nCollums))
    # copying spinpar histogram into the historgram that shall be printed
    # number of arrays in the histogram that shall be copied over

    spins_RAINIER[:, 0] = np.array(range(nRows)) + nStartbin  # copy bins
    spins_RAINIER[:, 1] = Ex  # copy excitation energyies

    spins_RAINIER[:, nCollumsExtra:] =\
        spinpar_hist  # copy spins
    spins_array_print = spins_RAINIER[nRowsOffset:, :]

    class prettyfloat(float):
        def __repr__(self):
            return "%0.2f" % self

    arr_Js_print = [("J= " + "{0:.1f}".format(J)) for J in Js]
    arr_header = [("bin"), ("Ex"), ("Popul.")]
    arr_header += arr_Js_print

    # Write spin distribution from Greg
    fout = open(fname, "w")
    fout.write(" ".join(map(str, arr_header)))
    fout.write("\n")
    pwrite(fout, spins_array_print)
    fout.close()


def cross_section(Ex, Emax):
    # triangle
    x = [0, Emax + 1]
    y = np.array([1., 0.8]) * 10
    f = interp1d(x, y)
    return f(Ex)


if __name__ == "__main__":
    # roughly after R. Chankova et al., Phys. Rev. C73, 034311 (2006)
    # this is inconsistent with PHYSICAL REVIEW C 88, 015805 (2013)
    D0 = [1300, 5.]  # eV

    Sn = [7.369, 0.001]  # MeV
    spincutModel = 'EB05'
    spincutPars = {"mass": 95, "NLDa": 10.56, "Eshift": -0.52,
                   "Sn": Sn[0]}
    Jtarget = 0

    # sanity-check
    nldSn_from_D0 = \
        om.NormalizerNLD.nldSn_from_D0(D0=D0, Sn=Sn[0], Jtarget=Jtarget,
                                       spincutModel=spincutModel,
                                       spincutPars=spincutPars)
    print("nldSn_from_D0 [Sn, nld]: ", nldSn_from_D0)
    nldSn = [*nldSn_from_D0, nldSn_from_D0[1]*D0[1]/D0[0]]

    # guess from article
    T = 0.87
    Eshift_CT = -1.2

    Ex = np.linspace(0, Sn[0] + 2, num=200)
    nld_ct = om.NormalizerNLD.const_temperature(E=Ex, T=T, Eshift=Eshift_CT)
    nld = np.loadtxt("oslo-nld.txt", usecols=[1, 2, 3])

    Efit_min, Efit_max = 2., 5.2
    imin = om.index(nld[:, 0], Efit_min)
    imax = om.index(nld[:, 0], Efit_max)
    nld_to_fit = nld[imin:imax+1, :]
    nld_to_fit = np.vstack((nld_to_fit, nldSn))

    nldSn_from_D0
    popt, pcov = curve_fit(om.NormalizerNLD.const_temperature,
                           nld_to_fit[:, 0], nld_to_fit[:, 1],
                           sigma=nld_to_fit[:, 2])


    fig, ax = plt.subplots()
    ax.errorbar(nld[:, 0], nld[:, 1], yerr=nld[:, 2], fmt="o",
                label="oslo-data")
    ax.errorbar(nldSn[0], nldSn[1], yerr=nldSn[2], fmt="o",
                label=r"$\rho(Sn)$ from $D_0$")
    ax.plot(Ex, nld_ct, label="CT-initial")
    ax.plot(Ex, om.NormalizerNLD.const_temperature(Ex, *popt), "--",
            label="CT-fit")
    ax.legend()
    ax.set_yscale("log")
    fig.savefig("nld.png")

    # replace by fit-parameters
    print("popt:", popt)
    T, Eshift_CT = popt


    # create nld and spin-cut
    spinfunctions = om.SpinFunctions(Ex=Ex, J=0, model=spincutModel,
                                     pars=spincutPars)
    sigma = np.sqrt(spinfunctions.get_sigma2())
    nld = om.NormalizerNLD.const_temperature(E=Ex, T=T, Eshift=Eshift_CT)

    print("nldSn_from_model [nld]: {:.2e}".format(
          om.NormalizerNLD.const_temperature(E=Sn[0], T=T, Eshift=Eshift_CT)))

    writeRAINIERnldTable("nld.dat", Ex=Ex, nld=nld,
                         sigma=sigma)


    # write initial population table for RAINIER
    Emin = 1.074 + 0.1  # level 10 + a little
    Emax = Sn[0]  # Sn
    Emax_xs = Emax + 1
    nStartbin = 11  # level number to Emin
    Nspins = 20
    Exs = np.linspace(Emin, Emax, num=70)
    matrix = np.zeros((len(Exs), Nspins))
    Js = np.arange(Nspins)

    spinfunctions.J = Js

    for i, Ex in enumerate(Exs):
        spinfunctions.Ex = np.atleast_1d(Ex)
        matrix[i, :] = (spinfunctions.distibution()
                        * cross_section(Ex, Emax_xs))
    matrix = np.round(matrix, 5)
    Exs = np.round(Exs, 3)
    WriteRAINIERTotPar(nJs=Nspins, nStartbin=nStartbin,
                       Ex=Exs,
                       spinpar_hist=matrix,
                       fname="Js2RAINER.txt")


    plt.show()
