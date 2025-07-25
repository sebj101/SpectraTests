import numpy as np
import scipy.constants as const
import pymc as pm
import pytensor.tensor as pt


def ApproxdGammadE(mNu: float, E0: float, energies: np.ndarray) -> np.ndarray:
    """
    Approximate the decay rate dGamma/dE for a given neutrino mass and endpoint energy.

    Parameters
    ----------
    mNu : float
        Neutrino mass in eV.
    E0 : float
        Endpoint energy in eV.
    energies : np.ndarray
        Energies at which to evaluate the decay rate in eV.

    Returns
    -------
    np.ndarray
        Approximated decay rates at the specified energies.
    """
    ME_EV = const.m_e * const.c**2 / const.e
    beta = np.sqrt(1.0 - (ME_EV / (energies + ME_EV))**2)
    p = np.sqrt(energies**2 + 2.0 * energies * ME_EV)
    eta = const.alpha * 2 / beta
    fermiFunc = 2 * np.pi * eta / (1. - np.exp(-2 * np.pi * eta))
    nuE = E0 - energies
    fermiConst = 1.1663787e-23  # eV^-2
    rate = fermiConst**2 * 0.97425**2 * fermiFunc * \
        (1 + 3 * (-1.2646)**2) * p * (energies + ME_EV) / (2 * np.pi**3)
    rate *= 1.0 / 6.58e-16  # Account for natural units
    nuE = np.clip(nuE, 0.0, None)  # Ensure non-negative for sqrt
    arg = np.clip(nuE**2 - mNu**2, 0.0, None)  # Ensure non-negative for sqrt

    return rate * nuE * np.sqrt(arg)  # dGamma/dE in eV^-1


def IntegrateSpectrumFine(x1: float, x2: float, mnu: float, E0: float, step: float = 5e-3):
    """
    Integrate the decay spectrum between limits on a fine scale

    Parameters
    ----------
    x1 : float
        Lower bin edge
    x2 : float
        Upper bin edge
    mnu : float
        Neutrino mass in eV
    E0 : float
        Endpoint energy in eV
    step : float
        Step size in eV

    Returns
    -------
    float
        The integral of the decay spectrum between limits x1 and x2
    """

    # Generate the fine-grained values between the limits
    nDivisions = np.ceil((x2 - x1) / step)
    EFine = np.linspace(x1, x2, nDivisions + 1)
    # Now use the trapezoidal rule to integrate along here
    return np.trapz(ApproxdGammadE(mnu, E0, EFine), EFine)


def CalcMuSignal(mNu: float, E0: float, binEdges: np.ndarray, tLive: float,
                 numDens: float, tVolume: float, subdivisionStep: float = 5e-3) -> np.ndarray:
    """
    Calculate the expected signal in each energy bin for a given neutrino mass 
    and endpoint energy.

    Parameters
    ----------
    mNu : float
        Neutrino mass in eV
    E0 : float
        Endpoint energy in eV
    binEdges : np.ndarray
        Edges of the energy bins in eV
    tLive : float
        Live time in seconds
    numDens : float
        Number density in particles per cubic meter
    tVolume : float
        Volume in cubic meters
    subdivisionStep : float
        Step size for integration (default is 5 meV)

    Returns
    -------
    np.ndarray
        Expected signal counts in each energy bin.
    """

    EMin, EMax = binEdges[0], binEdges[-1]
    eFine = np.arange(EMin, EMax + subdivisionStep, subdivisionStep)
    decayRatesFine = ApproxdGammadE(mNu, E0, eFine)

    bin_indices = np.digitize(eFine, binEdges) - 1
    # Handle edge cases
    valid_mask = (bin_indices >= 0) & (bin_indices < len(binEdges) - 1)
    bin_indices = bin_indices[valid_mask]
    decayRatesFine = decayRatesFine[valid_mask]
    eFine = eFine[valid_mask]

    # Sum contributions within each bin
    muSignal = np.bincount(bin_indices, weights=decayRatesFine * subdivisionStep,
                           minlength=len(binEdges) - 1)

    return muSignal * tLive * numDens * tVolume


def CalcMuBkg(bkgRate: float, binEdges: np.ndarray, tLive: float) -> np.ndarray:
    """
    Calculate the expected background in each energy bin.

    Parameters
    ----------
    bkgRate : float
        Background rate in counts per second per eV
    binEdges : np.ndarray
        Edges of the energy bins in eV
    tLive : float
        Live time in seconds

    Returns
    -------
    np.ndarray
        Expected background counts in each energy bin.
    """
    muBkg = bkgRate * (binEdges[1:] - binEdges[:-1]) * tLive

    return muBkg


def GenerateEvents(mNu: float, E0: float, binEdges: np.ndarray, tLive: float,
                   numDens: float, tVolume: float, b: float) -> np.ndarray:
    """
    Generate signal events based on the expected signal counts.

    Parameters
    ----------
    mNu : float
        Neutrino mass in eV
    E0 : float
        Endpoint energy in eV
    binEdges : np.ndarray
        Edges of the energy bins in eV
    tLive : float
        Live time in seconds
    numDens : float
        Number density in particles per cubic meter
    tVolume : float
        Volume in cubic meters
    b : float
        Background rate in counts per second per eV

    Returns
    -------
    np.ndarray
        Generated signal events.
    """
    muSignal = CalcMuSignal(mNu, E0, binEdges, tLive, numDens, tVolume)
    muBkg = CalcMuBkg(b, binEdges, tLive)
    rng = np.random.default_rng()
    # Generate Poisson-distributed events for signal and background
    events = rng.poisson(lam=muSignal + muBkg)
    return events


def DecayRate_pytensor(energies, mnu: pt.TensorVariable, E0: pt.TensorVariable):
    """
    A neutrino mass spectrum model suitable for fitting with PyMC.

    Parameters
    ----------
    energies : pytensor.tensor.TensorVariable
        Energies at which to evaluate the spectrum in eV.
    mnu : pytensor.tensor.TensorVariable
        Neutrino mass in eV.
    E0 : pytensor.tensor.TensorVariable
        Endpoint energy of the beta decay in eV.
    """
    # energies = pt.as_tensor_variable(energies)  # Ensure energies is a tensor variable
    ME_EV = const.m_e * const.c**2 / const.e
    beta = pm.math.sqrt(1.0 - (ME_EV / (energies + ME_EV))**2)
    p = pm.math.sqrt(energies**2 + 2.0 * energies * ME_EV)
    eta = const.alpha * 2 / beta
    fermiFunc = 2 * np.pi * eta / (1. - pm.math.exp(-2 * np.pi * eta))
    nuE = E0 - energies
    fermiConst = 1.1663787e-23  # eV^-2
    rate = fermiConst**2 * 0.97425**2 * fermiFunc * \
        (1 + 3 * (-1.2646)**2) * p * (energies + ME_EV) / (2 * np.pi**3)
    rate *= 1.0 / 6.58e-16  # Account for natural units

    stepCondition = (E0 - energies - mnu) >= 0.0
    spectrum = pm.math.switch(stepCondition,
                              rate * nuE * pm.math.sqrt(nuE**2 - mnu**2), 0.0)
    return spectrum


def trapz_pytensor(y, x):
    """
    Perform trapezoidal integration using PyTensor.

    Parameters
    ----------
    y : pytensor.tensor.TensorVariable
        The values to integrate.
    x : pytensor.tensor.TensorVariable
        The x-coordinates corresponding to the y-values.

    Returns
    -------
    pytensor.tensor.TensorVariable
        The result of the trapezoidal integration.
    """
    dx = pt.diff(x)  # Ensure the first element is included
    return pt.sum(0.5 * (y[:-1] + y[1:]) * dx)  # Trapezoidal rule


def IntegrateSpectrumFine_pytensor(x1, x2, mnu: float, E0: float, step=5e-3):
    """
    Integrate the decay spectrum between limits on a fine scale

    Parameters
    ----------
    x1  
        Lower bin edge
    x2  
        Upper bin edge
    mnu : float
        Neutrino mass in eV
    E0 : float
        Endpoint energy in eV
    step : float
        Step size in eV

    Returns
    -------
    float
        The integral of the decay spectrum between limits 
    """

    # Generate the fine-grained values between the limits
    nDivisions = pt.ceil((x2 - x1) / step)
    EFine = pt.linspace(x1, x2, nDivisions + 1)
    # Now use the trapezoidal rule to integrate along here
    return trapz_pytensor(DecayRate_pytensor(EFine, mnu, E0), EFine)


def CalcMuSignal_pytensor(mNu: pt.TensorVariable, E0: pt.TensorVariable,
                          binEdges: np.ndarray, tLive: float, numDens: float,
                          tVolume: float, subdivisionStep: float = 5e-3) -> pt.TensorVariable:
    """
    Calculate the expected signal in each energy bin for a given neutrino mass 
    and endpoint energy using PyTensor.

    Parameters
    ----------
    mNu : pytensor.tensor.TensorVariable
        Neutrino mass in eV
    E : pytensor.tensor.TensorVariable
        Endpoint energy in eV
    binEdges : np.ndarray
        Edges of the energy bins in eV 
    tLive : float
        Live time in seconds
    numDens : float
        Number density in particles per cubic meter
    tVolume : float
        Volume of the detector in cubic meters
    subdivisionStep : float
        Step size for the integration in eV

    Returns
    -------
    pytensor.tensor.TensorVariable
        The expected signal events in each energy bin
    """
    sigMuList = []
    # Ensure binEdges is a tensor variable
    binEdgesPt = pt.as_tensor_variable(binEdges)
    for i in range(len(binEdges) - 1):
        binCont = IntegrateSpectrumFine_pytensor(binEdgesPt[i], binEdgesPt[i+1],
                                                 mNu, E0, subdivisionStep)
        binCont *= tLive * numDens * tVolume
        sigMuList.append(binCont)

    sigMu = pt.stack(sigMuList)
    return sigMu


def GenerateBkgEvents(b: float, tLive: float, energyWindow: tuple[float, float]):
    """
    Generate background events from a flat background rate.

    Parameters
    ----------
    b : float
        Background rate in counts per second per eV
    tLive : float
        Live time of the detector in seconds
    energyWindow : tuple of float, optional
        Energy window for the events in keV (default is (17.7, 18.7))

    Returns
    -------
    np.ndarray
        Array of generated event energies in keV
    """
    nEvents = np.random.poisson(
        b * tLive * (energyWindow[1] - energyWindow[0]))
    # Uniformly distributed in the energy window
    return np.random.uniform(energyWindow[0], energyWindow[1], nEvents)


def CalculateR(numDens: float, volume: float):
    """
    Calculate the signal rate for a given number density and volume.

    Parameters
    ----------
    numDens : float
        Number density of tritium atoms in m^-3
    volume : float
        Effective volume of the detector in m^3

    Returns
    -------
    float
        Signal rate in counts per second
    """
    MEAN_LIFE_T = 12.33 * 365.25 * 24 * 3600 / np.log(2)  # in seconds
    return 2e-13 * numDens * volume / MEAN_LIFE_T


def OptimalDeltaE(r: float, b: float):
    """
    Calculate the optimal energy resolution for a given signal and background rate.

    Parameters
    ----------
    r : float
        Signal rate in the last eV
    b : float
        Background rate in counts per second

    Returns
    -------
    float
        Optimal energy resolution in eV
    """
    return np.sqrt(b / r)  # in eV


def Calculate90CL(r: float, b: float, tLive: float):
    """
    Calculate the frequentist 90% confidence level upper limit for the neutrino
    mass based on the signal and background rates.

    Parameters
    ----------
    r : float
        Signal rate in counts per second
    b : float
        Background rate in counts per second
    tLive : float
        Live time of the detector in seconds

    Returns
    -------
    float
        The 90% confidence level upper limit for the neutrino mass in eV
    """
    dE = OptimalDeltaE(r, b)
    return 2.0 * np.sqrt(r * tLive * dE + b * tLive / dE) / (3 * r * tLive)
