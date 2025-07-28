import numpy as np
import scipy.constants as const
import pymc as pm
import pytensor.tensor as pt

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