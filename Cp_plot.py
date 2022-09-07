import sys
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from polynomial_coefficients import set_species_name

R = 8.31446261815324  # Ideal gas constant

"""
The below code calculates for the entire temperature range:
1. Heat capacity from newly obtained coefficients (200 - tmax)
2. Heat capacity from 2002 NASA coefficients (200 - 6000 K)
And generates a plot to comapare #1, #2, JANAF data, and heat capacity previously 
generated from partition functions.
"""


def get_NASA_coefficients(species, t_range="1"):

    NASA_data = (
        sys.path[0] + "/data/NASA_2002/NASA_" + species + "_" + t_range + ".csv"
    )
    NASA_coefficients = np.loadtxt(fname=NASA_data)

    return NASA_coefficients


def set_temperature_range(t_range="1"):

    if t_range == "1":
        t_grid = np.arange(200, 1001)
        temperature = t_grid.reshape(801, 1)

    elif t_range == "2":
        t_grid = np.arange(1000, 6001)
        temperature = t_grid.reshape(5001, 1)

    return temperature


def Cp_from_coefficients(tgrid1, tgrid2, coefficients1, coefficients2):
    """This function calculates heat capacity from coefficients based on this equation:
    Cp(T)/R = a1 * T^-2 + a2 * T^-1 + a3 + a4 * T + a5 * T^2 + a6 * T^3 + a7 * T^4
    """

    Cp_array1 = np.zeros(len(tgrid1))
    Cp_array2 = np.zeros(len(tgrid2))

    Cp_array1 = Cp_array1.reshape(len(tgrid1), 1)
    Cp_array2 = Cp_array2.reshape(len(tgrid2), 1)

    for t in range(len(tgrid1)):
        Cp_array1[t - 1] = (
            coefficients1[0] * np.float_power(tgrid1[t - 1], -2)
            + coefficients1[1] * np.float_power(tgrid1[t - 1], -1)
            + coefficients1[2]
            + coefficients1[3] * tgrid1[t - 1]
            + coefficients1[4] * tgrid1[t - 1] ** (2)
            + coefficients1[5] * tgrid1[t - 1] ** (3)
            + coefficients1[6] * tgrid1[t - 1] ** (4)
        ) * R

    for t in range(len(tgrid2)):
        Cp_array2[t - 1] = (
            coefficients2[0] * np.float_power(tgrid2[t - 1], -2)
            + coefficients2[1] * np.float_power(tgrid2[t - 1], -1)
            + coefficients2[2]
            + coefficients2[3] * tgrid2[t - 1]
            + coefficients2[4] * tgrid2[t - 1] ** (2)
            + coefficients2[5] * tgrid2[t - 1] ** (3)
            + coefficients2[6] * tgrid2[t - 1] ** (4)
        ) * R

    return Cp_array1, Cp_array2


def combine_coefficients_data(tgrid1, tgrid2, Cp_array1, Cp_array2):

    t_range = np.vstack((tgrid1, tgrid2))
    Cp_array = np.vstack((Cp_array1, Cp_array2))

    Cp = np.stack((t_range, Cp_array))

    return Cp


def save_Cp_from_coefficients_data(Cp, data_source, species):

    data = np.squeeze(np.transpose(Cp))
    df = pd.DataFrame(data)
    df.columns = ["temperature", "heat_capacity"]
    df = df.set_index("temperature")
    df.to_csv(
        sys.path[0]
        + "/results/heat_capacity/from_"
        + data_source
        + "_coefficients/"
        + species
        + ".csv"
    )


def get_new_coefficients(species, t_range="1"):

    new_data = (
        sys.path[0]
        + "/results/polynomial_coefficients/"
        + t_range
        + "/"
        + species
        + ".csv"
    )
    new_coefficients = np.loadtxt(fname=new_data, skiprows=1)

    return new_coefficients


def get_tgrid2_max(species):

    with open(sys.path[0] + "/data/linelists/tmax_values.json", "r") as jsonFile:
        jsonObject = json.load(jsonFile)
        jsonFile.close()

        tmax = int(jsonObject[species])
        if tmax <= 6000:
            t_grid = np.arange(1000, (tmax + 1))
            tgrid2 = t_grid.reshape((tmax + 1 - 1000), 1)
        else:
            t_grid = np.arange(1000, 6001)
            tgrid2 = t_grid.reshape(5001, 1)

    return tgrid2


def get_JANAF_data(species):

    JANAF = sys.path[0] + "/data/JANAF" + "/" + species + "_JANAF.txt"
    JANAF_data = np.loadtxt(fname=JANAF, skiprows=2, usecols=(0, 1), unpack=True)
    return JANAF_data


def get_pf_heat_capacity_data(species):

    t_range_1 = pd.read_csv(
        sys.path[0] + "/results/heat_capacity/from_pf/1/" + species + ".csv"
    )
    t_range_1 = t_range_1.to_numpy()
    t_1, Cp_1 = np.hsplit(t_range_1, 2)

    t_range_2 = pd.read_csv(
        sys.path[0] + "/results/heat_capacity/from_pf/2/" + species + ".csv"
    )
    t_range_2 = t_range_2.to_numpy()
    t_2, Cp_2 = np.hsplit(t_range_2, 2)

    t_range = np.vstack((t_1, t_2))
    Cp_array = np.vstack((Cp_1, Cp_2))

    Cp_pf = np.concatenate((t_range, Cp_array), axis=-1)

    return Cp_pf


def plot(species, Cp_pf, Cp_NASA, Cp_coeff, Cp_JANAF):
    """
    This function plots heat capacity as obtained from partition functions, 
    heat capacity data from JANAF,
    heat capacity as calculated from NASA and new coefficients,
    and provides a residuals subplot for heat capacity from partition functions vs 
    its coefficients.
    """

    t_, Cp_ = np.hsplit((Cp_pf), 2)
    t_ = np.squeeze(t_)
    Cp_ = np.squeeze(Cp_)

    t_JANAF, Cp_JANAF = np.vsplit((Cp_JANAF), 2)
    t_JANAF = np.squeeze(t_JANAF)
    Cp_JANAF = np.squeeze(Cp_JANAF)

    t_coef, Cp_coef = np.vsplit((Cp_coeff), 2)
    t_coef = np.squeeze(t_coef)
    Cp_coef = np.squeeze(Cp_coef)

    t_N, Cp_N = np.vsplit((Cp_NASA), 2)
    t_N = np.squeeze(t_N)
    Cp_N = np.squeeze(Cp_N)

    residuals = np.subtract(Cp_, Cp_coef)

    fig, ax = plt.subplots(figsize=(8, 6))

    plt.title("Specific Heat for " + species)
    plt.xlabel("T(K)")
    plt.ylabel("Cp")

    ax.plot(t_JANAF, Cp_JANAF, linestyle="-", color="red", label="JANAF")
    ax.plot(t_N, Cp_N, linestyle="--", color="orange", label="NASA Polynomials")
    ax.plot(t_, Cp_, linestyle="-", color="blue", label="This Work")
    ax.plot(
        t_coef,
        Cp_coef,
        linestyle="-",
        color="green",
        label="This Work from Coefficients",
    )

    axins = ax.inset_axes([0.65, 0.075, 0.3, 0.3])
    axins.plot(Cp_, residuals)

    ax.legend()
    plt.savefig(sys.path[0] + "/results/plots/" + species + ".png")
    plt.show()


def main():
    species = set_species_name()
    # Gather data for NASA
    NASA_coefficients_1 = get_NASA_coefficients(species, "1")
    NASA_coefficients_2 = get_NASA_coefficients(species, "2")
    tgrid1 = set_temperature_range("1")
    tgrid2 = set_temperature_range("2")
    # Calculate for NASA
    Cp_NASA_array1, Cp_NASA_array2 = Cp_from_coefficients(
        tgrid1, tgrid2, NASA_coefficients_1, NASA_coefficients_2
    )
    Cp_NASA = combine_coefficients_data(tgrid1, tgrid2, Cp_NASA_array1, Cp_NASA_array2)
    save_Cp_from_coefficients_data(Cp_NASA, "NASA", species)
    # Gather data for new coefficients
    new_coefficients1 = get_new_coefficients(species, "1")
    new_coefficients2 = get_new_coefficients(species, "2")
    tgrid2_max = get_tgrid2_max(species)
    # Calculate for new coefficients
    Cp_array1, Cp_array2 = Cp_from_coefficients(
        tgrid1, tgrid2_max, new_coefficients1, new_coefficients2
    )
    Cp_coeff = combine_coefficients_data(tgrid1, tgrid2_max, Cp_array1, Cp_array2)
    save_Cp_from_coefficients_data(Cp_coeff, "new", species)
    # Gather existing JANAF and new heat capacity data
    Cp_JANAF = get_JANAF_data(species)
    Cp_pf = get_pf_heat_capacity_data(species)
    # Plot
    plot(species, Cp_pf, Cp_NASA, Cp_coeff, Cp_JANAF)


if __name__ == "__main__":
    main()
