"""
Below code calculates for a given species and t_range (200-1000) and (1000-tmax):
1. Partition functions from ExoMol linelist data
2. Heat capacity from obtained part. functions
3. Least squares polynomial coefficients for heat capacity

And saves all the above in /results folder as csv files.
"""

import sys 
import os.path
import numpy as np
from numpy.polynomial import polynomial as P
import pandas as pd
import json
from numba import njit

R = 8.31446261815324  # Ideal gas constant
C2 = 1.438777  # Second radiation constant hc/k


# Class InputError to raise error for incorrect input
class InputError(Exception):
    pass


def set_species_name():
    species = input("Enter species name:")
    if (
        os.path.exists(sys.path[0] + "/input_data/linelists/" + species + ".txt")
        == True
    ):
        return species
    else:
        raise InputError(
            "Incorrect species name or data for this species is not available. Try again."
        )


def set_t_range():
    t_range = input("Enter tempereture range ('1' for 200-1000K, '2' for 1000-tmax):")

    if t_range == "1" or t_range == "2":
        return t_range
    else:
        raise InputError("Temperature range can only be '1' or '2'.")


def get_linelist_data(species):

    linelist_data = sys.path[0] + "/input_data/linelists/" + species + ".txt"

    data = np.loadtxt(fname=linelist_data, usecols=(0, 1, 2))

    levels = data[:, [0]]
    levels = levels.astype(int)
    levels = levels.flatten()
    Energy = data[:, [1]]  # Energy
    Energy = Energy.flatten()
    degeneracy = data[:, [2]]  # Nuclear spin degeneracy factor
    degeneracy = degeneracy.flatten()

    return levels, Energy, degeneracy


def get_tmax(species):
    """
    Gets maximum temperature for a line list of a given species. This data was manually obtained from ExoMol def files.
    """
    with open("tmax_values.json", "r") as jsonFile:
        jsonObject = json.load(jsonFile)
        jsonFile.close()

        tmax = int(jsonObject[species])

    return tmax


def set_temperature_range(t_range, tmax):
    """
    This function sets a temperature range for a given calculation. For species for which line list temperature range is above 6000, the function sets the maximum temperature to 6000 K.
    """

    if t_range == "1":
        t_grid = np.arange(200, 1001)
        temperature = t_grid.reshape(801, 1)

    elif t_range == "2":
        if tmax <= 6000:
            t_grid = np.arange(1000, (tmax + 1))
            temperature = t_grid.reshape((tmax + 1 - 1000), 1)
        else:
            t_grid = np.arange(1000, 6001)
            temperature = t_grid.reshape(5001, 1)

    return temperature


@njit
def calculate_int_pf(linelists, temperature):
    """
    This function calculates internal parition function from line list data using this equation:

    """

    levels, Energy, degeneracy = linelists

    Q_total = np.zeros(len(temperature))
    Q_total = Q_total.reshape(len(temperature), 1)

    for t in range(len(temperature)):
        for i in levels:
            Q_total[t - 1] = Q_total[t - 1] + degeneracy[i - 1] * (
                np.exp(-(Energy[i - 1] * C2) / temperature[t - 1])
            )

    return Q_total


@njit
def calculate_int_pf_dif(linelists, temperature):

    levels, Energy, degeneracy = linelists

    Q_total_dif = np.zeros(len(temperature))
    Q_total_dif = Q_total_dif.reshape(len(temperature), 1)

    for t in range(len(temperature)):
        for i in levels:
            Q_total_dif[t - 1] = Q_total_dif[t - 1] + degeneracy[i - 1] * (
                np.exp(-(Energy[i - 1] * C2) / temperature[t - 1])
            ) * ((Energy[i - 1] * C2) / temperature[t - 1])

    return Q_total_dif


@njit
def calculate_int_pf_twice_dif(linelists, temperature):

    levels, Energy, degeneracy = linelists

    Q_total_twice_dif = np.zeros(len(temperature))
    Q_total_twice_dif = Q_total_twice_dif.reshape(len(temperature), 1)

    for t in range(len(temperature)):
        for i in levels:
            Q_total_twice_dif[t - 1] = Q_total_twice_dif[t - 1] + degeneracy[i - 1] * (
                np.exp(-(Energy[i - 1] * C2) / temperature[t - 1])
            ) * ((Energy[i - 1] * C2) / temperature[t - 1]) * (
                (Energy[i - 1] * C2) / temperature[t - 1]
            )

    return Q_total_twice_dif


def combine_pf_data(temperature, Q_total, Q_total_dif, Q_total_twice_dif):
    pf_data = np.concatenate(
        (temperature, Q_total, Q_total_dif, Q_total_twice_dif), axis=-1
    )
    return pf_data


def save_pf_data(t_range, species, pf_data):

    df = pd.DataFrame(pf_data)
    df.columns = ["temperature", "Q", "Qdif", "Qtwice_dif"]
    df = df.set_index("temperature")
    df.to_csv(
        sys.path[0] + "/results/partition_functions/" + t_range + "/" + species + ".csv"
    )
    print("Partition functions are calcualated and available in the results folder.")


@njit
def calculate_heat_capacity(temperature, Q_total, Q_total_dif, Q_total_twice_dif):
    """
    This function calculates heat capacity from the partition function via this equation:
    Cp = [Q"/Q - (Q'/Q)^2]R + (5/2)*R
    """

    heat_capacity_array = np.zeros(len(temperature))
    heat_capacity_array = heat_capacity_array.reshape(len(temperature), 1)

    for t in range(len(temperature)):
        heat_capacity_array[t - 1] = (
            (
                (Q_total_twice_dif[t - 1] / Q_total[t - 1])
                - np.square(Q_total_dif[t - 1] / Q_total[t - 1])
            )
            * R
        ) + ((5 / 2) * R)

    return heat_capacity_array


def combine_heat_capacity_data(temperature, heat_capacity_array):
    heat_capacity = np.concatenate((temperature, heat_capacity_array), axis=-1)
    return heat_capacity


def save_heat_capacity_data(t_range, species, heat_capacity):

    data = np.squeeze(heat_capacity)
    df = pd.DataFrame(data)
    df.columns = ["temperature", "heat_capacity"]
    df = df.set_index("temperature")
    df.to_csv(
        df.to_csv(
            sys.path[0]
            + "/results/heat_capacity/from_pf/"
            + t_range
            + "/"
            + species
            + ".csv"
        )
    )
    print("Heat capacity data is calculated and available in the results folder.")


def polynomial_coefficients(temperature, heat_capacity_array):

    y = np.zeros(len(temperature))

    for t in range(len(temperature)):
        y[t - 1] = (heat_capacity_array[t - 1] * np.square(temperature[t - 1])) / R

    x = np.squeeze(temperature)
    p = P.polyfit(x, y, 6)

    coefficients = []

    for i in range(7):
        coefficients.append(p[i])

    return coefficients


def save_polynomial_coefficients_data(coefficients, t_range, species):

    # coefficients = np.squeeze(coefficients)
    df = pd.DataFrame(np.transpose(coefficients))
    df.columns = ["polynomial coefficients"]
    df = df.set_index("polynomial coefficients")
    df.to_csv(
        (
            sys.path[0]
            + "/results/polynomial_coefficients/"
            + t_range
            + "/"
            + species
            + ".csv"
        )
    )
    print("Polynomial coefficients are generated and available in the resutls folder.")


def main():
    species = set_species_name()
    t_range = set_t_range()
    linelists = get_linelist_data(species)
    tmax = get_tmax(species)
    temperature = set_temperature_range(t_range, tmax)
    Q_total = calculate_int_pf(linelists, temperature)
    Q_total_dif = calculate_int_pf_dif(linelists, temperature)
    Q_total_twice_dif = calculate_int_pf_twice_dif(linelists, temperature)
    pf_data = combine_pf_data(temperature, Q_total, Q_total_dif, Q_total_twice_dif)
    save_pf_data(t_range, species, pf_data)
    heat_capacity_array = calculate_heat_capacity(
        temperature, Q_total, Q_total_dif, Q_total_twice_dif
    )
    heat_capacity = combine_heat_capacity_data(temperature, heat_capacity_array)
    save_heat_capacity_data(t_range, species, heat_capacity)
    coefficients = polynomial_coefficients(temperature, heat_capacity_array)
    save_polynomial_coefficients_data(coefficients, t_range, species)


if __name__ == "__main__":
    main()
