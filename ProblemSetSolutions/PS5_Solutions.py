# Import the packages we would use
import numpy as np
import pandas as pd
import scipy.optimize as opt
from scipy.stats import norm
from geopy.distance import distance
import os
import time


""""
Define function to use for estimation
"""


def create_array_ids(x):
    """
    This function creates arrays of buyer and target ids to represent
    all actual and counter factural combinations

    The four arrays are:
    (b, t), (b', t'), (b, t'), (b', t)

    Args:
        x (Pandas DataFrame): dataframe with columns identifying buyer
            and target ids for observed mergers and market year named
            buyer_id, target_id, year

    Returns:
        bt_arrays (dict): dictionary with Numpy arrays of buyer
            and target ids
    """
    # Set number of rows to put in arrays - will delete rows that aren't
    # filled in later
    N = 10000
    # Find list of unique years
    set_of_years = set(x.year)
    # initialize numpy arrays - columns will represent year, buyer_id,
    # target_id
    bt, bptp, btp, bpt = (
        np.zeros((N, 3)),
        np.zeros((N, 3)),
        np.zeros((N, 3)),
        np.zeros((N, 3)),
    )
    # set counter to identify row to put loop items into
    counter = 0
    for y in set_of_years:
        # get targets of buyer b
        x = x.sort_index()
        df = x[x.year == y]
        buyers_in_year = list(set(df.buyer_id.values))
        for b in buyers_in_year[:-1]:  # all but last buyer
            df = x[(x.buyer_id == b) & (x.year == y)]
            # get targets of buyer i
            b_target = set(df.target_id.values)
            for t in b_target:
                # get buyers b' who are not buyer b
                for bp in buyers_in_year[buyers_in_year.index(b) + 1:]:
                    df = df = x[(x.buyer_id == bp) & (x.year == y)]
                    # get targets of buyer b'
                    bp_target = set(df.target_id.values)
                    for tp in bp_target:
                        bt[counter, :] = [y, b, t]
                        bptp[counter, :] = [y, bp, tp]
                        btp[counter, :] = [y, b, tp]
                        bpt[counter, :] = [y, bp, t]
                        counter += 1

    # delete rows not filled in
    bt = bt[:counter, :]
    bptp = bptp[:counter, :]
    btp = btp[:counter, :]
    bpt = bpt[:counter, :]

    bt_arrays = {"bt": bt, "bptp": bptp, "btp": btp, "bpt": bpt}

    return bt_arrays


def create_x(merger_df, id_array):
    """
    Creates arrays with covariates for buyer-target combinations

    Args:
        merger_df (Pandas DataFrame):  data with matches and characteristics
        id_array (Numpy array): array with buyer and target ids

    Returns:
        df (Pandas DataFrame): dataframe of specified buyer and target
            pairs with characteristics and calculated variables
    """
    # separate out buyer and target characteristics
    buyer_chars = merger_df[
        [
            "buyer_id",
            "year",
            "buyer_lat",
            "buyer_long",
            "num_stations_buyer",
            "corp_owner_buyer",
        ]
    ].copy()
    # in case buyers are repeated (i.e., matches are one to many),
    # drop any duplicates
    buyer_chars.drop_duplicates(
        subset=["buyer_id", "year"], keep="first", inplace=True)
    target_chars = merger_df[
        [
            "year",
            "target_id",
            "target_lat",
            "target_long",
            "hhi_target",
            "population_target",
            "price",
        ]
    ].copy()
    # there should not be repeated targets, but drop duplicates to be safe
    target_chars.drop_duplicates(
        subset=["target_id", "year"], keep="first", inplace=True
    )

    # put buyer and target ids in a dataframe to enable merge with X's
    df = pd.DataFrame(
        id_array,
        index=range(id_array.shape[0]),
        columns=["year", "buyer_id", "target_id"],
    )

    # Merge buyer and target characteristics onto dataframes with buyer
    # and target ids
    df = df.merge(
        buyer_chars,
        how="left",
        left_on=["year", "buyer_id"],
        right_on=["year", "buyer_id"],
        copy=True,
    )
    df = df.merge(
        target_chars,
        how="left",
        left_on=["year", "target_id"],
        right_on=["year", "target_id"],
        copy=True,
    )

    # create additional variables for the X matrix in response to the
    # question
    df["pop"] = df["population_target"] / 1000000
    df["price"] = df["price"] / 1000000
    df["stations_pop"] = df["num_stations_buyer"] * df["pop"]
    df["corp_owner_pop"] = df["corp_owner_buyer"] * df["pop"]
    df["buyer_loc"] = df[["buyer_lat", "buyer_long"]].apply(tuple, axis=1)
    df["target_loc"] = df[["target_lat", "target_long"]].apply(tuple, axis=1)
    df["distance"] = df.apply(
        lambda row: distance(row["buyer_loc"], row["target_loc"]).miles, axis=1
    )

    return df


# create the payoff function
def payoff(parameters, df, covariates):
    """
    Compute the payoff matrices for a buyer-target match, f(b,t)

    Args:
        parameters (Numpy array): values of parameters in model
        df (Pandas DataFrame): data of buyer, target, and match characteristics
        covariates (list): list of strings with names of covariates to
            use in the payoff function

    Returns:
        f (Numpy array): vectors with payoffs to the mergers in df

    """

    f = (parameters * df[covariates]).sum(axis=1)

    return f


def Qscore(coeffs, data_dict, covariates, use_price, smoothed_estimator):
    """
    Statistical objective function for the maximum score estimator.

    Args:
        parameters (Numpy array): guesses for coefficients on covariates
        data_dict (dictionary): keys represent buyer-target pairs
            (e.g, (b,t), (b',t')), values are
            dataframes with buyer, target, match characteristics
        covariates (list): list of strings with names of covariates to
            use in the payoff function
        use_price (boolean): indicator for use estimator with prices
        smoothed_estimator (boolean): indicator for use smoothed MSE

    Returns:
        f (scalar): the fraction of inequalities that are satisfied
    """
    if smoothed_estimator:
        if use_price:
            coeffs = np.append(coeffs, -1.0)  # price in last column
            value = (
                payoff(coeffs, data_dict["bt"], covariates)
                - payoff(coeffs, data_dict["btp"], covariates)
            ) + (
                payoff(coeffs, data_dict["bptp"], covariates)
                - payoff(coeffs, data_dict["bpt"], covariates)
            )
            ineq = norm.cdf(value.astype(float), scale=1 / 30)
        else:
            # 1st covariate has coeff of one
            coeffs = np.insert(coeffs, 0, 1.0)
            value = (
                payoff(coeffs, data_dict["bt"], covariates)
                + payoff(coeffs, data_dict["bptp"], covariates)
            ) - (
                payoff(coeffs, data_dict["btp"], covariates)
                + payoff(coeffs, data_dict["bpt"], covariates)
            )
            ineq = norm.cdf(value.astype(float), scale=1 / 30)

    else:
        if use_price:
            coeffs = np.append(coeffs, -1.0)  # price in last column
            ineq = (
                payoff(coeffs, data_dict["bt"], covariates)
                >= payoff(coeffs, data_dict["btp"], covariates)
            ) & (
                payoff(coeffs, data_dict["bptp"], covariates)
                >= payoff(coeffs, data_dict["bpt"], covariates)
            )
        else:
            # 1st covariate has coeff of one
            coeffs = np.insert(coeffs, 0, 1.0)
            ineq = (
                payoff(coeffs, data_dict["bt"], covariates)
                + payoff(coeffs, data_dict["bptp"], covariates)
            ) >= (
                payoff(coeffs, data_dict["btp"], covariates)
                + payoff(coeffs, data_dict["bpt"], covariates)
            )

    # sum over all years - number satisfied and total number
    Q = sum(ineq)
    H = len(ineq)
    # return standardized score (fraction of inequalities satisfied)
    f = -Q / H

    return f


def estimate_mse(
    init_params,
    covariates,
    data_dict,
    use_price=True,
    smoothed_estimator=False,
    method="NM",
    print_results=False,
):
    """
    This function calls the optimizer to estimate the Maximum Score Estimator.

    Args:
        init_params parameters (Numpy array): guesses for coefficients
            on covariates
        data_dict (dictionary): keys represent buyer-target pairs
            (e.g, (b,t), (b',t')), values are
            dataframes with buyer, target, match characteristics
        covariates (list): list of strings with names of covariates to
            use in the payoff function
        use_price (boolean): indicator for use estimator with prices
        smoothed_estimator (boolean): indicator for use smoothed MSE
        method (string): minimization method to use (NM = Nelder-Mead,
                DE = differential evolution, SA = simulated annealing)
        print_results (boolean): whether results of the estimation printed

    Returns:
        results (Scipy optimize results object): results from optimization
    """
    start_time = time.time()

    # Nelder-Mead method
    if method == "NM":
        results = opt.minimize(
            Qscore,
            init_params,
            method="Nelder-Mead",
            args=(data_dict, covariates, use_price, smoothed_estimator),
            tol=1e-13,
        )
    # Differential evolution method
    elif method == "DE":
        bnds = [(-20000, 20000)] * (len(covariates) - 1)
        results = opt.differential_evolution(
            Qscore,
            bnds,
            args=(data_dict, covariates, use_price, smoothed_estimator),
            strategy="best1bin",
            maxiter=1000,
            popsize=15,
            tol=0.01,
            mutation=(0.5, 1.0),
            recombination=0.7,
            seed=None,
            callback=None,
            disp=False,
            polish=True,
            init="random",
            atol=0,
        )
    # Simulated annealing method
    elif method == "SA":
        results = opt.basinhopping(
            Qscore,
            init_params,
            niter=1000,
            T=1.0,
            stepsize=0.05,
            minimizer_kwargs={
                "args": (data_dict, covariates, use_price, smoothed_estimator)
            },
            interval=50,
        )
    else:
        print(
            "Please enter a valid optimization method - or nothing"
            + " to use the default (Nelder-Mead)."
        )
        results = None
    end_time = time.time()
    if use_price:
        results["x"] = np.append(results["x"], -1.0)
    else:
        results["x"] = np.insert(results["x"], 0, 1.0)
    if print_results:
        print("beta_hat: ", results["x"])
        print("Estimation took ", end_time - start_time, " seconds to complete")

    return results


def merger_estimate(
    init_params,
    covariates,
    data,
    use_price=False,
    smoothed_estimator=False,
    method="DE",
):
    """
    Interface function to estimate model of radio mergers.

    Args:
        init_params parameters (Numpy array): guesses for coefficients
            on covariates
        data (Pandas DataFrame): raw data with mergers
        covariates (list): list of strings with names of covariates to
            use in the payoff function
        use_price (boolean): indicator for use estimator with prices
        smoothed_estimator (boolean): indicator for use smoothed MSE
        method (string): minimization method to use (NM = Nelder-Mead,
                DE = differential evolution, SA = simulated annealing)
        print_results (boolean): whether results of the estimation printed

    Returns:
        results (Scipy optimize results object): results from optimization
    """
    bt_arrays = create_array_ids(data)
    data_dict = {}
    for k, v in bt_arrays.items():
        data_dict[k] = create_x(data, v)
    results = estimate_mse(
        init_params, covariates, data_dict, use_price, smoothed_estimator, method
    )

    return results


## Do the stuff

# Read in data
filepath = os.path.join("..", "Matching", "radio_merger_data.csv")
radio_data = pd.read_csv(filepath)

# initialize list for results
mse_results = []

# Model 1
covariate_list1 = ["stations_pop", "corp_owner_pop", "distance"]
# Note, only specify two since first coefficient will be normalized to 1
init_guesses = np.array([7.54, 2.28])
results = merger_estimate(
    init_guesses,
    covariate_list1,
    radio_data,
    use_price=False,
    smoothed_estimator=False,
    method="SA",
)
mse_results.append(results)
# Model 2
covariate_list2 = ["stations_pop", "corp_owner_pop", "distance", "hhi_target", "price"]
# Note, only specify 4 since last coeff on price normalized to -1
init_guesses = np.array(
    [-0.002, 7.54, 2.28, -0.18]
)
results = merger_estimate(
    init_guesses,
    covariate_list2,
    radio_data,
    use_price=True,
    smoothed_estimator=False,
    method="SA",
)
mse_results.append(results)

# Put results in a dict, then dataframe, so can print in markdown easily
out_dict = {"Variable": covariate_list2 + ["Score"], "Model 1": [], "Model 2": []}
for i, v in enumerate(mse_results):
    out_dict["Model " + str(i + 1)].extend(v["x"])
    if len(out_dict["Model " + str(i + 1)]) < len(covariate_list2):
        out_dict["Model " + str(i + 1)].extend(
            ["-"] * (len(covariate_list2) - len(out_dict["Model " + str(i + 1)]))
        )
    out_dict["Model " + str(i + 1)].append(v["fun"] * -1)
df_out = pd.DataFrame(out_dict)
print(df_out.to_markdown(index=False))
