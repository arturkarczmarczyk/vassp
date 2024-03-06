import itertools
import typing

import numpy as np

from pyrepo_mcda.mcda_methods.mcda_method import MCDA_method


class VASSP(MCDA_method):
    def __init__(self, normalization_method: typing.Callable[[np.ndarray, np.ndarray], np.ndarray] = None):
        """Create the VASSP method object.

        Parameters
        -----------
            normalization_method : function
                VIKOR, the underlying method behind VASSP does not use normalization by default, thus
                `normalization_method` is set to None by default.
                However, you can choose method for normalization of decision matrix chosen `normalization_method` from `normalizations`.
                It is used in a way `normalization_method(X, types)` where `X` is a decision matrix
                and `types` is a vector with criteria types where 1 means profit and -1 means cost.
        """
        self.normalization_method = normalization_method

    def __call__(self, matrix: np.ndarray, weights: np.ndarray, types: np.ndarray, v: float = 0.5, s_coeffs : np.ndarray = np.array([0])) -> np.ndarray:
        """
        Score alternatives provided in decision matrix `matrix` using criteria `weights` and criteria `types`.

        Parameters
        -----------
            matrix : ndarray
                Decision matrix with m alternatives in rows and n criteria in columns.
            weights: ndarray
                Vector with criteria weights. Sum of weights must be equal to 1.
            types: ndarray
                Vector with criteria types. Profit criteria are represented by 1 and cost by -1.

            v : float
                parameter that is weight of strategy of the majority of criteria (the maximum group utility)

            s_coeffs: ndarray
                Vector including values of sustainability coefficient for each criterion. It takes values
                from 0 to 1.
                0 denotes unchanged VIKOR rankings with unlimited criteria compensation.
                1 represents a maximum reduction of criteria compensation.

        Returns
        --------
            ndrarray
                Vector with preference values of each alternative. The best alternative has the lowest preference value.

        Examples
        ---------
        >>> vassp = VASSP(normalization_method = minmax_normalization)
        >>> pref = vassp(matrix, weights, types, v, s_coeffs)
        >>> rank = rank_preferences(pref, reverse = False)
        """

        self._verify_input_data(matrix, weights, types)
        return self._vassp(
            matrix=matrix,
            weights=weights,
            types=types,
            normalization_method=self.normalization_method,
            v=v,
            s_coeffs=s_coeffs
        )

    def _equalization(self, matrix: np.ndarray, types: np.ndarray, s_coeffs: np.ndarray) -> np.ndarray:
        """Apply the SSP paradigm. Reduce compensation of criteria based on the s_coeffs vector (separate value for each criterion or a scalar value if same for all criteria).

        """

        # Calculate the mean deviation values of the performance values in matrix.
        mad = (matrix - np.mean(matrix, axis=0)) * s_coeffs

        # Set as 0, those mean deviation values that for profit criteria are lower than 0
        # and those mean deviation values that for cost criteria are higher than 0
        for j, i in itertools.product(range(matrix.shape[1]), range(matrix.shape[0])):
            # for profit criteria
            if types[j] == 1:
                if mad[i, j] < 0:
                    mad[i, j] = 0
            # for cost criteria
            elif types[j] == -1:
                if mad[i, j] > 0:
                    mad[i, j] = 0

        # Subtract from performance values in decision matrix standard deviation values multiplied by a
        # sustainability coefficient.
        return matrix - mad

    def _vassp(self, matrix, weights: np.ndarray, types: np.ndarray, normalization_method: typing.Callable[[np.ndarray, np.ndarray], np.ndarray], v: float, s_coeffs: np.ndarray) -> np.ndarray:
        # reducing compensation in decision matrix
        e_matrix = self._equalization(matrix, types, s_coeffs)

        # Without special normalization method
        if normalization_method == None:

            # Determine the best `fstar` and the worst `fmin` values of all criterion function
            maximums_matrix = np.amax(e_matrix, axis=0)
            minimums_matrix = np.amin(e_matrix, axis=0)

            fstar = np.zeros(e_matrix.shape[1])
            fmin = np.zeros(e_matrix.shape[1])

            # for profit criteria (`types` == 1) and for cost criteria (`types` == -1)
            fstar[types == 1] = maximums_matrix[types == 1]
            fstar[types == -1] = minimums_matrix[types == -1]
            fmin[types == 1] = minimums_matrix[types == 1]
            fmin[types == -1] = maximums_matrix[types == -1]

            # Calculate the weighted matrix
            weighted_matrix = weights * ((fstar - e_matrix) / (fstar - fmin))
        else:
            # With special normalization method
            norm_matrix = normalization_method(e_matrix, types)
            fstar = np.amax(norm_matrix, axis=0)
            fmin = np.amin(norm_matrix, axis=0)

            # Calculate the weighted matrix
            weighted_matrix = weights * ((fstar - norm_matrix) / (fstar - fmin))

        # Calculate the `S` and `R` values
        S = np.sum(weighted_matrix, axis=1)
        R = np.amax(weighted_matrix, axis=1)
        # Calculate the Q values
        Sstar = np.min(S)
        Smin = np.max(S)
        Rstar = np.min(R)
        Rmin = np.max(R)
        Q = v * (S - Sstar) / (Smin - Sstar) + (1 - v) * (R - Rstar) / (Rmin - Rstar)
        return Q
