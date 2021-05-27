
"""
Experimental methods for parameter tuning of an estimator from "sklearn"
==============================================
Created by Ng Yiu Wai, Jan 2021

"""
import numpy as np
import copy
from sklearn.model_selection import cross_val_score


class PSOSearchCV_SVM:
    """
    For parameter tuning using PSO for an estimator. It is initially designed for "sklearn.svm.SVC".

    Parameters
    ----------
    estimator: estimator
        Assumed to be the classifier from "sklearn.svm.SVC"
    param_grid: dict
        Assumed to be SVC's parameter, i.e. {'C': float, 'gamma': float}
    scoring: str
        Assumed to be SVC's parameter, i.e. by default = 'roc_auc'
    cv: int
        Number of folds for "sklearn.model_selection.cross_val_score"
        By default, it is Stratified K-Fold CV.
    n_jobs: int
        Number of jobs to run in parallel for "sklearn.model_selection.cross_val_score"
    pso_size: int
        Number of initial particles for Particle Swarm Optimization (PSO)
    pso_max_iter: int
        Maximum number of iterations for Particle Swarm Optimization (PSO)
    velocity_scale:   float
        TO UPDATE

    Returns
    ----------
    None

    """

    def __init__(self, estimator, param_grid, scoring='roc_auc', cv=5, n_jobs=1, pso_size=5, pso_max_iter=100, velocity_scale=1, verbose=False):
        self.estimator = estimator
        self.scoring = scoring
        self.param_grid = param_grid
        self.cv = cv
        self.n_jobs = n_jobs
        self.pso_size = pso_size
        self.pso_max_iter = pso_max_iter
        self.velocity_scale = velocity_scale
        self.verbose = verbose

    def fit(self, X, y):
        """
        Run fit with parameters within range.

        Parameters
        ----------
        X:  array-like, shape (n_samples, n_features)
            Training vector. Assume that it is ndarray of "numpy".

        y:  array-like, shape (n_samples,)
            Target relative to X for classification or regression.
            Assume that it is ndarray of "numpy".

        Returns
        ----------
        None

        """

        ############################################################################
        # Initialize the particles
        ############################################################################

        # Generate initial coordination of particles
        C_range = np.logspace(
            np.log10(self.param_grid['C'][0]),  np.log10(self.param_grid['C'][1]),
            self.pso_size)
        gamma_range = np.logspace(
            np.log10(self.param_grid['gamma'][0]),  np.log10(self.param_grid['gamma'][1]),
            self.pso_size)
        np.random.shuffle(C_range)
        np.random.shuffle(gamma_range)

        # Create particles by assigning coordination
        p_all = []
        p_curr = {'C': [], 'gamma': [], 'score': [], 'velocity': []}
        for i in range(0, self.pso_size):
            p_curr['C'].append(C_range[i])
            p_curr['gamma'].append(gamma_range[i])
            p_curr['score'].append(0)
            p_curr['velocity'].append([0, 0])   # 1st one is velocity of C. 2nd one is velocity of gamma.

        # Create particles by assigning scores
        for i in range(0, self.pso_size):
            self.estimator.C = p_curr['C'][i]
            self.estimator.gamma = p_curr['gamma'][i]
            p_curr['score'][i] = np.mean(cross_val_score(
                self.estimator, X, y, scoring=self.scoring, cv=self.cv, n_jobs=self.n_jobs))

        ############################################################################
        # Initialize the local best and global best
        ############################################################################

        # Update local best within the path of each particles
        p_lb = {'C': [], 'gamma': [], 'score': []}
        for i in range(0, self.pso_size):
            p_lb['C'].append(p_curr['C'][i])
            p_lb['gamma'].append(p_curr['gamma'][i])
            p_lb['score'].append(p_curr['score'][i])

        # Update local best within current round of iteration
        ind = np.array(p_curr['score']).argmax()
        p_ib = {'C': p_curr['C'][ind], 'gamma': p_curr['gamma'][ind], 'score': p_curr['score'][ind]}

        # Update global best, if local best of this round is better than global best.
        p_gb = p_ib

        p_all.append(copy.deepcopy(p_curr))

        ############################################################################
        # Run PSO - Compute the scores, update the coordinations,
        #           and loop until max_iter is reached.
        ############################################################################

        # Iterative process until max_iter is reached.
        for iter_count in range(0, self.pso_max_iter):

            ########################################################################
            # Update coordination for each particles
            ########################################################################

            for i in range(0, self.pso_size):
                # =============================================================
                # Find new velocity, which is determined by
                # (a) inertia
                # (b) velocity to local best
                # (c) velocity to global best
                # =============================================================
                inertia = p_curr['velocity'][i]
                velocity_to_lb = [p_lb['C'][i] - p_curr['C'][i], p_lb['gamma'][i] - p_curr['gamma'][i]]
                velocity_to_gb = [p_gb['C'] - p_curr['C'][i], p_gb['gamma'] - p_curr['gamma'][i]]

                # =============================================================
                # Find new position, which new_pos = old_pos + new_velocity
                # Note the new_velocity is a probabilitic function, decided by
                #   (a) inertia, (b) v to local best, (c) v to global best
                # I define it as
                #   new_velocity = [0.5*(a) + r1*(b) + r2*(c)
                #                   where r1, r2 ~ uniform(0,1)]
                # =============================================================
                r = np.random.uniform(0, 1, size=2)
                new_velocity = [0, 0]
                new_velocity[0] = self.velocity_scale * (0.5 * inertia[0]
                                                         + r[0] * velocity_to_lb[0]
                                                         + r[1] * velocity_to_gb[0])
                new_velocity[1] = self.velocity_scale * (0.5 * inertia[1]
                                                         + r[0] * velocity_to_lb[1]
                                                         + r[1] * velocity_to_gb[1])
                p_curr['C'][i] = p_curr['C'][i] + new_velocity[0]
                p_curr['gamma'][i] = p_curr['gamma'][i] + new_velocity[1]
                p_curr['velocity'][i] = new_velocity

            for i in range(0, self.pso_size):
                # =============================================================
                # Check if any of new coordinates are out of boundary
                # =============================================================
                p_curr['C'][i] = max(p_curr['C'][i], self.param_grid['C'][0])
                p_curr['C'][i] = min(p_curr['C'][i], self.param_grid['C'][1])
                p_curr['gamma'][i] = max(p_curr['gamma'][i], self.param_grid['gamma'][0])
                p_curr['gamma'][i] = min(p_curr['gamma'][i], self.param_grid['gamma'][1])

            ########################################################################
            # Update score for each particles
            ########################################################################
            for i in range(0, self.pso_size):
                self.estimator.C = p_curr['C'][i]
                self.estimator.gamma = p_curr['gamma'][i]
                p_curr['score'][i] = np.mean(cross_val_score(
                    self.estimator, X, y, scoring=self.scoring, cv=self.cv, n_jobs=self.n_jobs))

            ########################################################################
            # Update local best and global best
            ########################################################################

            # Update local best within the path of each particles
            for i in range(0, self.pso_size):
                if (p_curr['score'][i] > p_lb['score'][i]):
                    p_lb['C'][i] = p_curr['C'][i]
                    p_lb['gamma'][i] = p_curr['gamma'][i]
                    p_lb['score'][i] = p_curr['score'][i]

            # Update local best within current round of iteration
            ind = np.array(p_curr['score']).argmax()
            if (p_curr['score'][ind] > p_ib['score']):
                p_ib['C'] = p_curr['C'][ind]
                p_ib['gamma'] = p_curr['gamma'][ind]
                p_ib['score'] = p_curr['score'][ind]

            # Update global best, if local best of this round is better than global best.
            if (p_ib['score'] > p_gb['score']):
                p_gb = p_ib

            ########################################################################
            # Save information of current particles for further analysis
            ########################################################################

            if self.verbose:
                print('.................. Iteration #', iter_count + 1, '..................')
                print('> Curr iter best score = ', p_ib['score'],
                      'at C =', p_ib['C'], ', gamma = ', p_ib['gamma'])
                print('> Global    best score = ', p_gb['score'],
                      'at C =', p_gb['C'], ', gamma = ', p_gb['gamma'])
                print('C :')
                print(np.round(np.array(p_curr['C']), 4))
                print('gamma :')
                print(np.round(np.array(p_curr['gamma']), 4))

            p_all.append(copy.deepcopy(p_curr))

        self.result = p_all
        self.best_param = p_gb

        return
