from abc import ABC, abstractmethod
from loguru import logger
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


class Bandit(ABC):
    """Abstract base class for all Bandit algorithms."""
    ##==== DO NOT REMOVE ANYTHING FROM THIS CLASS ====##
    @abstractmethod
    def __init__(self, p):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def pull(self):
        """ """
        pass

    @abstractmethod
    def update(self):
        """ """
        pass

    @abstractmethod
    def experiment(self):
        """ """
        pass

    @abstractmethod
    def report(self):
        """ """
        # Storing data in csv for reporting purposes
        # Printing average reward (using f strings to make it informative)
        # Printing average regret (using f strings to make it informative)
        pass

# =============================
# 1. ALGORITHM IMPLEMENTATIONS
# =============================

class EpsilonGreedy(Bandit):
    """Epsilon-Greedy algorithm implementation for a Multi-Armed Bandit problem."""

    def __init__(self, p):
        """
        Initialize the EpsilonGreedy bandit with true reward probabilities.

        Parameters
        ----------
        p : list or numpy.ndarray
            True mean rewards for each arm.
        """
        self.p = np.array(p)
        self.num_arms = len(p)
        self.N = np.zeros(self.num_arms)
        self.m_estimate = np.zeros(self.num_arms)

        # Tracking state for the abstract pull and update cycle
        self.last_pulled_arm = None
        self.last_reward = None

        # Tracking experiment logs for later visualization and reporting
        self.rewards_log = []
        self.arms_log = []
        self.regret_log = []
        self.optimal_mean = np.max(self.p)

    def __repr__(self):
        """Return the string representation of the algorithm for logging."""
        return "EpsilonGreedy"

    def pull(self, t):
        """Pull an arm using the epsilon-greedy strategy with decaying epsilon.

        Parameters
        ----------
        t : int
            The current trial number (time step).

        Returns
        -------
        float
            The simulated reward from the environment.
        """
        # Decaying epsilon by 1/t to gradually shift towards exploitation
        epsilon = 1.0 / t

        # Deciding between exploration and exploitation using the epsilon probability
        if np.random.random() < epsilon:
            j = np.random.choice(self.num_arms) # Choosing a random arm for exploration
        else:
            j = np.argmax(self.m_estimate)      # Choosing the best known arm for exploitation

        # Drawing a reward from a Gaussian centered at true mean with variance 1 to simulate the environment
        x = np.random.randn() + self.p[j]

        self.last_pulled_arm = j
        self.last_reward = x
        return x

    def update(self):
        """Update the estimated mean of the last pulled arm using the received reward."""
        j = self.last_pulled_arm
        x = self.last_reward
        self.N[j] += 1

        # Updating the sample mean in O(1) space to prevent memory issues over many trials
        self.m_estimate[j] = ((self.N[j] - 1) * self.m_estimate[j] + x) / self.N[j]

    def experiment(self, num_trials):
        """Run the complete experiment loop for the specified number of trials.

        Parameters
        ----------
        num_trials : int
            Total number of times to pull arms.
        """
        logger.info(f"Starting {self.__repr__()} experiment...")
        for t in range(1, num_trials + 1):
            reward = self.pull(t)
            self.update()

            # Appending to logs for tracking experiment history
            self.rewards_log.append(reward)
            self.arms_log.append(self.last_pulled_arm)
            self.regret_log.append(self.optimal_mean - self.p[self.last_pulled_arm])

    def report(self):
        """Calculate metrics and structure data into a DataFrame for exporting.

        Returns
        -------
        pandas.DataFrame
            The logged data of the experiment.
        """
        # Calculating cumulative metrics for the final printout
        cum_reward = np.sum(self.rewards_log)
        cum_regret = np.sum(self.regret_log)

        logger.success(f"{self.__repr__()} Report:")
        print(f"Cumulative Reward: {cum_reward:.4f}")
        print(f"Cumulative Regret: {cum_regret:.4f}")

        # Structuring data into a DataFrame for merging and exporting to CSV
        df = pd.DataFrame({'Bandit': self.arms_log,
                          'Reward': self.rewards_log,
                          'Algorithm': self.__repr__()
                          })
        return df


class ThompsonSampling(Bandit):
    """Thompson Sampling algorithm implementation for Gaussian rewards."""

    def __init__(self, p):
        """
        Initialize the ThompsonSampling bandit with true reward means.

        Parameters
        ----------
        p : list or numpy.ndarray
            True mean rewards for each arm.
        """
        self.p = np.array(p)
        self.num_arms = len(p)

        # Setting known precision parameters for the Gaussian distribution updates
        self.tau = 1.0                        # Setting likelihood precision
        self.lambda_ = np.ones(self.num_arms) # Setting prior precision (lambda_0 = 1)
        self.m = np.zeros(self.num_arms)      # Setting prior mean (m_0 = 0)
        self.sum_x = np.zeros(self.num_arms)

        # Tracking state for the abstract pull and update cycle
        self.last_pulled_arm = None
        self.last_reward = None

        # Tracking experiment logs for later visualization and reporting
        self.rewards_log = []
        self.arms_log = []
        self.regret_log = []
        self.optimal_mean = np.max(self.p)

    def __repr__(self):
        """Return the string representation of the algorithm for logging."""
        return "ThompsonSampling"

    def pull(self):
        """Pull an arm by sampling from the posterior distribution.

        Returns
        -------
        float
            The simulated reward from the environment.
        """
        # Sampling from posterior distributions to perform Thompson Sampling logic
        samples = np.random.randn(self.num_arms) / np.sqrt(self.lambda_) + self.m
        j = np.argmax(samples)

        # Drawing actual reward from the environment using true mean and precision tau
        x = np.random.randn() / np.sqrt(self.tau) + self.p[j]

        self.last_pulled_arm = j
        self.last_reward = x
        return x

    def update(self):
        """Update the posterior precision and mean using the latest reward."""
        j = self.last_pulled_arm
        x = self.last_reward

        # Updating posterior precision and mean for Bayesian inference
        self.lambda_[j] += self.tau
        self.sum_x[j] += x
        self.m[j] = (self.tau * self.sum_x[j]) / self.lambda_[j]

    def experiment(self, num_trials):
        """Run the complete experiment loop for the specified number of trials.

        Parameters
        ----------
        num_trials : int
            Total number of times to pull arms.
        """
        logger.info(f"Starting {self.__repr__()} experiment...")
        for _ in range(num_trials):
            reward = self.pull()
            self.update()

            # Appending to logs for tracking experiment history
            self.rewards_log.append(reward)
            self.arms_log.append(self.last_pulled_arm)
            self.regret_log.append(self.optimal_mean - self.p[self.last_pulled_arm])

    def report(self):
        """Calculate metrics and structure data into a DataFrame for exporting.

        Returns
        -------
        pandas.DataFrame
            The logged data of the experiment.
        """
        # Calculating cumulative metrics for the final printout
        cum_reward = np.sum(self.rewards_log)
        cum_regret = np.sum(self.regret_log)

        logger.success(f"{self.__repr__()} Report:")
        print(f"Cumulative Reward: {cum_reward:.4f}")
        print(f"Cumulative Regret: {cum_regret:.4f}")

        # Structuring data into a DataFrame for merging and exporting to CSV
        df = pd.DataFrame({
            'Bandit': self.arms_log,
            'Reward': self.rewards_log,
            'Algorithm': self.__repr__()
        })
        return df

# ================================
# 2. VISUALIZATION AND COMPARISON
# ================================

class Visualization():
    """Class for handling the plotting of experiment results."""

    def plot1(self, eg_rewards, ts_rewards):
        """Plot the cumulative average rewards on linear and log scales.

        Parameters
        ----------
        eg_rewards : numpy.ndarray
            Reward log from the Epsilon-Greedy experiment.
        ts_rewards : numpy.ndarray
            Reward log from the Thompson Sampling experiment.
        """
        eg_cum_avg = np.cumsum(eg_rewards) / (np.arange(len(eg_rewards)) + 1)
        ts_cum_avg = np.cumsum(ts_rewards) / (np.arange(len(ts_rewards)) + 1)

        fig, axes = plt.subplots(1, 2, figsize=(20, 8))

        # Plotting the linear scale comparison to visualize standard learning curves
        axes[0].plot(eg_cum_avg, label='Epsilon-Greedy')
        axes[0].plot(ts_cum_avg, label='Thompson Sampling')
        axes[0].set_title("Learning Process: Cumulative Avg Reward (Linear)")
        axes[0].set_xlabel("Number of Trials")
        axes[0].set_ylabel("Average Reward")
        axes[0].legend()

        # Plotting the log scale comparison to better visualize early exploration stages
        axes[1].plot(eg_cum_avg, label='Epsilon-Greedy')
        axes[1].plot(ts_cum_avg, label='Thompson Sampling')
        axes[1].set_title("Learning Process: Cumulative Avg Reward (Log)")
        axes[1].set_xlabel("Number of Trials")
        axes[1].set_ylabel("Average Reward")
        axes[1].set_xscale('log')
        axes[1].legend()

        plt.tight_layout()
        plt.show()

    def plot2(self, eg_rewards, ts_rewards, eg_regrets, ts_regrets):
        """Plot the cumulative rewards and cumulative regrets.

        Parameters
        ----------
        eg_rewards : numpy.ndarray
            Reward log from the Epsilon-Greedy experiment.
        ts_rewards : numpy.ndarray
            Reward log from the Thompson Sampling experiment.
        eg_regrets : list
            Regret log from the Epsilon-Greedy experiment.
        ts_regrets : list
            Regret log from the Thompson Sampling experiment.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plotting cumulative rewards to show total payoff over time
        axes[0].plot(np.cumsum(eg_rewards), label='Epsilon-Greedy')
        axes[0].plot(np.cumsum(ts_rewards), label='Thompson Sampling')
        axes[0].set_title("Cumulative Rewards Comparison")
        axes[0].set_xlabel("Number of Trials")
        axes[0].set_ylabel("Cumulative Reward")
        axes[0].legend()

        # Plotting cumulative regrets to show total loss against optimal strategy
        axes[1].plot(np.cumsum(eg_regrets), label='Epsilon-Greedy')
        axes[1].plot(np.cumsum(ts_regrets), label='Thompson Sampling')
        axes[1].set_title("Cumulative Regrets Comparison")
        axes[1].set_xlabel("Number of Trials")
        axes[1].set_ylabel("Cumulative Regret")
        axes[1].legend()

        plt.tight_layout()
        plt.show()


def comparison():
    """Main function to execute the bandit algorithms and generate comparisons."""
    BANDIT_REWARD = [1, 2, 3, 4]
    NUM_TRIALS = 20000

    eg = EpsilonGreedy(BANDIT_REWARD)
    ts = ThompsonSampling(BANDIT_REWARD)

    eg.experiment(NUM_TRIALS)
    ts.experiment(NUM_TRIALS)

    df_eg = eg.report()
    df_ts = ts.report()

    # Concatenating the dataframes to save all experiment data in one unified CSV
    final_df = pd.concat([df_eg, df_ts])
    csv_path = "experiment_results.csv"
    final_df.to_csv(csv_path, index=False)
    logger.info(f"Data successfully saved to {csv_path}")

    viz = Visualization()
    viz.plot1(df_eg['Reward'].values, df_ts['Reward'].values)
    viz.plot2(df_eg['Reward'].values, df_ts['Reward'].values, eg.regret_log, ts.regret_log)

if __name__=='__main__':
    comparison()


# ======================================
# 3. SUGGEST BETTER IMPLEMENTATION PLAN
# ======================================
"""
The first improvement I would make is adding `np.random.seed(42)` inside `comparison()`.
This makes both algorithms run under the same random conditions, which makes the comparison
more fair and the results reproducible across runs. Note: reproducibility was not listed
as a requirement, but would be a natural first improvement.

For performance, the current loop with `.append()` works fine for 20k trials, but if the
number of trials gets much larger, switching to pre-allocated `np.zeros(num_trials)` arrays
would make it noticeably faster.

Since `report()` already saves a CSV, it would make sense to also save the plots using
`plt.savefig()` — that way everything from one run is stored together in one place.

In terms of future extensions, UCB could be added as a third subclass without changing
anything that already exists, which would be a good way to test if the abstract class
structure actually scales. To make testing different setups easier, it would also help
to turn `BANDIT_REWARD` and `NUM_TRIALS` into parameters of `comparison()` instead of
hardcoding them.
"""
