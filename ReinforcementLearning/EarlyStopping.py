from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class EarlyStoppingCallback(BaseCallback):
    def __init__(self, patience=10, verbose=1):
        super(EarlyStoppingCallback, self).__init__(verbose)
        self.patience = patience
        self.best_mean_reward = -np.inf
        self.counter = 0
        self.episode_rewards = []  # List to keep track of rewards

    def _on_step(self) -> bool:
        # Append reward information to the list
        if self.locals.get('rewards') is not None:
            self.episode_rewards.extend(self.locals['rewards'])

        # Check if it's time to compute mean reward
        if self.n_calls % self.model.n_envs == 0:
            if len(self.episode_rewards) > 0:
                mean_reward = np.mean(self.episode_rewards)
                self.episode_rewards = []  # Reset rewards list after calculation

                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    self.counter = 0  # Reset counter if improvement is observed
                else:
                    self.counter += 1  # Increment counter if no improvement

                # Check if patience has been exceeded
                if self.counter >= self.patience:
                    print(f"Early stopping triggered after {self.n_calls} steps")
                    return False  # Stop training

        return True  # Continue training
