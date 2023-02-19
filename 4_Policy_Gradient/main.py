import os
from collections import deque

import gym
import gym_pygame
import numpy as np
import torch
import torch.optim as optim
from huggingface_hub import HfFolder, whoami
from src.model import Policy
from src.push_to_hub import push_to_hub


def create_env(env_id):
    # Create the env
    env = gym.make(env_id)

    # Create the evaluation env
    eval_env = gym.make(env_id)

    # Get the state space and action space
    s_size = env.observation_space.shape[0]
    a_size = env.action_space.n
    return env, eval_env, s_size, a_size


def reinforce(policy, optimizer, n_training_episodes, max_t, gamma, print_every):
    # Help us to calculate the score during the training
    scores_deque = deque(maxlen=100)
    scores = []
    # Line 3 of pseudocode
    for i_episode in range(1, n_training_episodes + 1):
        saved_log_probs = []
        rewards = []
        state = env.reset()
        # Line 4 of pseudocode
        for t in range(max_t):
            action, log_prob = policy.act(state)
            saved_log_probs.append(log_prob)
            state, reward, done, _ = env.step(action)
            rewards.append(reward)
            if done:
                break
        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))

        # Line 6 of pseudocode: calculate the return
        returns = deque(maxlen=max_t)
        n_steps = len(rewards)
        # Compute the discounted returns at each timestep,
        # as
        #      the sum of the gamma-discounted return at time t (G_t) + the reward at time t
        #
        # In O(N) time, where N is the number of time steps
        # (this definition of the discounted return G_t follows the definition of this quantity
        # shown at page 44 of Sutton&Barto 2017 2nd draft)
        # G_t = r_(t+1) + r_(t+2) + ...

        # Given this formulation, the returns at each timestep t can be computed
        # by re-using the computed future returns G_(t+1) to compute the current return G_t
        # G_t = r_(t+1) + gamma*G_(t+1)
        # G_(t-1) = r_t + gamma* G_t
        # (this follows a dynamic programming approach, with which we memorize solutions in order
        # to avoid computing them multiple times)

        # This is correct since the above is equivalent to (see also page 46 of Sutton&Barto 2017 2nd draft)
        # G_(t-1) = r_t + gamma*r_(t+1) + gamma*gamma*r_(t+2) + ...

        ## Given the above, we calculate the returns at timestep t as:
        #               gamma[t] * return[t] + reward[t]
        #
        ## We compute this starting from the last timestep to the first, in order
        ## to employ the formula presented above and avoid redundant computations that would be needed
        ## if we were to do it from first to last.

        ## Hence, the queue "returns" will hold the returns in chronological order, from t=0 to t=n_steps
        ## thanks to the appendleft() function which allows to append to the position 0 in constant time O(1)
        ## a normal python list would instead require O(N) to do this.
        for t in range(n_steps)[::-1]:
            disc_return_t = returns[0] if len(returns) > 0 else 0
            returns.appendleft(gamma * disc_return_t + rewards[t])

        ## standardization of the returns is employed to make training more stable
        eps = np.finfo(np.float32).eps.item()
        ## eps is the smallest representable float, which is
        # added to the standard deviation of the returns to avoid numerical instabilities
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)

        # Line 7:
        policy_loss = []
        for log_prob, disc_return in zip(saved_log_probs, returns):
            policy_loss.append(-log_prob * disc_return)
        policy_loss = torch.cat(policy_loss).sum()

        # Line 8: PyTorch prefers gradient descent
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        if i_episode % print_every == 0:
            print(
                "Episode {}\tAverage Score: {:.2f}".format(
                    i_episode, np.mean(scores_deque)
                )
            )

    return scores


if __name__ == "__main__":
    env_id = "Pixelcopter-PLE-v0"
    env, eval_env, s_size, a_size = create_env(env_id)

    HfFolder.save_token(os.environ["TOKEN"])

    print(whoami())

    pixelcopter_hyperparameters = {
        "h_size": 64,
        "n_training_episodes": 1000000,
        "n_evaluation_episodes": 10,
        "max_t": 50000,
        "gamma": 0.99,
        "lr": 1e-4,
        "env_id": env_id,
        "state_space": s_size,
        "action_space": a_size,
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using", device)

    # Create policy and place it to the device
    pixelcopter_policy = Policy(
        pixelcopter_hyperparameters["state_space"],
        pixelcopter_hyperparameters["action_space"],
        pixelcopter_hyperparameters["h_size"],
        device=device,
    ).to(device)
    pixelcopter_optimizer = optim.Adam(
        pixelcopter_policy.parameters(), lr=pixelcopter_hyperparameters["lr"]
    )

    print("Training...")
    scores = reinforce(
        pixelcopter_policy,
        pixelcopter_optimizer,
        pixelcopter_hyperparameters["n_training_episodes"],
        pixelcopter_hyperparameters["max_t"],
        pixelcopter_hyperparameters["gamma"],
        1000,
    )

    repo_id = "Kurokabe/Reinforce-Pixelcopter-PLE"
    push_to_hub(
        repo_id,
        pixelcopter_policy,  # The model we want to save
        pixelcopter_hyperparameters,  # Hyperparameters
        eval_env,  # Evaluation environment
        video_fps=30,
    )
