import os
import random
import time

import click
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

from src.agent import Agent


@click.command()
@click.option(
    "--exp-name",
    default=os.path.basename(__file__).rstrip(".py"),
    help="The name of this experiment",
)
@click.option(
    "--gym-id", type=str, default="CartPole-v1", help="The id of the gym environment"
)
@click.option(
    "--learning-rate",
    type=float,
    default=2.5e-4,
    help="The learning rate of the optimizer",
)
@click.option("--seed", type=int, default=1, help="The seed of the experiment")
@click.option(
    "--total-timesteps",
    type=int,
    default=25000,
    help="The total timesteps of the experiments",
)
@click.option(
    "--torch-deterministic",
    type=bool,
    default=True,
    help="Whether to set `torch.backends.cudnn.deterministic=True`",
)
@click.option(
    "--cuda", type=bool, default=True, help="Whether to use CUDA whenever possible"
)
@click.option(
    "--capture-video",
    type=bool,
    default=False,
    help="Whether to capture videos of the agent performances (check out `videos` folder)",
)
@click.option(
    "--num-envs", type=int, default=4, help="How many envs to run in parallel"
)
@click.option(
    "--num-steps",
    type=int,
    default=128,
    help="The number of steps to run in each environment per policy rollout",
)
@click.option(
    "--anneal-lr",
    type=bool,
    default=True,
    help="Toggle learning rate anealing for policy and value networks",
)
@click.option(
    "--gae", type=bool, default=True, help="Toggle Generalized Advantage Estimation"
)
@click.option("--gamma", type=float, default=0.99, help="The discount factor gamma")
@click.option(
    "--gae-lambda",
    type=float,
    default=0.95,
    help="The lambda for the general advantage estimation",
)
@click.option(
    "--num-minibatches",
    type=int,
    default=4,
    help="The number of mini-batches",
)
@click.option(
    "--update-epochs",
    type=int,
    default=4,
    help="The K epochs to update the policy",
)
@click.option(
    "--norm-adv",
    type=bool,
    default=True,
    help="Toggles advantages normalization",
)
@click.option(
    "--clip-coef",
    type=float,
    default=0.2,
    help="The surrogate clipping coefficient",
)
@click.option(
    "--clip-vloss",
    type=bool,
    default=True,
    help="Toggles whether or not to use a clipped loss for the value function, as per the paper",
)
@click.option(
    "--ent-coef",
    type=float,
    default=0.01,
    help="Coefficient of the entropy",
)
@click.option(
    "--vf-coef",
    type=float,
    default=0.05,
    help="Coefficient of the value function",
)
@click.option(
    "--max-grad-norm",
    type=float,
    default=0.5,
    help="The maximum norm for the gradient clipping",
)
def main(
    exp_name,
    gym_id,
    learning_rate,
    seed,
    total_timesteps,
    torch_deterministic,
    cuda,
    capture_video,
    num_envs,
    num_steps,
    anneal_lr,
    gae,
    gamma,
    gae_lambda,
    num_minibatches,
    update_epochs,
    norm_adv,
    clip_coef,
    clip_vloss,
    ent_coef,
    vf_coef,
    max_grad_norm,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")

    envs = gym.vector.SyncVectorEnv(
        [make_env(gym_id, seed, i, capture_video, exp_name) for i in range(num_envs)]
    )

    agent = Agent(envs).to(device)

    optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)

    batch_size = num_envs * num_steps

    # ALGO Logic: Storage setup
    obs = torch.zeros((num_steps, num_envs) + envs.single_observation_space.shape).to(
        device
    )
    actions = torch.zeros((num_steps, num_envs) + envs.single_action_space.shape).to(
        device
    )
    logprobs = torch.zeros((num_steps, num_envs)).to(device)
    rewards = torch.zeros((num_steps, num_envs)).to(device)
    dones = torch.zeros((num_steps, num_envs)).to(device)
    values = torch.zeros((num_steps, num_envs)).to(device)

    global_step = 0
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(num_envs).to(device)
    num_updates = total_timesteps // batch_size

    for update in tqdm(range(1, num_updates + 1)):
        # Annealing the rate if instructed to do so.
        if anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = learning_rate * frac
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, num_steps):
            global_step += 1 * num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # Execute the game and log data.
            next_obs, reward, done, _ = envs.step(action.cpu().numpy())
            rewards[step] = torch.Tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(
                done
            ).to(device)

        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            if gae:
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(num_steps)):
                    if t == num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = (
                        rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
                    )
                    advantages[t] = lastgaelam = (
                        delta + gamma * gae_lambda * nextnonterminal * lastgaelam
                    )
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(num_steps)):
                    if t == num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns = rewards[t] + gamma * nextnonterminal * next_return
                    advantages = returns - values
        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(batch_size)

        for epoch in range(update_epochs):

            np.random.shuffle(b_inds)
            for start in range(0, batch_size, num_minibatches):
                end = start + num_minibatches
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_advantages = b_advantages[mb_inds]
                if norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - clip_coef, 1 + clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds], -clip_coef, clip_coef
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - entropy_loss * ent_coef + v_loss * vf_coef
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()


def make_env(gym_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(gym_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(
                    env,
                    f"videos/{run_name}",
                    record_video_trigger=lambda x: x % 1000 == 0,
                )
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


if __name__ == "__main__":
    main()
