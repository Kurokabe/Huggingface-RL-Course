# Virtual display
from pyvirtualdisplay import Display

virtual_display = Display(visible=0, size=(1400, 900))
virtual_display.start()

import os

import gym
import panda_gym
import pybullet_envs
from huggingface_sb3 import load_from_hub, package_to_hub
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from huggingface_hub import HfFolder


def create_env(env_id):
    # Create the env
    env = make_vec_env(env_id, n_envs=4)
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # Create the evaluation env
    eval_env = DummyVecEnv([lambda: gym.make(env_id)])
    # eval_env = VecNormalize.load("vec_normalize.pkl", eval_env)

    #  do not update them at test time
    eval_env.training = False
    # reward normalization is not needed at test time
    eval_env.norm_reward = False

    # Get the state space and action space
    s_size = env.observation_space.shape
    a_size = env.action_space
    return env, eval_env, s_size, a_size


if __name__ == "__main__":

    HfFolder.save_token(os.environ["TOKEN"])

    env_id = "PandaReachDense-v2"

    env, eval_env, s_size, a_size = create_env(env_id)

    model = A2C(
        policy="MultiInputPolicy",
        env=env,
        gae_lambda=0.9,
        gamma=0.99,
        learning_rate=0.00096,
        max_grad_norm=0.5,
        n_steps=8,
        vf_coef=0.4,
        ent_coef=0.0,
        policy_kwargs=dict(log_std_init=-2, ortho_init=False),
        normalize_advantage=False,
        use_rms_prop=True,
        use_sde=True,
        verbose=1,
    )

    model.learn(2_000_000)

    # Save the model and  VecNormalize statistics when saving the agent
    mean_reward, std_reward = evaluate_policy(model, env)

    print(f"Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")

    package_to_hub(
        model=model,
        model_name=f"a2c-{env_id}",
        model_architecture="A2C",
        env_id=env_id,
        eval_env=eval_env,
        repo_id=f"Kurokabe/a2c-{env_id}",  # Change the username
        commit_message="Initial commit",
    )
