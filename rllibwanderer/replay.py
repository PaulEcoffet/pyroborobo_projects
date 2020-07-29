import ray
from ray import tune
from ray.tune.logger import pretty_print

from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy

from ray.tune.registry import register_env
from wanderer_roborobo import WandererRoborobo


if __name__ == "__main__":

    ray.init(num_cpus=1, num_gpus=1)

    #%%

    n_players = 1
    max_moves = 1000
    agents_id = ['player{:d}'.format(i) for i in range(n_players)]
    actions = {agents_id[i]: 1 for i in range(n_players)}

    register_env("wanderer_roborobo",
                 lambda _: WandererRoborobo(n_players, max_moves))
    act_space = WandererRoborobo.action_space

    obs_space = WandererRoborobo.observation_space

    policies = {agents_id[i]: (None, obs_space, act_space, {}) for i in range(n_players)}


    def select_policy(agent_id):
        return agent_id


    config = {
        "num_gpus": 0,
        'num_workers': 0,
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": select_policy,
        },
        'model': {
            'fcnet_hiddens': [4, 4]
        },
        "clip_actions": True,
        "framework": "torch",
        "num_sgd_iter": 4,
        "lr": 1e-4,
        "kl_target": 0.03,
        "train_batch_size": 256,
        "rollout_fragment_length": 128
    }

    trainer = PPOTrainer(env="wanderer_roborobo", config=config)
    trainer.restore('model/checkpoint_2000/checkpoint-2000')
    stop_iter = 2000

    #%%
    import numpy as np
    for i in range(stop_iter):
        print("== Iteration", i, "==")
        result_ppo = trainer.train()
