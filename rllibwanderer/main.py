import ray
from ray import tune
from ray.tune.logger import pretty_print

from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.a3c import A3CTrainer
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy

from ray.tune.registry import register_env
from wanderer_roborobo import WandererRoborobo


if __name__ == "__main__":

    ray.init(num_cpus=8, num_gpus=1)

    #%%

    n_players = 100
    max_moves = 10000
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
        "num_gpus": 1,
        'num_workers': 1,
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": select_policy,
            },
        'model': {
            'fcnet_hiddens': [4, 4]
        },
        "clip_actions": True,
        "use_pytorch": True,
        "rollout_fragment_length": 200 * n_players
    }

    trainer = A3CTrainer(env="wanderer_roborobo", config=config)
    print('model built')
    stop_iter = 2000

    #%%
    import numpy as np
    for i in range(stop_iter):
        print("== Iteration", i, "==")
        result_ppo = trainer.train()
        print(str(result_ppo)[:100])
        if (i+1) % 200 == 0:
            trainer.save('model')
    trainer.save('model')
