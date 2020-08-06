import ray
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.pg import PGTrainer
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from wanderer_roborobo import WandererRoborobo
from ray.rllib.models import ModelCatalog
from negotiate_model import NegotiateModelReference

if __name__ == "__main__":

    ray.init(num_cpus=8, num_gpus=0)

    #%%
    ModelCatalog.register_custom_model(
        "negotiate", NegotiateModelReference)

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
        return 'player0'


    config = {
        "num_gpus": 0,
        'num_workers': 0,
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": select_policy,
            },
        "clip_actions": True,
        "framework": "torch",
        #"num_sgd_iter": 4,
        "lr": 1e-4,
        #"kl_target": 0.03,
        #"train_batch_size": 1024,
        "rollout_fragment_length": 100,
        #"sgd_minibatch_size": 32
    }

    trainer = PPOTrainer(env="wanderer_roborobo", config=config)
    print(trainer.config.get('no_final_linear'))
    print('model built')
    stop_iter = 2000

    #%%
    import numpy as np
    for i in range(stop_iter):
        print("== Iteration", i, "==")
        result_ppo = trainer.train()
        pretty_print(result_ppo)
        if (i+1) % 200 == 0:
            trainer.save('model')
    trainer.save('model')
    del trainer
    ray.shutdown()
