if __name__ == '__main__':
    from negotiate_roborobo import NegotiateRoborobo
    import numpy as np

    env = NegotiateRoborobo(1)

    obs = env.reset()
    print(obs)

    for i in range(1000):
        everything = env.step({'player0': np.array([1, 0, 1, 1])})
        print(everything)

    del env
