if __name__ == '__main__':
    from wanderer_roborobo import WandererRoborobo
    w = WandererRoborobo(10, 10000)
    w.reset()
    act = {'player'+str(i): [1, -15] for i in range(10)}
    for _ in range(1000):
        w.step(act)
    w.reset()
    for _ in range(1000):
        w.step(act)
 

