import imageio
def obs_load(folder, N, s):
    folder = folder + 'train/'
    obs = []
    for i in range(N):
        obs_i = imageio.imread(folder+'%d.png' % (i+s))
        obs.append(obs_i)
    return obs
