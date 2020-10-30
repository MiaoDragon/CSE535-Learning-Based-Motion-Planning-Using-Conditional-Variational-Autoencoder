import sys
sys.path.insert(1, '/freespace/local/ym420/course/cse535/CSE535-Learning-Based-Motion-Planning-Using-Conditional-Variational-Autoencoder/FasterRobusterMotionPlanningLibrary/python')
import numpy as np

from planners.planner import Planner as Planner
from env.s2d_utility import Environment, CollisionChecker, Metrics
from nearest_computer.naive_nearest_neighbor import NearestComputer
from planner_structure.plan_tree import PlanTree

from tools.data_loader_s2d import load_test_dataset


obs_center,obs_pcd,paths,path_lengths = load_test_dataset(N=1,NP=1, s=0,sp=4000, folder='/freespace/local/ym420/course/cse535/data/s2d/')

obs_width = 5.0
print('obstacle center:')
print(obs_center.shape)

obs_center = obs_center[0]
paths = paths[0][0]
path_lengths = path_lengths[0][0]
paths = paths[:path_lengths]
# create environment
env = Environment(obs_center, obs_width)
collision_checker = CollisionChecker(env)
d_metrics = Metrics()
nearest_computer = NearestComputer()
planner_structure = PlanTree()
class Sampler():
    def sample(self):
        # uniformly sample in the state space
        return np.random.uniform(low=env.x_lower_bound, high=env.x_upper_bound)



sampler = Sampler()
planner = Planner(sampler, nearest_computer, d_metrics, collision_checker, planner_structure)
eps = 0.01
radius = 5.
planner.setup(paths[0], paths[-1], 0.1, env, eps, radius)
planner.step()