import numpy as np
def start_goal_gen(x_low, x_high, collision_checker, eps):
    # generate samples that are not in collision
    while True:
        x_start = np.random.uniform(low=x_low, high=x_high)
        # check if start is in collision
        if collision_checker.invalid(x_start):
            continue
        x_goal = np.random.uniform(low=x_low, high=x_high)
        if collision_checker.invalid(x_goal):
            continue

        # we may also want to make sure x_start and x_goal are "interesting"
        if np.linalg.norm(x_goal - x_start) <= 0.3 * np.max(x_high - x_low):
            continue
        # we don't want the path to be straight line
        if not collision_checker.line_invalid(x_start, x_goal, eps):
            continue
        
        return x_start, x_goal