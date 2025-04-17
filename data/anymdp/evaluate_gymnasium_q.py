import gymnasium as gym
import xenoverse
import numpy
import multiprocessing
import argparse
import sys
sys.path.append("../../projects/OmniRL")
from gym_env_wapper import DiscreteEnvWrapper

from xenoverse.anymdp import AnyMDPTaskSampler
from xenoverse.anymdp import AnyMDPSolverOpt, AnyMDPSolverOTS, AnyMDPSolverQ
from xenoverse.utils import pseudo_random_seed

def make_env():
    env = DiscreteEnvWrapper(gym.make("Pendulum-v1", g=9.81), 
            'pendulum', 
            action_space=5, 
            state_space_dim1=12, 
            state_space_dim2=5, 
            reward_shaping = False, 
            skip_frame=0)
    env_opt_reward = -385
    return env, env_opt_reward

def test_AnyMDP_task(result, 
                     max_epochs_rnd=200, 
                     max_epochs_q=10000,
                     sub_sample=100, 
                     gamma=0.999,
                     exploration=0.005,
                     lr=0.01):
    env, opt_perf = make_env()
    epoch_rews_rnd = []
    epoch_rews_q =[]
    epoch_steps_q = []

    for epoch in range(max_epochs_rnd):
        obs, info = env.reset()
        terminated, truncated = False, False
        epoch_rew = 0
        while not terminated and not truncated:
            act = int(env.action_space.sample())
            obs, rew, terminated, truncated, info = env.step(act)
            epoch_rew += rew
        epoch_rews_rnd.append(epoch_rew)

    rnd_perf = numpy.mean(epoch_rews_rnd)

    solver_q = AnyMDPSolverQ(env, gamma=gamma, c=exploration, alpha=lr)
    for epoch in range(max_epochs_q):
        last_obs, info = env.reset()
        epoch_rew = 0
        epoch_step = 0
        terminated, truncated = False, False
        while not terminated and not truncated:
            act = solver_q.policy(last_obs)
            obs, rew, terminated, truncated, info = env.step(act)
            solver_q.learner(last_obs, act, obs, rew, terminated or truncated)
            epoch_rew += rew
            epoch_step += 1
            last_obs = obs
        epoch_rews_q.append(epoch_rew)
        epoch_steps_q.append(epoch_step)

    normalized_q = (numpy.array(epoch_rews_q) - rnd_perf) / max(1.0e-2, opt_perf - rnd_perf)
    steps = normalized_q.shape[0] // sub_sample
    eff_size = steps * sub_sample
    normalized_q = numpy.reshape(normalized_q[:eff_size], (-1, sub_sample))
    normalized_steps = numpy.reshape(numpy.array(epoch_steps_q)[:eff_size], (-1, sub_sample))
    result.put((numpy.mean(normalized_q, axis=-1), 
                numpy.sum(normalized_steps, axis=-1), opt_perf - rnd_perf))
            
if __name__=="__main__":
    # Parse the arguments, should include the output file name
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default="./res.txt", help="Output result path")
    parser.add_argument("--max_steps", type=int, default=4000, help="max steps, default:4000")
    parser.add_argument("--max_epochs", type=int, default=10000, help="multiple epochs:default:1000")
    parser.add_argument("--workers", type=int, default=4, help="number of multiprocessing workers")
    parser.add_argument("--sub_sample", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--exploration", type=float, default=0.005)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    args = parser.parse_args()

    # Data Generation
    processes = []
    res = multiprocessing.Manager().Queue()
    for worker_id in range(args.workers):
        process = multiprocessing.Process(target=test_AnyMDP_task, 
                args=(res,
                      200, 
                      args.max_epochs, 
                      args.sub_sample,
                      args.gamma,
                      args.exploration,
                      args.learning_rate))
        processes.append(process)
        process.start()

    for process in processes:
        process.join() 
    
    scores = []
    steps = []
    deltas = []
    while not res.empty():
        score, step, delta = res.get()
        scores.append(score)
        steps.append(step)
        deltas.append(delta)

    scores = numpy.array(scores)
    s_mean = numpy.mean(scores, axis=0)
    s2_mean = numpy.mean(scores**2, axis=0)
    std = numpy.sqrt(s2_mean - s_mean**2)
    conf = 2.0 * std / numpy.sqrt(scores.shape[0])

    steps = numpy.cumsum(numpy.array(steps), axis=-1)
    sp_mean = numpy.mean(steps, axis=0)
    sp2_mean = numpy.mean(steps**2, axis=0)
    pstd = numpy.sqrt(sp2_mean - sp_mean**2)

    deltas = numpy.array(deltas)
    d_mean = numpy.mean(deltas)
    d2_mean = numpy.mean(deltas**2)
    dstd = numpy.sqrt(d2_mean - d_mean**2)

    fout = open(args.output_path, "w")
    for i in range(s_mean.shape[0]):
        fout.write("%d\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n" % 
              ((i+1)*args.sub_sample, s_mean[i], conf[i], sp_mean[i], pstd[i], d_mean, dstd))
    fout.close()
