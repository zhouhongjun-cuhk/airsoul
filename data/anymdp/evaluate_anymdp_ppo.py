import gym
import xenoverse
import numpy
import multiprocessing
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from xenoverse.anymdp import AnyMDPTaskSampler
from xenoverse.anymdp import AnyMDPSolverOpt, AnyMDPSolverOTS, AnyMDPSolverQ
from xenoverse.utils import pseudo_random_seed


class RolloutLogger(BaseCallback):
    """
    A custom callback for logging the total reward and episode length of each rollout.
    
    :param env_name: Name of the environment.
    :param max_rollout: Maximum number of rollouts to perform.
    :param max_step: Maximum steps per episode.
    :param downsample_trail: Downsample trail parameter.
    :param verbose: Verbosity level: 0 = no output, 1 = info, 2 = debug
    """
    def __init__(self, max_rollout, max_step, downsample_trail, verbose=0):
        super(RolloutLogger, self).__init__(verbose)
        self.max_rollout = max_rollout
        self.max_steps = max_step
        self.current_rollout = 0
        self.reward_sums = []
        self.step_counts = []
        self.success_rate = []
        self.success_rate_f = 0.0
        self.downsample_trail = downsample_trail
        self.episode_reward = 0
        self.episode_length = 0

    def _on_step(self) -> bool:
        """
        This method is called after every step in the environment.
        Here we update the current episode's reward and length.
        """
        # Accumulate the episode reward
        self.episode_reward += self.locals['rewards'][0]
        self.episode_length += 1
        
        if 'terminated' in self.locals:
            terminated = self.locals['terminated'][0]
        elif 'dones' in self.locals:  # Fallback to 'done' flag
            done = self.locals['dones'][0]
            terminated = done  # Assuming 'done' means the episode has ended, either successfully or due to failure

        if 'truncated' in self.locals:
            truncated = self.locals['truncated'][0]
        elif 'infos' in self.locals and len(self.locals['infos']) > 0:
            info = self.locals['infos'][0]
            truncated = info.get('TimeLimit.truncated', False)

        if terminated or truncated:
            # Episode is done, record the episode information

            self.reward_sums.append(self.episode_reward)
            self.step_counts.append(self.episode_length)

            # Reset episode counters
            self.episode_reward = 0
            self.episode_length = 0

            # Check if we have reached the maximum number of rollouts
            self.current_rollout += 1
            if self.current_rollout >= self.max_rollout:
                if self.verbose >= 1:
                    print(f"Reached maximum rollouts ({self.max_rollout}). Stopping training.")
                self.model.stop_training = True
                return False
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        For algorithms that do not use rollout_buffer, this method can be left empty.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered at the end of training.
        We can perform any final logging here if needed.
        """
        pass

class WrapperEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        obs, info = super().reset()
        return obs

    def step(self, action):
        obs, reward, done, info = super().step(action)
        return obs, reward, done, info

def test_AnyMDP_task(result, ns=16, na=5, 
                     max_epochs_rnd=200, max_epochs_q=5000, 
                     sub_sample=100, gamma=0.9):
    env = gym.make("anymdp-v0")
    task = AnyMDPTaskSampler(ns, na)
    env.set_task(task)
    epoch_rews_rnd = []
    epoch_rews_opt = []
    epoch_rews_q =[]
    epoch_steps_q = []

    for epoch in range(max_epochs_rnd):
        obs, info = env.reset()
        epoch_rew = 0
        done = False
        while not done:
            act = env.action_space.sample()
            obs, rew, done, info = env.step(act)
            epoch_rew += rew
        epoch_rews_rnd.append(epoch_rew)

    solver_opt = AnyMDPSolverOpt(env, gamma=gamma)
    for epoch in range(max_epochs_rnd):
        obs, info = env.reset()
        epoch_rew = 0
        done = False
        while not done:
            act = solver_opt.policy(obs)
            obs, rew, done, info = env.step(act)
            epoch_rew += rew
        epoch_rews_opt.append(epoch_rew)

    rnd_perf = numpy.mean(epoch_rews_rnd)
    opt_perf = numpy.mean(epoch_rews_opt)
    if(opt_perf - rnd_perf < 1.0e-2):
        print("[Trivial task], skip")
        return

    log_callback = RolloutLogger(max_epochs_q, 4000, sub_sample, verbose=1)
    model = PPO(policy='MlpPolicy', env=WrapperEnv(env), verbose=1)
    model.learn(total_timesteps=int(4e6), callback=log_callback)
    epoch_rews_q = log_callback.reward_sums
    epoch_steps_q = log_callback.step_counts

    normalized_q = (numpy.array(epoch_rews_q) - rnd_perf) / max(1.0e-2, opt_perf - rnd_perf)
    steps = normalized_q.shape[0] // sub_sample
    eff_size = steps * sub_sample
    normalized_q = numpy.reshape(normalized_q[:eff_size], (-1, sub_sample))
    normalized_steps = numpy.reshape(numpy.array(epoch_steps_q)[:eff_size], (-1, sub_sample))
    normalized_steps = numpy.cumsum(numpy.sum(normalized_steps, axis=-1), axis=0)
    result.put((numpy.mean(normalized_q, axis=-1), normalized_steps, opt_perf - rnd_perf))
            
if __name__=="__main__":
    # Parse the arguments, should include the output file name
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default="./res.txt", help="Output result path")
    parser.add_argument("--state_num", type=int, default=128, help="state num, default:128")
    parser.add_argument("--action_num", type=int, default=5, help="action num, default:5")
    parser.add_argument("--min_state_space", type=int, default=16, help="minimum state dim in task, default:8")
    parser.add_argument("--max_steps", type=int, default=4000, help="max steps, default:4000")
    parser.add_argument("--max_epochs", type=int, default=20, help="multiple epochs:default:1000")
    parser.add_argument("--workers", type=int, default=4, help="number of multiprocessing workers")
    parser.add_argument("--sub_sample", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    args = parser.parse_args()

    # Data Generation
    processes = []
    res = multiprocessing.Manager().Queue()
    for worker_id in range(args.workers):
        process = multiprocessing.Process(target=test_AnyMDP_task, 
                args=(res,
                      int(args.state_num), 
                      int(args.action_num),
                      200, 
                      args.max_epochs, 
                      args.sub_sample,
                      args.gamma))
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

    steps = numpy.cumsum(numpy.array(steps))
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