import numpy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import A2C, PPO, DQN, TD3
from xenoverse.anymdp import AnyMDPSolverOpt
# For QLearning class
from gymnasium import spaces
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.utils import get_schedule_fn
from typing import Optional, Tuple, Union

class QLearning(BaseAlgorithm):
    """
    The QLearning algorithm is implemented by inheriting the base class of stable_baselines3 
    to ensure consistent logging format.
    """
    
    def __init__(
        self,
        env: GymEnv,
        learning_rate: Union[float, Schedule] = 0.1,
        discount_factor: float = 0.99,
        epsilon: float = 0.1,
        epsilon_decay: float = 0.995,
        min_epsilon: float = 0.01,
        tensorboard_log: Optional[str] = None,
        verbose: int = 0,
        _init_setup_model: bool = True,
    ):
        # Verify environment type 
        assert isinstance(env.observation_space, spaces.Discrete), "Requires a discrete state space."
        assert isinstance(env.action_space, spaces.Discrete), "Requires a discrete action space."

        # Initialize learning rate scheduler 
        self.lr_schedule = get_schedule_fn(learning_rate)
        
        # Call base class init
        super().__init__(
            policy=None,
            env=env,
            learning_rate=1.0,  # Virtual value, the actual use of custom scheduling
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            supported_action_spaces=(spaces.Discrete,),
            support_multi_env=False,
        )

        # Q-Learning parameters
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        # Q table initialization
        self.q_table = numpy.zeros(
            (self.env.observation_space.n, 
             self.env.action_space.n),
            dtype=numpy.float32
        )

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        """Virtual policy"""
        self.policy = self  # Use self as policy to ensure consistency

    def predict(
        self,
        observation: numpy.ndarray,
        state: Optional[Tuple] = None,
        episode_start: Optional[numpy.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[numpy.ndarray, Optional[Tuple]]:
        """epsilon-greedy policy"""
        state_idx = observation.item()
        if deterministic or numpy.random.rand() >= self.epsilon:
            action = numpy.argmax(self.q_table[state_idx])
        else:
            action = self.env.action_space.sample()
        return numpy.array([action], dtype=numpy.int64), state

    def learn(
        self,
        total_timesteps: int,
        callback: Optional[BaseCallback] = None,
        log_interval: int = 100,
        tb_log_name: str = "QLearning",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> "QLearning":
        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        state = self.env.reset()
        episode_reward = 0.0

        while self.num_timesteps < total_timesteps:
            # Get learning rate
            current_lr = self.lr_schedule(self._current_progress_remaining)
            
            # Action choose and execute
            action, _ = self.predict(state)
            next_state, reward, terminated, truncated, *_ = self.env.step(action)
            
            terminated = terminated[0]
            truncated = truncated[0]["TimeLimit.truncated"]

            done = terminated or truncated

            state_idx = state.item()
            action_idx = action.item()
            next_state_idx = next_state.item()

            # Q table update
            best_next_q = numpy.max(self.q_table[next_state_idx])
            td_target = reward + self.discount_factor * best_next_q
            self.q_table[state_idx, action_idx] += current_lr * (td_target - self.q_table[state_idx, action_idx])

            # Fill callback
            callback.locals = {
                'rewards': reward,
                'dones': done,
                'terminated': [terminated],
                'truncated': [truncated]
            }

            if not callback.on_step():
                break

            state = next_state if not done else self.env.reset()
            self.num_timesteps += 1

            if done:
                self.logger.record("train/episode_reward", episode_reward)
                self.logger.record("train/epsilon", self.epsilon)
                episode_reward = 0.0
                self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)

        callback.on_training_end()
        return self

    def save(self, path: str) -> None:
        numpy.savez_compressed(
            f"{path}.npz",
            q_table=self.q_table,
            epsilon=self.epsilon,
            discount_factor=self.discount_factor,
            learning_rate=self.lr_schedule(1.0),
        )

    @classmethod
    def load(cls, path: str, env: GymEnv, **kwargs) -> "QLearning":
        data = numpy.load(f"{path}.npz")
        model = cls(env=env, **kwargs)
        model.q_table = data["q_table"]
        model.epsilon = float(data["epsilon"])
        model.discount_factor = float(data["discount_factor"])
        model.lr_schedule = get_schedule_fn(float(data["learning_rate"]))
        return model

class RolloutLogger(BaseCallback):
    """
    A custom callback for logging the total reward and episode length of each rollout.
    
    :param env_name: Name of the environment.
    :param max_rollout: Maximum number of rollouts to perform.
    :param max_step: Maximum steps per episode.
    :param downsample_trail: Downsample trail parameter.
    :param verbose: Verbosity level: 0 = no output, 1 = info, 2 = debug
    """
    def __init__(self, env_name, max_rollout, max_step, downsample_trail, verbose=0):
        super(RolloutLogger, self).__init__(verbose)
        self.env_name = env_name.lower()
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

    def is_success_fail(self, reward, total_reward, terminated):
        if "lake" in self.env_name:
            return int(reward > 1.0e-3)
        elif "lander" in self.env_name:
            return int(total_reward >= 200)
        elif "mountaincar" in self.env_name:
            return terminated
        elif "cliff" in self.env_name:
            return terminated
        else:
            return 0

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
            succ_fail = self.is_success_fail(self.locals['rewards'][0], self.episode_reward, terminated)
            
            if self.current_rollout < self.downsample_trail:
                self.success_rate_f = (1 - 1 / (self.current_rollout + 1)) * self.success_rate_f + succ_fail / (self.current_rollout + 1)
            else:
                self.success_rate_f = (1 - 1 / self.downsample_trail) * self.success_rate_f + succ_fail / self.downsample_trail

            self.reward_sums.append(self.episode_reward)
            self.step_counts.append(self.episode_length)
            self.success_rate.append(self.success_rate_f)

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


class OnlineRL:
    def __init__(self, env, env_name, model_name, max_trails, max_steps, downsample_trail):
        self.env = env
        self.model_name = model_name
        self.log_callback = RolloutLogger(env_name, max_trails, max_steps, downsample_trail, verbose=1)
        
    def create_model(self):
        model_classes = {'a2c': A2C, 'ppo': PPO, 'dqn': DQN, 'td3': TD3, 'qlearning': QLearning}
        if self.model_name not in model_classes:
            raise ValueError("Unknown policy type: {}".format(self.model_name))
        
        # Create the model with appropriate parameters
        if self.model_name.lower() in ['a2c', 'ppo']:
            self.model = model_classes[self.model_name.lower()](
                policy='MlpPolicy', env=self.env, verbose=1)
        elif self.model_name.lower() == 'dqn':
            self.model = DQN(
                policy='MlpPolicy', env=self.env,
                learning_rate=0.00025, buffer_size=100_000, exploration_fraction=0.1,
                exploration_final_eps=0.01, batch_size=32, tau=0.005,
                train_freq=(4, 'step'), gradient_steps=1, seed=None, optimize_memory_usage=False,
                verbose=1)
        elif self.model_name.lower() == 'td3':
            self.model = TD3(
                policy='MlpPolicy', env=self.env,
                learning_rate=0.0025, buffer_size=1_000_000, train_freq=(1, 'episode'),
                gradient_steps=1, action_noise=None, optimize_memory_usage=False,
                replay_buffer_class=None, replay_buffer_kwargs=None, verbose=1)
        elif self.model_name.lower() == 'qlearning':
            self.model = QLearning(env=self.env, learning_rate=0.7, discount_factor=0.95,
                                   epsilon=1.0, epsilon_decay=0.999, min_epsilon=0.01, verbose=1)

    def __call__(self):
        self.create_model()
        self.model.learn(total_timesteps=int(1e6), callback=self.log_callback)
        return (self.log_callback.reward_sums, 
                self.log_callback.step_counts, 
                self.log_callback.success_rate)

class LoadRLModel:
    def __init__(self, env, env_name, model_name = None, model_path=None):
        self.env = env
        self.env_name = env_name.lower()
        if model_name is None:
            self.model_name = model_name.lower()
        if model_path is None:
            self.model_path = model_path
        self.supported_gym_env = ["lake", "lander", "mountaincar", "pendulum", "cliff"]

    def load(self):
        if self.env_name.find("anymdp") >= 0:
            model = AnyMDPSolverOpt(self.env)
            def benchmark_model(state):
                return model.policy(state)
            self.benchmark_opt_model = benchmark_model
        elif any(self.env_name.find(name) == 0 for name in self.supported_gym_env):
            model_classes = {'dqn': DQN, 'a24': A2C, 'td3': TD3, 'ppo': PPO}
            if self.model_name not in model_classes:
                raise ValueError("Unknown policy type: {}".format())
            model = model_classes[self.model_name].load(f'{self.model_path}/model/{self.model_name}.zip', env=self.env)
            def benchmark_model(state):
                action, _ = model.predict(state)
                return int(action)
            self.benchmark_opt_model = benchmark_model
        else:
            raise ValueError("Unsupported environment:", self.env_name)

    def __call__(self):
        self.load()
        return self.benchmark_opt_model
    
if __name__ == "__main__":
    model_name = "dqn"
    env_name = "lake"
    max_trails = 50
    max_steps = 200
    downsample_trail = 10

    if env_name == "lake":
        import gymnasium
        env = gymnasium.make('FrozenLake-v1', map_name="4x4", is_slippery=True)
    else:
        raise ValueError(f"Unknown environment: {env_name}")

    online_rl = OnlineRL(env, env_name, model_name, max_trails, max_steps, downsample_trail)
    reward_sums, step_counts, success_rate = online_rl()
    
    print("Reward Sums:", reward_sums)
    print("Step Counts:", step_counts)
    print("Success Rate:", success_rate)
