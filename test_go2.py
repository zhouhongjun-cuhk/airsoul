import argparse
import math
import cv2
import imageio
import numpy as np
import gymnasium as gym
import mujoco
import torch
from scipy.spatial.transform import Rotation
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback

# Training parameters
train_timesteps_ = 500_000_000
show_render_ = False
save_video_ = True
num_envs_ = 64

class CustomGo2Env(gym.Env):
    """Custom Go2 environment."""
    def __init__(self, velocity_command):
        super(CustomGo2Env, self).__init__()
        self.model = mujoco.MjModel.from_xml_path("/home/wangfan/fangdong/airsoul/robot/go2/go2.xml")
        self.data = mujoco.MjData(self.model)
        self.obs_dim = self.model.nq + self.model.nv + self.model.nu + 3 + 3 #加入3维的projected_gravity，和3维的velocity_commmand
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
        self.act_dim = self.model.nu
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.act_dim,), dtype=np.float32)
        self.frame_skip = 4
        self.action_scale = 0.5

        # Rewards: lin_vel_xy
        self.velocity_command = velocity_command
        self.std_lin_vel_xy = math.sqrt(0.25)
        # Rewards: ang_vel_z
        self.target_ang_vel_z = 0.0
        self.std_ang_vel_z = math.sqrt(0.25)

        # Penalty: joint_torque
        self.weight_joint_torque = -0.0002
        self.prev_action = np.zeros(self.act_dim)
        self.feet_contact_history = []
        self.last_step_time = 0

    def _get_projected_gravity(self):
        """计算投影重力：将世界坐标系的重力投影到基座坐标系"""
        quat = self.data.qpos[3:7]  # 基座四元数 [qw, qx, qy, qz]
        rot = Rotation.from_quat(quat)  # 转换为旋转对象
        gravity_world = self.model.opt.gravity  # 世界坐标系重力，默认 [0, 0, -9.81]
        projected_gravity = rot.apply(gravity_world)  # 投影到基座坐标系
        return projected_gravity

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3] += np.random.uniform(-0.1, 0.1, size=3)
        rot = Rotation.from_euler('xyz', np.random.uniform(-0.05, 0.05, size=3))
        self.data.qpos[3:7] = rot.as_quat()
        mujoco.mj_forward(self.model, self.data) 
        self.prev_action = np.zeros(self.act_dim)
        self.feet_contact_history = []
        self.last_step_time = 0
        obs = np.concatenate([
            self.data.qvel[:3], 
            self.data.qvel[3:6],
            self._get_projected_gravity(),
            self.velocity_command,
            self.data.qpos[7:],
            self.data.qvel[6:],
            self.prev_action 
            ])
        info = {}
        return obs, info

    def step(self, action):
        scaled_action = np.clip(action, -1.0, 1.0) * self.action_scale
        self.data.ctrl[:] = scaled_action
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        obs = np.concatenate([
            self.data.qvel[:3], 
            self.data.qvel[3:6],
            self._get_projected_gravity(),
            self.velocity_command,
            self.data.qpos[7:],
            self.data.qvel[6:],
            self.prev_action 
            ])

        # Extract base velocity from qvel (world frame)
        base_velocity = self.data.qvel[:3]
        base_ang_velocity = self.data.qvel[3:6]
        joint_torques = self.data.qfrc_actuator[self.model.nu:]

        # Compute exponential velocity tracking reward
        lin_vel_xy_error = np.sum(np.square(self.velocity_command[:2] - base_velocity[:2]))
        lin_vel_xy_reward = np.exp(-lin_vel_xy_error / (self.std_lin_vel_xy ** 2))* 1.0

        # Angular velocity tracking reward (Z-axis, yaw)
        ang_vel_z_error = np.square(self.velocity_command[2] - base_ang_velocity[2])
        ang_vel_z_reward = np.exp(-ang_vel_z_error / (self.std_ang_vel_z ** 2))* 0.5

        # Penalties
        lin_vel_z_penalty = np.sum(np.square(base_velocity[2])) * -2.0
        ang_vel_xy_penalty = np.sum(np.square(base_ang_velocity[:2])) * -0.05
        dof_torques_penalty = np.sum(np.square(joint_torques)) * -1.0e-5
        dof_acc_penalty = np.sum(np.square(self.data.qacc[6:])) * -2.5e-7
        action_rate_penalty = np.sum(np.square(action - self.prev_action)) * -0.01

        # Feet air time (simplified implementation)
        contact_forces = self.data.cfrc_ext
        feet_indices = [i for i in range(self.model.nbody) if "FOOT" in mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)]
        feet_contact = sum(1 for i in feet_indices if np.linalg.norm(contact_forces[i]) > 0.5)
        air_time_reward = feet_contact * 0.125 if feet_contact > 0 else 0.0


        # Compute new reward
        new_reward = (
            lin_vel_xy_reward +
            ang_vel_z_reward +
            lin_vel_z_penalty +
            ang_vel_xy_penalty +
            dof_torques_penalty +
            dof_acc_penalty +
            action_rate_penalty +
            air_time_reward
        )

        # Update previous action
        self.prev_action =  action

        # Set done condition
        rot = Rotation.from_quat(self.data.qpos[3:7])
        euler_angles = rot.as_euler('xyz')
        done = np.abs(euler_angles[:2]).max() > np.pi / 2.0
        if done:
            new_reward -= 1.0

        truncated = False
        info = {"base_velocity": base_velocity, "base_ang_velocity": base_ang_velocity, "euler_angles": euler_angles}
        return obs, new_reward, done, truncated, info
    
    def render(self):
        width, height = 480, 480
        renderer = mujoco.Renderer(self.model, width, height)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)
        cam.lookat[:] = self.data.qpos[:3]
        cam.distance = 2.0
        cam.azimuth = -90.0
        cam.elevation = -20.0
        renderer.update_scene(self.data, camera=cam)
        img = renderer.render()
        return img

    def close(self):
        pass

def make_env(velocity_command):
    """Create and return a Go2 environment."""
    return CustomGo2Env(velocity_command)

def get_device():
    """Determine the best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS (Metal Performance Shaders)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device

class TrainingProgressCallback(BaseCallback):
    """Custom callback to display training progress."""
    def __init__(self, verbose=0):
        super(TrainingProgressCallback, self).__init__(verbose)
        self.total_timesteps = train_timesteps_
        self.num_envs = num_envs_

    def _on_step(self) -> bool:
        if self.n_calls % 1000 == 0:
            progress = (self.n_calls * self.num_envs) / self.total_timesteps * 100
            mean_reward = np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer]) if self.model.ep_info_buffer else 0
            print(f"Step: {self.n_calls * self.num_envs}/{self.total_timesteps} ({progress:.2f}%) | Mean Reward: {mean_reward:.2f}")
        return True

def test_model(model=None, save_video=True, velocity_command=None):
    """Test the trained SAC model and record a video."""
    print("🎬 Starting model testing and video recording...")
    env = make_env(velocity_command)
    device = get_device()
    
    if model is None:
        model = SAC.load("sac_go2_mujoco", device=device)

    obs, info = env.reset()
    frame_list = []
    num_steps = 1000
    reward = 0
    action = np.zeros(12)

    for _ in range(num_steps):
        _, _, _, action, _ = model.predict(obs, velocity_command, "unknow", action, reward) #TODO
        obs, reward, done, truncated, info = env.step(action)
        frame = env.render()
        frame_list.append(frame)

        if show_render_:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("Go2 Simulation", frame_bgr)
            cv2.waitKey(1)

        if done or truncated:
            obs, info = env.reset()

    env.close()
    cv2.destroyAllWindows()

    if save_video:
        video_path = "go2_simulation_sac.mp4"
        imageio.mimsave(video_path, frame_list, fps=30)
        print(f"🎥 Video saved at {video_path}")

def main():
    """Parse command line arguments and execute training or testing."""
    parser = argparse.ArgumentParser(description="Train or test Go2 SAC model")
    parser.add_argument("--test", action="store_true", help="Test the model only")
    parser.add_argument("--show", action="store_true", help="Show real-time rendering")
    parser.add_argument("--command", type=float, nargs=3, default=[0.0, 0.0, 0.0], help="Velocity command as 3D vector (x, y, z)")
    parser.add_argument(
        "--envs",
        type=int,
        default=64,
        help="Number of environments to use for training"
    )
    
    args = parser.parse_args()
    global show_render_, num_envs_
    show_render_ = args.show
    num_envs_ = args.envs
    velocity_command = np.array(args.command)

    if args.test:
        test_model(velocity_command)

if __name__ == "__main__":
    main()