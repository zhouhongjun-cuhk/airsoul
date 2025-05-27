import os
import glob
import sys
import random
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset

class MultiAgentDataSet(Dataset):
    """
    All of the id and value are transform into discrete integer values, forming the vocabular.
    - i.e., we have m obs_id, n agent_id -> m + n words
        idx_policy, idx_tag, idx_a_self, idx_reward, idx_end_timestep, idx_reset_env -> 6 words
        t tag_value -> t words 
        value of obs, action, and reward ~ [-10, 10], resolution 0.1 -> 200 words
        Then the total vocabular size = m + n + t + 206
        
    - For one timestep the sequence is arranged as: 
        [ idx_o1, o1, idx_o3, o3, idx_o4, o4, ..., 
        idx_a1, a1, idx_a2, a2, idx_a4, a4, ..., 
        idx_policy, idx_tag, tag, idx_a_self, a_self, idx_reward, reward, idx_end_timestep ]

    - For obs connection 2D matrix, the line is obs_id, the column is agent_idx,
        value is 1 if the obs is useful for agent.

    - For agent connection 2D matrix, the line is agent_idx, the column is agent_idx,
        value is 1 if the two agent have connection.    

    - The data is save as observations_*.npy, actions_behavior_*.npy, actions_behavior_*.npy, tags_*.npy, rewards.npy
        The sequential data is constructed base on two connection matrix above as each time steps, and each agent can form a sequence.
    """
    def __init__(self, directory, time_step, max_obs, max_agent, max_tag, value_num, resolution, verbose=False):
        if(verbose):
            print("\nInitializing data set from file: %s..." % directory)
        self.file_list = []
        directories = []
        if(isinstance(directory, list)):
            directories.extend(directory)
        else:
            directories.append(directory)
        for d in directories:
            file_list = os.listdir(d)
            self.file_list.extend([os.path.join(d, file) for file in file_list])
            
        self.time_step = time_step
        self.max_obs = max_obs
        self.max_agent = max_agent
        self.max_tag = max_tag
        self.resolution = resolution
        self.min_value = - resolution * value_num / 2
        self.max_value = - self.min_value

        if(verbose):
            print("...finished initializing data set, number of samples: %s\n" % len(self.file_list))

    def __len__(self):
        # TODO multiple sequence in one file
        return len(self.file_list)

    def _init_vocab_offsets(self):
        self.BASE_OFFSET = self.max_obs + self.max_agent
        self.SPECIAL_TOKENS = {
            'idx_policy': 0,
            'idx_tag': 1,
            'idx_a_self': 2,
            'idx_reward': 3,
            'idx_end_timestep': 4,
            'idx_reset_env': 5
        }
        self.TAG_BASE = self.BASE_OFFSET + 6
        self.VALUE_BASE = self.TAG_BASE + self.max_tag

    def vocabularize(self, type, value):
        handler = getattr(self, f'_handle_{type}', None)
        if handler:
            return handler(value)
        raise ValueError(f"Invalid type: {type}")

    def _handle_obs_id(self, value):
        return value

    def _handle_agent_id(self, value):
        return self.max_obs + value

    def _handle_special_token(self, value, token_type):
        return self.BASE_OFFSET + self.SPECIAL_TOKENS[token_type]

    def _handle_idx_policy(self, _):
        return self.BASE_OFFSET
    
    def _handle_idx_tag(self, _):
        return self.BASE_OFFSET + 1

    def _handle_tag_value(self, value):
        return self.TAG_BASE + value

    def _handle_value(self, value):
        clipped = np.clip(value, self.min_value, self.max_value)
        quantized = ((clipped - self.min_value) / self.resolution).astype(int)
        return self.VALUE_BASE + quantized

    def _load_and_process_data(self, path):
        try:
            observation_files = sorted(
                glob.glob(os.path.join(path, 'observations_*.npy')),
                key=lambda x: int(x.split('_')[-1].split('.')[0])
            )
            actions_behavior_files = sorted(
                glob.glob(os.path.join(path, 'actions_behavior_*.npy')),
                key=lambda x: int(x.split('_')[-1].split('.')[0])
            )
            actions_label_files = sorted(
                glob.glob(os.path.join(path, 'actions_behavior_*.npy')),
                key=lambda x: int(x.split('_')[-1].split('.')[0])
            )
            tags_files = sorted(
                glob.glob(os.path.join(path, 'tags_*.npy')),
                key=lambda x: int(x.split('_')[-1].split('.')[0])
            )
            observations = [np.load(f) for f in observation_files]
            actions_behavior = [np.load(f) for f in actions_behavior_files]
            actions_label = [np.load(f) for f in actions_label_files]
            tags = [np.load(f) for f in tags_files]
            rewards = np.load(path + '/rewards.npy')
            resets = np.load(path + '/resets.npy')

            obs_matrix = np.load(path + '/obs_graph.npy')
            agent_matrix = np.load(path + 'agent_graph.npy')
            num_obs = len(observation_files) 
            num_agent = len(actions_behavior_files)

            max_t = min(actions_label[0].shape[0], 
                        rewards.shape[0], 
                        actions_behavior[0].shape[0],
                        observations[0].shape[0],
                        tags[0].shape[0])

            # Shape Check
            if(self.time_step > max_t):
                print(f'[Warning] Load samples from {path} that is shorter ({max_t}) than specified time step ({self.time_step})')
                n_b = 0
                n_e = max_t
            else:
                n_b = 0
                n_e = self.time_step

            sequences = []
            for agent_id in range(num_agent):
                relevant_obs = np.where(obs_matrix[:, agent_id] == 1)[0]
                connected_agents = np.where(agent_matrix[agent_id] == 1)[0]
                agent_sequence = []
                for t in range(n_b, n_e):
                    time_step_seq = []
                    # idx_o1, o1, idx_o3, o3, idx_o4, o4, ...,
                    for obs_id in relevant_obs:
                        obs_value = observations[obs_id][t]
                        time_step_seq.append(self.vocabularize('obs_id', obs_id))
                        time_step_seq.append(self.vocabularize('value', obs_value))
                    # idx_a1, a1, idx_a2, a2, idx_a4, a4, ..., 
                    for other_agent_id in connected_agents:
                        other_agent_value = actions_behavior[other_agent_id][t]
                        time_step_seq.append(self.vocabularize('agent_id', other_agent_value))
                        time_step_seq.append(self.vocabularize('value', other_agent_value))
                    # idx_policy, idx_tag, tag, idx_a_self, a_self, idx_reward, reward, idx_end_timestep
                    time_step_seq.append(self.vocabularize('special_token', 'idx_policy'))
                    time_step_seq.append(self.vocabularize('special_token', 'idx_tag'))
                    time_step_seq.append(self.vocabularize('tag_value', tags[agent_id][t]))
                    time_step_seq.append(self.vocabularize('special_token', 'idx_a_self'))
                    time_step_seq.append(self.vocabularize('value', actions_behavior[agent_id][t]))
                    time_step_seq.append(self.vocabularize('special_token', 'idx_reward'))
                    time_step_seq.append(self.vocabularize('value', rewards[t]))
                    time_step_seq.append(self.vocabularize('special_token', 'idx_end_timestep'))
                    if resets[t]:
                        time_step_seq.append(self.vocabularize('special_token', 'idx_reset_env'))
                    agent_sequence.append(time_step_seq)

                sequences.append(np.concatenate(agent_sequence))                        

            return sequences
        except Exception as e:
            print(f"Unexpected reading error founded when loading {path}: {e}")
            return (None,) * 6