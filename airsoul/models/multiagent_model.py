import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from airsoul.modules import MLPEncoder, ResidualMLPDecoder, CausalBlock
from airsoul.utils import format_cache
from airsoul.utils import weighted_loss
from airsoul.utils import count_parameters
from airsoul.utils import Logger, log_progress, log_debug, log_warn, log_fatal

class MultiAgentModel(nn.Module):
    '''
    Autoregressive Larguage_model
    '''
    def __init__(self, config, verbose=False):
        super().__init__()

        self.word_emb = MLPEncoder(config.word_embeddings)

        self.nvocab = config.vocab_size
        
        self.encoder = CausalBlock(config.causal_block)

        self.output_mapping = ResidualMLPDecoder(config.output_layers)

        if(verbose):
            print("Language Model initialized, total params: {}".format(count_parameters(self)))

    def forward(self, inputs, cache=None, need_cache=True, T=1.0, update_memory=True):
        """
        Input Size:
            inputs:[B, NT], int
        """

        outputs = self.word_emb(inputs)

        outputs, new_cache = self.encoder(outputs, cache=cache, need_cache=need_cache, update_memory=update_memory)

        outputs = self.output_mapping(outputs, T=T)

        return outputs, new_cache
    
    def reset(self):
        self.encoder.reset()

class OmniRL_MultiAgent(MultiAgentModel):
    """
    Input format: List[idx]
    All of the id and value are transform into discrete integer values, forming the vocabular.
    - i.e., we have m obs_id, n action_id -> m + n words
        idx_policy, idx_tag, idx_a_self, idx_reward, idx_end_timestep, idx_reset_env -> 6 words
        t tag_value -> t words 
        value of obs, action, and reward ~ [-10, 10], resolution 0.1 -> 200 words
        Then the total vocabular size = m + n + t + 206
        

    - For one timestep the sequence is arranged as: 
        [ idx_o1, o1, idx_o3, o3, idx_o4, o4, ..., 
        idx_a1, a1, idx_a2, a2, idx_a4, a4, ..., 
        idx_policy, idx_tag, tag, idx_a_self, a_self, idx_reward, reward, idx_end_timestep ]

        World Model (obs) position: idx_o1, idx_o3, idx_o4, ...
        World Model (action of other agent): idx_a1, idx_a2, idx_a4, ...
        Policy Model: idx_policy
        World Model (reward): idx_reward

    """
    def __init__(self, config, verbose=False): 
        super().__init__(config)

        loss_weight = torch.cat((
                torch.linspace(0.0, 1.0, config.context_warmup),
                torch.full((config.max_position - config.context_warmup,), 1.0)), dim=0)
        loss_weight = loss_weight / torch.sum(loss_weight)
        self.register_buffer('loss_weight', loss_weight)

        self.nobs = config.nobs
        self.nother_agent = config.nother_agent

    def find_position(self, inputs):
        """
        inputs: [batch_size, seq_len]
        World Model (obs) position: value in [0, nobs)
        World Model (action of other agent): value in [nobs, nother_agent)
        Policy Model: value == nobs + nother_agent
        World Model (reward): value == nobs + nother_agent + 3; (idx_policy, idx_tag, idx_a_self, idx_reward)
        return world_model_obs_out, world_model_action_out, policy_out, reward_out
        """

        nobs = self.nobs
        nother_agent = self.nother_agent

        world_model_obs_mask = (inputs < nobs)
        world_model_action_mask = (inputs >= nobs) & (inputs < (nobs + nother_agent))
        policy_mask = (inputs == (nobs + nother_agent))
        reward_mask = (inputs == (nobs + nother_agent + 3))

        return ~world_model_obs_mask, ~world_model_action_mask, ~policy_mask, ~reward_mask

    def sequential_loss(self, inputs, label_actions, use_loss_weight=True, update_memory=True, reduce_dim=1):
        """
        label_actions should have same shape as inputs, and replace the idx_policy with label action.
        """
        seq_len = inputs.shape[1]
        ps = self.encoder.position
        pe = ps + seq_len
        if(self.loss_weight.shape[0] < pe):
            log_fatal(f"Loss weight (shape {self.loss_weight.shape[0]}) should be longer" +
                    f" than sequence length {pe}")

        outputs, _ = self.forward(inputs, need_cache=False, update_memory=update_memory)

        world_model_obs_mask, world_model_action_mask, policy_mask, reward_mask = self.find_position(inputs)
    
        
        loss_weight_wm_obs = world_model_obs_mask
        loss_weight_wm_action = world_model_action_mask
        loss_weight_policy = policy_mask
        loss_weight_reward = reward_mask
        if use_loss_weight:
            loss_weight_wm_obs *= self.loss_weight[ps:pe].unsqueeze(0)
            loss_weight_wm_action *= self.loss_weight[ps:pe].unsqueeze(0)
            loss_weight_policy *= self.loss_weight[ps:pe].unsqueeze(0)
            loss_weight_reward *= self.loss_weight[ps:pe].unsqueeze(0)
        
        loss = dict()
        loss["wm_obs"], loss["count_s"] = weighted_loss(outputs, 
                                                        gt=inputs[:, 1:], 
                                                        loss_type="ce",
                                                        loss_wht=loss_weight_wm_obs, 
                                                        reduce_dim=reduce_dim,
                                                        need_cnt=True)
        loss["wm_action"] = weighted_loss(outputs,
                                          gt=inputs[:, 1:],
                                          loss_type="ce",
                                          loss_wht=loss_weight_wm_action,
                                          reduce_dim=reduce_dim,
                                          need_cnt=False)
        loss["policy"] = weighted_loss(outputs,
                                       gt=label_actions,
                                       loss_type="ce",
                                       loss_wht=loss_weight_policy,
                                       reduce_dim=reduce_dim,
                                       need_cnt=False)
        loss["reward"] = weighted_loss(outputs,
                                       gt=inputs[:, 1:],
                                       loss_type="ce",
                                       loss_wht=loss_weight_reward,
                                       reduce_dim=reduce_dim,
                                       need_cnt=False)
        
    def generate(self, inputs, need_numpy=True, single_batch=True, reward_prediction=False):
        """
        0, inputs seqences: 
            [ idx_o1, o1, idx_o3, o3, idx_o4, o4, ..., 
            idx_a1, a1, idx_a2, a2, idx_a4, a4, ..., 
            idx_policy]
        1, Forward with inputs, update memory = False. Get wd_obs, wd_action, and action.
        2, if reward_prediction:
                Form the sequences with action:
                    [ idx_o1, o1, idx_o3, o3, idx_o4, o4, ..., 
                    idx_a1, a1, idx_a2, a2, idx_a4, a4, ..., 
                    idx_policy, idx_tag, tag, idx_a_self, action, idx_reward]
                Forward again and get wd_reward.
                return wd_obs, wd_action, action, wd_reward.
            else:
                reutrn wd_obs, wd_action, action
        """
        pass

    def incontext_learn(self, inputs):
        pass
                                          

if __name__=='__main__':
    pass
