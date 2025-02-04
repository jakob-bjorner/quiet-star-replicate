# just get the thing to work with quiet-star's setup, 
# so the model handles the latent variable generation for now, 
# and later on I can expand some setup to generate variable length latent rationale
# with some Async GRPO setup or some shit. but that is beyond what I am trying to test right now, 
# so delaying my experiments to accomodate for that potential future wouldn't make sense.
import torch
from torch.distributions.normal import Normal
from torch.utils.data import DataLoader
from typing import cast, Any
from abc import ABC, abstractmethod
from collections import OrderedDict

def combine_normals(normal_distributions_list, dim):
    locs = torch.concat([n.loc for n in normal_distributions_list], dim=dim)
    scales = torch.concat([n.scale for n in normal_distributions_list], dim=dim)
    return torch.distributions.Normal(loc=locs, scale=scales)

class RandomizedGRUCell(torch.nn.Module):
    def __init__(self, input_size, hidden_dim, sample_h_tilda:bool, sample_identity:bool, reparameterize:bool):
        '''has assumed batch by sequence input dimension, and assume randomness is in the hidden representation to be sent to the output and the next layer'''
        super().__init__()
        self.hidden_dim = hidden_dim
        self.linear_z = torch.nn.Linear(in_features=hidden_dim + input_size, out_features=hidden_dim)
        self.linear_r = torch.nn.Linear(in_features=hidden_dim + input_size, out_features=hidden_dim)
        self.linear_h_tilda_x = torch.nn.Linear(in_features=input_size, out_features=hidden_dim)
        self.linear_h_tilda_h_m1 = torch.nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.sample_h_tilda = sample_h_tilda
        self.sample_identity = sample_identity
        self.reparameterize = reparameterize

        self.distribution_params = torch.nn.Linear(in_features=hidden_dim, out_features=hidden_dim*2)

    # @torch.compile(options={"triton.cudagraphs": True}, fullgraph=True)
    def forward(self, x_t, h_t_m1, h_tilda_start=None):
        ''' expect [batch, 1, input_dim] for x hx and h_tilda (if provided)'''
        dict_return = dict()
        x_concat_h = torch.concat([x_t, h_t_m1], dim=-1)
        z_t = self.linear_z(x_concat_h).sigmoid()
        r_t = self.linear_r(x_concat_h).sigmoid()
        if h_tilda_start is not None and self.sample_h_tilda:
            h_tilda_t = h_tilda_start
        else:
            h_tilda_t = torch.tanh(self.linear_h_tilda_x(x_t) + r_t * self.linear_h_tilda_h_m1(h_t_m1))

        if self.sample_h_tilda:
            if h_tilda_start is not None:
                h_tilda_t = h_tilda_start.tanh() # This has tanh on it because I only record the h_tilda before the final tanh is applied, so this will need a tanh to make it the right one.
                dict_return['h_tilda_t'] = h_tilda_t # TODO: account for the fact that I won't be returning a distibution or log prob if a starting h_tilda is given.
            else:
                h_tilda_t_base = h_tilda_t 
                # we take in a tanh'd computation, so that the gradient with respect to the h_t_m1 doesn't expload on extreme values. with tanh activation the partial derivative is 1 - tanh^2(x). without it we see gradient explosion.
                # DONE: currently, I am debugging this, to see if the gradient explosion was due to some other source, which I found while investigating rgruh gradient properties.
                # the conclusion was that worse NLL was achieved, and so I revert back to tanh before and after random state sampling. It makes sense that it was a different error, because this problem wasn't nans. This was just instability.
                h_tilda_dist, h_tilda_t, log_prob_h_tilda_t = self.create_dist_sample_get_log_prob(h_tilda_t_base)
                dict_return['log_prob_h_tilda_t'] = log_prob_h_tilda_t
                dict_return['h_tilda_t'] = h_tilda_t # this is recorded without the tanh applied, TODO: rename everything to indicate that this is before tanh is applied, and move tanh
                dict_return['h_tilda_dist'] = h_tilda_dist
                h_tilda_t = (h_tilda_t).tanh() # this to keep the output in the same distirbution as without this sampling procedure.
            h_t = z_t * h_t_m1 + (1 - z_t) * h_tilda_t
        else:
            h_t_base = z_t * h_t_m1 + (1 - z_t) * h_tilda_t
            h_t_dist, h_t, log_prob_h_t = self.create_dist_sample_get_log_prob(h_t_base)
            dict_return['log_prob_h_t'] = log_prob_h_t
            dict_return['h_t_dist'] = h_t_dist

        dict_return["h_t"] = h_t
        return dict_return
    
    def create_dist_sample_get_log_prob(self, representation):
        if self.sample_identity: # this for testing if base GRU implementation is ok.
            return None, representation, torch.zeros_like(representation)
        mean, log_var = self.distribution_params(representation).chunk(2, dim=-1)

        data_type_info = torch.finfo(torch.float32)
        max_log_var_val = 30
        max_mean_val = 1000
        mean = torch.clamp(mean, min=-max_mean_val, max=max_mean_val)
        log_var = torch.clamp(log_var, min=-max_log_var_val, max=max_log_var_val) 
        # do clamping first because the clamp post exp with inf doesn't work the grad turns to nan instead of 0.0
        dist = torch.distributions.Normal(loc=mean, scale=(0.5 * log_var).exp()) # numerical stability.
        if self.reparameterize:
            sample = dist.rsample()
        else:
            sample = dist.sample()
        log_prob_sample = dist.log_prob(sample)
        return dist, sample, log_prob_sample
        
class RandomizedGRU(torch.nn.Module):
    def __init__(self, input_size, hidden_dim, num_layers, sample_h_tilda:bool, sample_identity:bool, reparameterize:bool):
        super().__init__()
        # TODO: support multiple layers
        # TODO: checkout the torch implementation for rnn.py where they dictate a fast implementation where the batch isn't first because the sequence needs to be referenced always in order so memory management is bad if batch is first.
        # NOTE: this implementation doesn't work with any kind of padding...
        self.sample_h_tilda = sample_h_tilda
        self.cell = RandomizedGRUCell(input_size, hidden_dim, sample_h_tilda, sample_identity, reparameterize)
        self.hidden_dim = hidden_dim
    def forward(self, x, hx=None, h_tilda_start=None, return_dict=False):
        if hx is None:
            hx = torch.zeros_like(x[:, [0]][...,[0]]).repeat(1, 1, self.hidden_dim)
        h_t_m1 = hx
        h_ts = []
        h_tilda_ts = []
        log_prob_h_ts = []
        log_prob_h_tilda_ts = []
        h_t_dists = []
        h_tilda_dists = []
        for t in range(x.size(1)):
            x_t = x[:, [t], :]
            dict_return_t = self.cell(x_t, h_t_m1=h_t_m1, h_tilda_start=h_tilda_start if t == 0 and h_tilda_start is not None and self.sample_h_tilda else None)

            if self.sample_h_tilda:
                h_tilda_ts.append(dict_return_t['h_tilda_t'])
                log_prob_h_tilda_ts.append(dict_return_t['log_prob_h_tilda_t'])
                h_tilda_dists.append(dict_return_t['h_tilda_dist'])
            else:
                log_prob_h_ts.append(dict_return_t['log_prob_h_t'])
                h_t_dists.append(dict_return_t['h_t_dist'])
            h_ts.append(dict_return_t['h_t'])
            h_t_m1 = dict_return_t['h_t'] # .clone() this may be necessary if I want to figure out the compile thing??
        h_ts = torch.concat(h_ts, dim=1)
        if return_dict:
            dict_return: dict = {"h_ts": h_ts, "h_n": h_ts[:, [-1], :]}
            if self.sample_h_tilda:
                dict_return["h_tilda_ts"] = torch.concat(h_tilda_ts, dim=1)
                dict_return["log_prob_h_tilda_ts"] = torch.concat(log_prob_h_tilda_ts, dim=1)
                dict_return['h_tilda_dists'] = combine_normals(h_tilda_dists, dim=1)
            else:
                dict_return["log_prob_h_ts"] = torch.concat(log_prob_h_ts, dim=1)
                dict_return['h_t_dists'] = combine_normals(h_t_dists, dim=1)

            return dict_return
        else:
            return h_ts, h_ts[:, [-1], :]


class SampleMixin:
    token_embedding: torch.nn.Module
    model: torch.nn.Module
    lm_head: torch.nn.Module

    def sample(self, input_ids, hx=None, max_gen_length=10, greedy=True, return_logits=False, return_terminal_hidden_states=False):
        '''sample from the model'''
        x_t = self.token_embedding(input_ids)
        x_t, hidden_t = self.model(x_t, hx=hx)
        x_t = self.lm_head(x_t[:, [-1], :])
        logits = [x_t]
        if greedy:
            input_id_t = x_t[:, -1, :].argmax(-1, keepdims=True)
        else:
            input_id_t = torch.multinomial(x_t[:, -1, :].softmax(-1), 1)

        x_t = self.token_embedding(input_id_t)
        generated_ids = [input_id_t]
        for _ in range(max_gen_length - 1):
            x_t, hidden_t = self.model(x_t, hidden_t)
            x_t = self.lm_head(x_t)
            logits.append(x_t[:, [-1], :])
            if greedy:
                input_id_t = x_t[:, -1, :].argmax(-1, keepdims=True)
            else:
                input_id_t = torch.multinomial(x_t[:, -1, :].softmax(-1), 1)
            x_t = self.token_embedding(input_id_t)
            generated_ids.append(input_id_t)
        generated_ids = torch.concat(generated_ids, dim=-1)
        if return_logits:
            logits = torch.concatenate(logits, dim=1)
            if return_terminal_hidden_states:
                return generated_ids, logits, hidden_t
            return generated_ids, logits
        if return_terminal_hidden_states:
            return generated_ids, hidden_t
        return generated_ids
def get_model_type(model_type, hidden_dim, num_layers):
    if model_type == "lstm":
        model = torch.nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
    elif model_type == "gru":
        model = torch.nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
    elif model_type == 'rgruhtilda':
        model = RandomizedGRU(input_size=hidden_dim, hidden_dim=hidden_dim, num_layers=1, sample_h_tilda=True, sample_identity=False, reparameterize=False)
    elif model_type == 'rgruhtildar':
        model = RandomizedGRU(input_size=hidden_dim, hidden_dim=hidden_dim, num_layers=1, sample_h_tilda=True, sample_identity=False, reparameterize=True)
    elif model_type == 'rgruh':
        model = RandomizedGRU(input_size=hidden_dim, hidden_dim=hidden_dim, num_layers=1, sample_h_tilda=False, sample_identity=False, reparameterize=False)
    elif model_type == 'rgruhr':
        model = RandomizedGRU(input_size=hidden_dim, hidden_dim=hidden_dim, num_layers=1, sample_h_tilda=False, sample_identity=False, reparameterize=True)
    elif model_type == 'rgrui':
        model = RandomizedGRU(input_size=hidden_dim, hidden_dim=hidden_dim, num_layers=1, sample_h_tilda=True, sample_identity=True, reparameterize=False)
    else:
        raise NotImplementedError(f"{model_type=}. Is not impelmented")
    return model
class LanguageModelLSTM(torch.nn.Module, SampleMixin):
    def __init__(self, vocab_size, hidden_dim, num_layers, detach_hidden_state=False, model_type='lstm'):
        super().__init__()
        self.token_embedding = torch.nn.Embedding(vocab_size, hidden_dim)
        self.model = get_model_type(model_type, hidden_dim, num_layers)
        self.lm_head = torch.nn.Linear(in_features=hidden_dim, out_features=vocab_size)
        self.detach_hidden_state = detach_hidden_state
        self.detach_hidden_state_linear_layer = torch.nn.Sequential(
            # torch.nn.ReLU(),
            torch.nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        )
    def forward(self, x, hx=None, return_dict=False):
        x = self.token_embedding(x)
        hidden_states, terminal_hidden_states = self.model(x, hx=hx)
        if self.detach_hidden_state:
            x = self.detach_hidden_state_linear_layer(hidden_states).detach() # testing the model performance in same case as quiet-star to get worst case baseline
        x = self.lm_head(hidden_states)
        if return_dict:
            return OrderedDict(logits=x, hidden_states=hidden_states, terminal_hidden_states=terminal_hidden_states)
        return x
# now train based on the lstm model a quiet-star version... simulates having a good reward function for the quiet-star setting. 
# need to create some distirbution over the hidden representations that I can train analogous to the rational.

class QuietStarLSTMModel(torch.nn.Module):
    def __init__(self, hidden_dim, num_layers, reparameterization_trick, model_type):
        super().__init__()
        if model_type == 'lstm':
            self.model = torch.nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        elif model_type == 'gru':
            self.model = torch.nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        else:
            raise NotImplementedError(f"no implementation for the model type: {model_type}")
        self.distribution_params = torch.nn.Linear(in_features=hidden_dim, out_features=2*hidden_dim)
        self.reparameterization_trick = reparameterization_trick
    def forward(self, x, h_c=None):
        dist, (h_n, c_n) = self.get_hidden_dist(x, h_c)
        if self.reparameterization_trick:
            sample = dist.rsample()
        else:
            sample = dist.sample()
        return sample, (h_n, c_n) # doing sample instead of rsample means there is no gradient flowing through this computation.
    # sampling is technically a differentiable decision unless you sample all of the space and compute probability over those elements ??
    def get_hidden_dist(self, x, h_c=None):
        x, (h_n, c_n)= self.model(x, h_c)
        x = self.distribution_params(x)
        mu, log_var = x.chunk(2, dim=-1)
        dist = torch.distributions.Normal(loc=mu, scale=(0.5*log_var).exp())
        return dist, (h_n, c_n)
    
class QuietStarLanguageModel(ABC):
    token_embedding: Any
    model: Any
    lm_head: Any
    def forward(self, x):
        x = self.token_embedding(x)
        x, hidden = self.model(x)
        x = self.lm_head(x)
        return x
    @abstractmethod
    def get_logits_and_hidden_states_and_log_prob_hidden_states_dist(self, x) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        ...


class QuietStarLanguageModelLSTM(torch.nn.Module, SampleMixin, QuietStarLanguageModel):
    def __init__(self, vocab_size, hidden_dim, num_layers, reparameterization_trick=False, model_type='lstm'):
        super().__init__()
        self.token_embedding = torch.nn.Embedding(vocab_size, hidden_dim)
        self.model = QuietStarLSTMModel(hidden_dim, num_layers, reparameterization_trick, model_type)
        self.lm_head = torch.nn.Linear(in_features=hidden_dim, out_features=vocab_size)
    def get_logits_and_hidden_states_and_log_prob_hidden_states_dist(self, x):
        x = self.token_embedding(x)
        dist, _ = cast(QuietStarLSTMModel, self.model).get_hidden_dist(x)
        sampled_hidden_states = dist.sample()
        x = self.lm_head(sampled_hidden_states)
        log_p_hidden_states = dist.log_prob(sampled_hidden_states)
        return x, sampled_hidden_states, log_p_hidden_states, dist
    # support reward model setting, which seeks to get an estimate for the expected log probability of data under the model given the context, and the chosen hidden representation.
    # so need a way to get out hidden_states form the model along with the performance of the model given those hidden states. We can just return the logits, and have the calling 
    # function compute the performance, because performance can be measured differently, so pass the responsibility to caller.

class QuietStarLanguageModelrGRU(SampleMixin, torch.nn.Module, QuietStarLanguageModel):
    def __init__(self, vocab_size, hidden_dim, num_layers, model_type='rgruh'):
        super().__init__()
        assert model_type not in ["rgrui"], "no identity, becuase they don't impart a distirbution over the action space, so this model wouldn't be trainable by quiet-star."
        self.model_type = model_type
        self.token_embedding = torch.nn.Embedding(vocab_size, hidden_dim)
        self.model = cast(RandomizedGRU, get_model_type(model_type, hidden_dim, num_layers))
        self.lm_head = torch.nn.Linear(in_features=hidden_dim, out_features=vocab_size)
    def forward(self, x):
        x = self.token_embedding(x)
        x, hidden = self.model(x)
        x = self.lm_head(x)
        return x
    def get_logits_and_hidden_states_and_log_prob_hidden_states_dist(self, x, x_embed=None, hx=None):
        if x_embed is not None:
            x = x_embed
        else:
            x = self.token_embedding(x)
        dict_returned = self.model(x, hx=hx, return_dict=True)
        output_states = dict_returned['h_ts']
        x = self.lm_head(output_states)
        # dict_return: dict = {"h_ts": h_ts, "h_n": h_ts[:, [-1], :]}
        # if self.sample_h_tilda:
        #     dict_return["h_tilda_ts"] = torch.concat(h_tilda_ts, dim=1)
        #     dict_return["log_prob_h_tilda_ts"] = torch.concat(log_prob_h_tilda_ts, dim=1)
        #     dict_return['h_tilda_dists'] = combine_normals(h_tilda_dists, dim=1)
        # else:
        #     dict_return["log_prob_h_ts"] = torch.concat(log_prob_h_ts, dim=1)
        #     dict_return['h_t_dists'] = combine_normals(h_t_dists, dim=1)
        if cast(RandomizedGRU, self.model).sample_h_tilda:
            dist = dict_returned['h_tilda_dists']
            sampled_states = dict_returned["h_tilda_ts"]
            sampled_states_log_prob = dict_returned["log_prob_h_tilda_ts"]# .sum(-1)
        else:
            # assume the model is the hidden state variety of sampling.
            dist = dict_returned['h_t_dists']
            sampled_states = dict_returned["h_ts"]
            sampled_states_log_prob = dict_returned["log_prob_h_ts"]# .sum(-1)
        return x, sampled_states, sampled_states_log_prob, dist
    

class ReasonerInterpreterGRU(torch.nn.Module):
    def __init__(self, vocab_size, reasoner_interpreter_vocab_size, base_lm_hidden_dim, reasoner_hidden_dim, interpreter_hidden_dim, use_base_lm, use_reasoner, simple_lm_head, weight_groups, share_lm_head, max_reasoning_len, start_of_thought_token):
        super().__init__()
        self.use_base_lm = use_base_lm
        self.use_reasoner = use_reasoner
        self.max_reasoning_len = max_reasoning_len
        self.start_of_thought_token = start_of_thought_token
        self.simple_lm_head = simple_lm_head
        

        self.base_language_model_token_embedding = torch.nn.Embedding(vocab_size, embedding_dim=base_lm_hidden_dim)
        self.base_language_model = torch.nn.GRU(input_size=base_lm_hidden_dim, hidden_size=base_lm_hidden_dim, batch_first=True)

        if weight_groups[1] == weight_groups[0]:
            self.base_reasoner_token_embedding = self.base_language_model_token_embedding
            self.base_reasoner = self.base_language_model        
        else:
            self.base_reasoner_token_embedding = torch.nn.Embedding(vocab_size, reasoner_hidden_dim)
            self.base_reasoner = torch.nn.GRU(input_size=reasoner_hidden_dim, hidden_size=reasoner_hidden_dim, batch_first=True)        

        self.reasoner = LanguageModelLSTM(reasoner_interpreter_vocab_size, reasoner_hidden_dim, 1, model_type='gru')
        if share_lm_head is not None:
            self.reasoner.lm_head = share_lm_head
        if weight_groups[2] == weight_groups[0]:
            self.reasoner.token_embedding = self.base_language_model_token_embedding
            self.reasoner.model = self.base_language_model
        elif weight_groups[2] == weight_groups[1]:
            self.reasoner.token_embedding = self.base_reasoner_token_embedding
            self.reasoner.model = self.base_reasoner
        
        
        if weight_groups[3] == weight_groups[0]:
            self.interpreter_token_embedding = self.base_language_model_token_embedding
            self.interpreter = self.base_language_model
        elif weight_groups[3] == weight_groups[1]:
            self.interpreter_token_embedding = self.base_reasoner_token_embedding
            self.interpreter = self.base_reasoner
        elif weight_groups[3] == weight_groups[2]:
            self.interpreter_token_embedding = self.reasoner.token_embedding
            self.interpreter = self.reasoner.model
        else:
            self.interpreter_token_embedding = torch.nn.Embedding(reasoner_interpreter_vocab_size, interpreter_hidden_dim)
            self.interpreter = torch.nn.GRU(input_size=interpreter_hidden_dim, hidden_size=interpreter_hidden_dim, batch_first=True)
        self.downproject_interpreter_rep = torch.nn.Linear(self.interpreter.hidden_size, base_lm_hidden_dim)


    def forward(self, x, hx=None, return_dict=False):
        base_lm_embedding = self.base_language_model_token_embedding(x)
        base_reasoner_embedding = self.base_reasoner_token_embedding(x)
        if hx is not None:
            base_lm_hidden_rep, base_lm_final_hidden_rep = self.base_language_model(base_lm_embedding, hx[0])
            base_reasoner_hidden_rep, base_reasoner_final_hidden_rep = self.base_reasoner(base_reasoner_embedding, hx[1])
        else:
            base_lm_hidden_rep, base_lm_final_hidden_rep = self.base_language_model(base_lm_embedding)
            base_reasoner_hidden_rep, base_reasoner_final_hidden_rep = self.base_reasoner(base_reasoner_embedding)
        
        hidden_rep_seed_reasoning = base_reasoner_hidden_rep.reshape(-1, base_reasoner_hidden_rep.size(-1))
        start_and_end_prompt_tokens = torch.full_like(hidden_rep_seed_reasoning[..., 0], fill_value=self.start_of_thought_token).long()
        reasonings, reasoning_logits = self.reasoner.sample(start_and_end_prompt_tokens[:,None], hidden_rep_seed_reasoning[None,:,:], max_gen_length=self.max_reasoning_len, greedy=False, return_logits=True)
        prompt_adjusted_reasonings = torch.concat([start_and_end_prompt_tokens[:,None], reasonings, start_and_end_prompt_tokens[:,None]], dim=-1)
        interpreter_embedding = self.interpreter_token_embedding(prompt_adjusted_reasonings)
        _, interpreter_rep = self.interpreter(interpreter_embedding)
        interpreter_rep = interpreter_rep.reshape(*base_lm_hidden_rep.shape[:-1], -1)
        reasonings = reasonings.reshape(*base_lm_hidden_rep.shape[:-1], -1)
        reasoning_logits = reasoning_logits.reshape(*reasonings.shape, -1)
        
        if self.simple_lm_head:
            interpreter_rep_for_modeling = self.downproject_interpreter_rep(interpreter_rep)
        else:
            interpreter_rep_for_modeling = interpreter_rep

        if self.use_base_lm and self.use_reasoner:
            if self.simple_lm_head:
                rep_for_modeling = base_lm_hidden_rep + interpreter_rep_for_modeling
            else:
                rep_for_modeling = torch.concat([base_lm_hidden_rep, interpreter_rep_for_modeling], dim=-1)
        elif self.use_base_lm:
            rep_for_modeling = base_lm_hidden_rep
        else:
            rep_for_modeling = interpreter_rep_for_modeling
        
        output = rep_for_modeling
        if return_dict:
            # one of those weird gather/scatter things. Need to figure this out with the debugging thing.
            distribution = reasoning_logits.softmax(-1)
            log_distribution = reasoning_logits.log_softmax(-1)
            log_prob_hidden = log_distribution.gather(-1, reasonings[...,None])[...,0]
            return dict(output=output, hidden_states=reasonings, log_prob_hidden=log_prob_hidden, dist=distribution)
        return output, (base_lm_final_hidden_rep, base_reasoner_final_hidden_rep)

class DiscreteLM(ABC):
    model: Any # some type thing overwritting whatever.
    lm_head: Any
    def forward(self, x):
        x, _ = self.model(x)
        x = self.lm_head(x)
        return x
    def get_logits_and_hidden_states_and_log_prob_hidden_states_dist(self, x) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        ret_dict = self.model(x, return_dict=True)
        logits = self.lm_head(ret_dict['output'])
        return logits, ret_dict['hidden_states'], ret_dict['log_prob_hidden'], ret_dict['dist']
    
class ReasonerInterpreterModelGRU(DiscreteLM, SampleMixin, torch.nn.Module, QuietStarLanguageModel):
    def __init__(self, vocab_size, reasoner_interpreter_vocab_size, base_lm_hidden_dim, reasoner_hidden_dim, interpreter_hidden_dim, use_base_lm, use_reasoner, simple_lm_head, weight_groups, share_lm_head, max_reasoning_len=10, start_of_thought_token=0):
        super().__init__()
        assert interpreter_hidden_dim == reasoner_hidden_dim, "There is no reason I can think of right now for these two (interpreter_hidden_dim == reasoner_hidden_dim) to be different. This is a catch to make sure I don't make them different on accident."
        assert use_base_lm or use_reasoner, "must use one of either use_reasoner or use_base_lm"
        self.token_embedding = torch.nn.Identity()
        if simple_lm_head:
            self.lm_head = torch.nn.Linear(in_features=base_lm_hidden_dim, out_features=vocab_size)
        else:
            self.lm_head = torch.nn.Sequential(
                torch.nn.Linear(in_features = int(use_reasoner) * interpreter_hidden_dim + int(use_base_lm) * base_lm_hidden_dim, out_features=512),
                torch.nn.GELU(),
                torch.nn.Linear(512, 512),
                torch.nn.GELU(),
                torch.nn.Linear(512, vocab_size)
            )
        self.model = ReasonerInterpreterGRU(vocab_size, reasoner_interpreter_vocab_size, base_lm_hidden_dim, reasoner_hidden_dim, interpreter_hidden_dim, use_base_lm=use_base_lm, use_reasoner=use_reasoner, simple_lm_head=simple_lm_head, weight_groups=weight_groups, share_lm_head=(self.lm_head if share_lm_head else None), max_reasoning_len=max_reasoning_len, start_of_thought_token=start_of_thought_token)


class QuietStarDiscreteGRU(torch.nn.Module):
    def __init__(self, vocab_size, hidden_dim, use_base_lm, use_reasoner, max_reasoning_len, start_of_thought_token, debug_cfg:str):
        super().__init__()
        self.use_base_lm = use_base_lm
        self.use_reasoner = use_reasoner
        self.max_reasoning_len = max_reasoning_len
        self.start_of_thought_token = start_of_thought_token
        self.debug_cfg = debug_cfg
        self.rnn_base_lm_and_base_reasoner_and_reasoner_and_interpreter = LanguageModelLSTM(vocab_size, hidden_dim, 1, model_type='gru')


        self.debug_interpreter_embedding = torch.nn.Embedding(vocab_size, hidden_dim)
        self.debug_interpreter_model = torch.nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.debug_interpreter_lm_head = torch.nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hx=None, return_dict=False):
        rnn_return_dict = self.rnn_base_lm_and_base_reasoner_and_reasoner_and_interpreter.forward(x, hx, return_dict=True) # request return dict to get hidden representations 
        rnn_hidden_states = rnn_return_dict['hidden_states']
        rnn_terminal_hidden_states = rnn_return_dict['terminal_hidden_states']
        base_reasoner_hidden_rep = rnn_hidden_states
        base_lm_hidden_rep = rnn_hidden_states

        hidden_rep_seed_reasoning = base_reasoner_hidden_rep.reshape(-1, base_reasoner_hidden_rep.size(-1))
        start_and_end_prompt_tokens = torch.full_like(hidden_rep_seed_reasoning[..., 0], fill_value=self.start_of_thought_token).long()
        reasonings, reasoning_logits, reasoning_terminal_hidden_states = self.rnn_base_lm_and_base_reasoner_and_reasoner_and_interpreter.sample(start_and_end_prompt_tokens[:,None], hidden_rep_seed_reasoning[None,:,:], max_gen_length=self.max_reasoning_len, greedy=False, return_logits=True, return_terminal_hidden_states=True)
        
        # when use_base_lm and use_reasoner are both true I need to grab the hidden representation produced from sample and feed the model with the sot token again, 
        # which is my proxy for end of thought. And then check what the token produced is
        
        if self.use_base_lm and self.use_reasoner:
            if "seperateInterpreter" in self.debug_cfg:
                prompt_adjusted_reasonings = torch.concat([start_and_end_prompt_tokens[:,None], reasonings, start_and_end_prompt_tokens[:,None]], dim=-1) # don't need to model the text? so technically the end token isn't required for reasoning only.
                _, interpreter_rep = self.debug_interpreter_model(self.debug_interpreter_embedding(prompt_adjusted_reasonings)) 
                interpreter_rep = self.debug_interpreter_lm_head(interpreter_rep)
                # someway of combining the hidden representations from the interpreter and the langauge model?
                raise NotImplementedError("This hasn't been implemented because It might just not be worth doing.")
            else:
                # interpreter_reps = self.rnn_base_lm_and_base_reasoner_and_reasoner_and_interpreter.forward(prompt_adjusted_reasonings) # no hx, because the model should just learn to understand the task based on seeing sot when h0 is zero.
                # interpreter_rep = interpreter_reps[:,-1,:]
                output = self.rnn_base_lm_and_base_reasoner_and_reasoner_and_interpreter.forward(start_and_end_prompt_tokens[:,None], reasoning_terminal_hidden_states)
                output = cast(torch.Tensor, output)
                output = output.reshape(*base_lm_hidden_rep.shape[:-1], -1) # [Batch, seq, vocab]
        elif self.use_base_lm:
            output = rnn_return_dict['logits']
        else:
            # if use_base_lm is false, then I should force the reasoning to be the only thing which is used. I can maintain that the start of thought token is the first thing seen,
            # but instead of taking the hidden representation produced after the terminal of sample I should re run the language model on all generated tokens with zero starting hidden representation.
            prompt_adjusted_reasonings = torch.concat([start_and_end_prompt_tokens[:,None], reasonings, start_and_end_prompt_tokens[:,None]], dim=-1) # don't need to model the text? so technically the end token isn't required for reasoning only.
            if "seperateInterpreter" in self.debug_cfg:
                _, interpreter_rep = self.debug_interpreter_model(self.debug_interpreter_embedding(prompt_adjusted_reasonings)) 
                interpreter_rep = self.debug_interpreter_lm_head(interpreter_rep)
            else:
                interpreter_reps = self.rnn_base_lm_and_base_reasoner_and_reasoner_and_interpreter.forward(prompt_adjusted_reasonings) # no hx, because the model should just learn to understand the task based on seeing sot when h0 is zero.
                interpreter_rep = interpreter_reps[:,-1,:]
            
            
            output = interpreter_rep.reshape(*base_lm_hidden_rep.shape[:-1], -1)

        
        reasonings = reasonings.reshape(*base_lm_hidden_rep.shape[:-1], -1)
        reasoning_logits = reasoning_logits.reshape(*reasonings.shape, -1)
        if return_dict:
            # one of those weird gather/scatter things. Need to figure this out with the debugging thing.
            distribution = reasoning_logits.softmax(-1)
            log_distribution = reasoning_logits.log_softmax(-1)
            log_prob_hidden = log_distribution.gather(-1, reasonings[...,None])[...,0]
            return dict(output=output, hidden_states=reasonings, log_prob_hidden=log_prob_hidden, dist=distribution)
        return output, rnn_terminal_hidden_states


class QuietStarDiscreteGRULM(DiscreteLM, SampleMixin, torch.nn.Module, QuietStarLanguageModel):
    def __init__(self, vocab_size, hidden_dim, use_base_lm, use_reasoner, max_reasoning_len, start_of_thought_token, debug_cfg):
        super().__init__()
        assert use_base_lm or use_reasoner, "must use one of either use_reasoner or use_base_lm"
        self.token_embedding = torch.nn.Identity()
        self.model = QuietStarDiscreteGRU(vocab_size, hidden_dim, use_base_lm, use_reasoner, max_reasoning_len, start_of_thought_token, debug_cfg)
        self.lm_head = torch.nn.Identity()

# class QuietStarDiscreteTransformer(torch.nn.Module):
#     def __init__(self, ):
#         super().__init__()
#         self.model = torch.nn.


def get_quiet_star_loss(model: QuietStarLanguageModel, inputs: torch.Tensor, policy_loss_beta:float=1e6, nll_loss_beta:float=1, trice_samples:int=2, n_tokens_ahead=1, only_positive=False, print_stuff=True):
    # this loss consists of nll for the base model? 
    # well given that we don't parameterize a basemodel,
    # we will ignore that part of the loss for now.
    # getting nll for thoughts is still important tho, for training the lm head.
    n_tokens_ahead = min(inputs.size(1)-1, n_tokens_ahead)
    labels = inputs.clone()
    original_batch_size = inputs.size(0)
    repeated_inputs = inputs.repeat_interleave(trice_samples, dim=0) # every example shows up twice, this for trice!
    repeated_labels = labels.repeat_interleave(trice_samples, dim=0) # every example shows up twice, this for trice!
    repeated_logits, repeated_hidden_states, repeated_log_p_hidden_states, dist = model.get_logits_and_hidden_states_and_log_prob_hidden_states_dist(repeated_inputs)
    # if repeated_log_p_hidden_states.isnan().any():
    #     import ipdb; ipdb.set_trace()
    #     repeated_logits, _, repeated_log_p_hidden_states, dist = model.get_logits_and_hidden_states_and_log_prob_hidden_states_dist(repeated_inputs)
    repeated_shifted_logits = repeated_logits[:, :-1].contiguous()
    repeated_shifted_labels = repeated_labels[:, 1:].contiguous()
    repeated_reward = - torch.nn.CrossEntropyLoss(reduction='none')(repeated_shifted_logits.view(-1, repeated_shifted_logits.size(-1)), repeated_shifted_labels.view(-1))
    repeated_reward = repeated_reward.reshape(*repeated_shifted_labels.shape) # change repeated reward so that I can take the average over my samples
    repeated_reward_n_ahead = torch.clone(repeated_reward)
    for i in range(n_tokens_ahead-1):
        repeated_reward_n_ahead[:, :-(i+1)] += repeated_reward[:, (i+1):]
    # no baseline to regress from, but can still average to create a baseline
    trice_baseline_reward_n_ahead = repeated_reward_n_ahead.reshape(original_batch_size, trice_samples, -1)
    repeated_reward_n_ahead_minus_baseline = (repeated_reward_n_ahead.view(original_batch_size, trice_samples, -1) - trice_baseline_reward_n_ahead.mean(1, keepdim=True)).view(*repeated_reward_n_ahead.shape)
    if only_positive:
        repeated_reward_n_ahead_minus_baseline = repeated_reward_n_ahead_minus_baseline.clamp(min=0)
    quiet_star_policy_loss = - (repeated_reward_n_ahead_minus_baseline.detach() * repeated_log_p_hidden_states[:, :-1].sum(-1)).sum()
    quiet_star_policy_loss_numel = repeated_reward_n_ahead_minus_baseline.numel()
    nll_loss = (-repeated_reward).sum()
    nll_loss_numel = repeated_reward.numel()
    loss = policy_loss_beta * quiet_star_policy_loss + nll_loss_beta * nll_loss # mean allowed because no pad tokens.
    quiet_star_policy_loss = quiet_star_policy_loss.item() / quiet_star_policy_loss_numel
    dist_entropy = -(dist * dist.log()).sum(-1)
    sentence_entropy = dist_entropy.mean(-1)
    nll_loss = nll_loss.item() / nll_loss_numel
    if print_stuff:
        print()
        print(f"{quiet_star_policy_loss = }")
        print(f"{nll_loss = }")
        # avg_std = dist.scale.mean().item()
        # print(f"{avg_std= }")
        # print({n: p.grad.norm().item() for n, p in model.named_parameters() if p.grad is not None})
        # print("dist std min max:", dist.scale.min().item(), dist.scale.mean().item(), dist.scale.max().item())
        # print("hidden_states min max:", repeated_hidden_states.min().item(), repeated_hidden_states.max().item())
        # print("hidden_state minus mean squared max:", (repeated_hidden_states - dist.loc).square().max().item())
        # print("hidden_state minus mean divided by std max:", ((repeated_hidden_states - dist.loc)/ dist.scale).max().item())
        # print("log_prob min max:", repeated_log_p_hidden_states.min().item(), repeated_log_p_hidden_states.max().item())
        # this is for when it is discrete like word sampling.
        # what is prob_hidden * log_prob_hidden an approximation for?
        

        print("distribution entropy min (0), mean, max (4.18):", dist_entropy.min().item(), dist_entropy.mean().item(), dist_entropy.max().item())
        print("sentence entropy per token min, mean, max", sentence_entropy.min().item(), sentence_entropy.mean().item(), sentence_entropy.max().item())
        # if loss.isnan().any() or loss > 10:
        #     import ipdb; ipdb.set_trace()
    assert quiet_star_policy_loss_numel == nll_loss_numel, "need to have the samenumber of elements for the summed loss to have the correct meaning."
    ret_dict = {"loss": loss,
                "numel": quiet_star_policy_loss_numel,
                # "example_hidden_state": ,
                "nll_loss": float(nll_loss),
                "quiet_star_policy_loss": float(quiet_star_policy_loss),
                "reasoning_entropy_min": float(dist_entropy.min()),
                "reasoning_entropy_mean": float(dist_entropy.mean()),
                "reasoning_entropy_max": float(dist_entropy.max()),
                "sentence_reasoning_entropy_min": float(sentence_entropy.min()),
                "sentence_reasoning_entropy_mean": float(sentence_entropy.mean()),
                "sentence_reasoning_entropy_max": float(sentence_entropy.max())
                }
    return ret_dict

def get_nll_from_logits_and_labels(logits, labels):
    shifted_logits = logits[:, :-1].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    return torch.nn.CrossEntropyLoss(reduction='none')(shifted_logits.view(-1, logits.size(-1)), shifted_labels.view(-1))
def get_nll(model, inputs, print_stuff=False):
    labels = inputs.clone()
    logits = model(inputs)
    loss_per_token = get_nll_from_logits_and_labels(logits, labels)
    ret_dict = {"loss": loss_per_token.sum(), "numel": loss_per_token.numel()}
    return ret_dict
def eval_loss_fn(model, get_loss, dataloader):
    losses = []
    with torch.no_grad():
        for d in dataloader:
            loss = get_loss(model, d)
            losses.append(loss)
    return float(sum(losses)) / len(losses)
def get_grad_norm(model):
    params = [p.grad.flatten() for p in model.parameters() if p.grad is not None]
    return torch.concat(params).norm()
def get_model_param_norm(model):
    params = [p.flatten() for p in model.parameters()]
    return torch.concat(params).norm()
