# just get the thing to work with quiet-star's setup, 
# so the model handles the latent variable generation for now, 
# and later on I can expand some setup to generate variable length latent rationale
# with some Async GRPO setup or some shit. but that is beyond what I am trying to test right now, 
# so delaying my experiments to accomodate for that potential future wouldn't make sense.
import torch
from torch.utils.data import DataLoader
from typing import cast, Any
from abc import ABC, abstractmethod

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
        ''' expect [batch, 1, input_dim] for x h_start and h_tilda (if provided)'''
        dict_return = dict()
        x_concat_h = torch.concat([x_t, h_t_m1], dim=-1)
        z_t = self.linear_z(x_concat_h).sigmoid()
        r_t = self.linear_r(x_concat_h).sigmoid()
        if h_tilda_start is not None and self.sample_h_tilda:
            h_tilda_t = h_tilda_start
        else:
            h_tilda_t = torch.tanh(self.linear_h_tilda_x(x_t) + r_t * self.linear_h_tilda_h_m1(h_t_m1))

        if self.sample_h_tilda:
            h_tilda_dist, h_tilda_t, log_prob_h_tilda_t = self.create_dist_sample_get_log_prob(h_tilda_t)
            h_t = z_t * h_t_m1 + (1 - z_t) * h_tilda_t
            dict_return['log_prob_h_tilda_t'] = log_prob_h_tilda_t
            dict_return['h_tilda_t'] = h_tilda_t
            dict_return['h_tilda_dist'] = h_tilda_dist
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
        log_var = torch.clamp(log_var, min=-max_log_var_val, max=max_log_var_val) # do clamping first because the clamp post exp with inf doesn't work the grad turns to nan instead of 0.0
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
        # NOTE: this implementation doesn't work with any kind of padding...
        self.sample_h_tilda = sample_h_tilda
        self.cell = RandomizedGRUCell(input_size, hidden_dim, sample_h_tilda, sample_identity, reparameterize)
        self.hidden_dim = hidden_dim
    def forward(self, x, h_start=None, h_tilda_start=None, return_dict=False):
        if h_start is None:
            h_start = torch.zeros_like(x[:, [0]][...,[0]]).repeat(1, 1, self.hidden_dim)
        h_t_m1 = h_start
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
        
# r_gru = RandomizedGRU(3, 4, 1, True)
# r_gru(torch.zeros((1,5,3)), return_dict=True)
# train_model(get_nll, lambda model: eval_loss_fn(model, get_nll), RandomizedGRU(len(vocab), 100, 1, True), epochs=100)




class SampleMixin:
    token_embedding: torch.nn.Module
    model: torch.nn.Module
    lm_head: torch.nn.Module

    def sample(self, input_ids, max_gen_length=10):
        '''greedy sample from the model'''
        x_t = self.token_embedding(input_ids)
        x_t, hidden_t = self.model(x_t)
        x_t = self.lm_head(x_t)
        input_id_t = x_t[:, [-1], :].argmax(-1)
        # print(input_id_t)
        x_t = self.token_embedding(input_id_t)
        generated_ids = [input_id_t]
        for _ in range(max_gen_length):
            x_t, hidden_t = self.model(x_t, hidden_t)
            x_t = self.lm_head(x_t)
            input_id_t = x_t[:, [-1], :].argmax(-1)
            x_t = self.token_embedding(input_id_t)
            generated_ids.append(input_id_t)
        generated_ids = torch.concat(generated_ids, dim=-1)
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
    def forward(self, x):
        x = self.token_embedding(x)
        x, _ = self.model(x)
        if self.detach_hidden_state:
            x = self.detach_hidden_state_linear_layer(x).detach() # testing the model performance in same case as quiet-star to get worst case baseline
        x = self.lm_head(x)
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
    token_embedding: torch.nn.Embedding
    model: Any
    lm_head: torch.nn.Linear
    def forward(self, x):
        x = self.token_embedding(x)
        x, hidden = self.model(x)
        x = self.lm_head(x)
        return x
    @abstractmethod
    def get_logits_and_hidden_states_and_log_prob_hidden_states_dist(self, x) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.distributions.Normal]:
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
        self.model_type = model_type
        self.token_embedding = torch.nn.Embedding(vocab_size, hidden_dim)
        self.model = cast(RandomizedGRU, get_model_type(model_type, hidden_dim, num_layers))
        self.lm_head = torch.nn.Linear(in_features=hidden_dim, out_features=vocab_size)
    def forward(self, x):
        x = self.token_embedding(x)
        x, hidden = self.model(x)
        x = self.lm_head(x)
        return x
    def get_logits_and_hidden_states_and_log_prob_hidden_states_dist(self, x, x_embed=None):
        if x_embed is not None:
            x = x_embed
        else:
            x = self.token_embedding(x)
        dict_returned = self.model(x, return_dict=True)
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


def get_nll_from_logits_and_labels(logits, labels):
    shifted_logits = logits[:, :-1].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    return torch.nn.CrossEntropyLoss(reduction='mean')(shifted_logits.view(-1, len(vocab)), shifted_labels.view(-1))
def get_nll(model, inputs, device):
    inputs = inputs.to(device)
    labels = inputs.clone()
    logits = model(inputs)
    return get_nll_from_logits_and_labels(logits, labels)
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
