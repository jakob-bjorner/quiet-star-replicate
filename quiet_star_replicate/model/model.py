# just get the thing to work with quiet-star's setup, 
# so the model handles the latent variable generation for now, 
# and later on I can expand some setup to generate variable length latent rationale
# with some Async GRPO setup or some shit. but that is beyond what I am trying to test right now, 
# so delaying my experiments to accomodate for that potential future wouldn't make sense.
import torch
import math
from torch.distributions.normal import Normal
from torch.utils.data import DataLoader
from typing import cast, Any
from abc import ABC, abstractmethod
from collections import OrderedDict
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Model, GenerationConfig, GenerationMixin, PreTrainedModel, GPT2LMHeadModel
from transformers.models.gpt2.modeling_gpt2 import BaseModelOutputWithPastAndCrossAttentions # I can also try importing this from its root of transformers.modeling_outputs, maybe later.
from transformers import GPT2Config
# I need to rewrite the GPT2Model, and then for the GPT2LMHeadModel, I need to change the init function, replacing their GPT2 with my Custom one.
from typing import Tuple, Optional, Union
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask_for_sdpa, _prepare_4d_attention_mask_for_sdpa, _prepare_4d_causal_attention_mask


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

class CustomGPT2Model(GPT2Model):
    '''One line code change to allow for custom 4d attention masks to be passed'''
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        # Attention mask.
        _use_sdpa = self._attn_implementation == "sdpa" and output_attentions is False and head_mask is None
        if self._attn_implementation == "flash_attention_2":
            attention_mask = attention_mask.view(batch_size, -1) if attention_mask is not None else None # moved this one line down to allow for custom attention mask to be passed through.
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif _use_sdpa:
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask=attention_mask,
                input_shape=(batch_size, input_shape[-1]),
                inputs_embeds=inputs_embeds,
                past_key_values_length=past_length,
            )
        else:
            if attention_mask is not None:
                # We create a 3D attention mask from a 2D tensor mask.
                # Sizes are [batch_size, 1, 1, to_seq_length]
                # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
                # this attention mask is more simple than the triangular masking of causal attention
                # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
                attention_mask = attention_mask[:, None, None, :]

                # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
                # masked positions, this operation will create a tensor which is 0.0 for
                # positions we want to attend and the dtype's smallest value for masked positions.
                # Since we are adding it to the raw scores before the softmax, this is
                # effectively the same as removing these entirely.
                attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
                attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            if _use_sdpa:
                encoder_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                    mask=encoder_attention_mask, dtype=inputs_embeds.dtype, tgt_len=input_shape[-1]
                )
            elif not self._attn_implementation == "flash_attention_2":
                encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        for i in range(len(self.h)):
            block, layer_past = self.h[i], past_key_values[i]
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure layer_past is on same device as hidden_states (might not be correct)
                if layer_past is not None:
                    layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                outputs = self._gradient_checkpointing_func(
                    block.__call__,
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                    use_cache,
                    output_attentions,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )

class CustomGPT2LMHeadModel(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = CustomGPT2Model(config)
        self.post_init() # this needs to be tested to make sure there are no issues with tieing the weights to the new model or anything like initialization or somehow holding on to the old weights.
def invert_and_maxfloat_attn_mask(attention_mask, dtype):
    inverted_mask = 1.0 - attention_mask
    attention_mask = inverted_mask.masked_fill(
        inverted_mask.to(torch.bool), torch.finfo(dtype).min
    )
    return attention_mask

class SampleMixinRnnForward:
    token_embedding: torch.nn.Module
    model: torch.nn.Module
    lm_head: torch.nn.Module

    def sample(self, input_ids, hx=None, max_gen_length=10, greedy=True, return_logits=False, return_terminal_hidden_states=False, variable_len=False, start_and_end_of_thought_token=-1, pad_token=-1):
        '''sample from the model'''
        assert max_gen_length > 0, "doesn't support gen len 0"
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
        ended = None
        if variable_len:
            x_t_expanded = x_t
            hidden_t_expanded = hidden_t
            logits_expanded = logits[0]
            input_id_t_expanded = input_id_t
            ended = (input_id_t.squeeze(1) == start_and_end_of_thought_token)
            for _ in range(max_gen_length - 1):
                # only want to run the ones which haven't ended through the model. why are packed sequences sequence length first typically? this is related to processing. grabbing row 0 column 0 through 10 is easy because elements of a row are stored contiguously, so less seeking on the disk.
                if torch.all(ended):
                    break
                x_t, hidden_t = self.model(x_t_expanded[~ended], hidden_t_expanded[0, ~ended][None,:])
                x_t = self.lm_head(x_t)
                logits_expanded = torch.zeros_like(logits_expanded)
                logits_expanded[~ended] = x_t[:, [-1], :]
                logits.append(logits_expanded)
                if greedy:
                    input_id_t = x_t[:, -1, :].argmax(-1, keepdims=True)
                else:
                    input_id_t = torch.multinomial(x_t[:, -1, :].softmax(-1), 1)
                x_t = self.token_embedding(input_id_t)
                # make new ended, and only map back the ones which have not ended to the expanded versions.
                x_t_expanded = torch.zeros_like(x_t_expanded)
                x_t_expanded[~ended] = x_t
                input_id_t_expanded = torch.full_like(input_id_t_expanded, fill_value=pad_token)
                input_id_t_expanded[~ended] = input_id_t # only populate the ids which hadn't already ended. (without forgetting to populate the eot for sequences which just ended.)
                generated_ids.append(input_id_t_expanded)

                new_ended = ended.clone()
                new_ended[~ended] = (input_id_t.squeeze(1) == start_and_end_of_thought_token)
                ended = new_ended # update which ones have ended
        else:
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
        return_tuple = tuple()
        if variable_len:
            # this technically doesn't belong here as it is thought specific, but whatever it makes code cleaner in the Reasoner Interpreter GRU forward pass, and this feature will be used for thoughts alone I believe.
            start = torch.full_like(generated_ids[..., [0]], fill_value=start_and_end_of_thought_token)
            generated_ids = torch.concat([start, generated_ids], dim=-1)

            ended = cast(torch.Tensor, ended)
            if not torch.all(ended): # if this logic is executed only when they have all ended then the logits will be one longer then for the normal case when they are capped the logits will be less long, so my code later on which assumes I can just snipp off the [1:-1] to get reasonings will not work
                # make generated_ids variable length by setting the ids after the first instance of end of thought id to end of thought.
                # for sequences which haven't yet ended, cap with eot, the rest cap with pad.
                cap = torch.full_like(generated_ids[..., [0]], fill_value=pad_token)
                cap[~ended] = start_and_end_of_thought_token
                generated_ids = torch.concat([generated_ids, cap], dim=-1)
        return_tuple = return_tuple + (generated_ids,)
        if return_logits:
            logits = torch.concatenate(logits, dim=1)
            return_tuple = return_tuple + (logits,)
        if return_terminal_hidden_states:
            return_tuple = return_tuple + (hidden_t,)
        if len(return_tuple) == 1:
            return return_tuple[0]
        return return_tuple


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
class LanguageModelLSTM(torch.nn.Module, SampleMixinRnnForward):
    def __init__(self, vocab_size, hidden_dim, num_layers, detach_hidden_state=False, model_type='lstm', linear_lm_head=False):
        super().__init__()
        self.token_embedding = torch.nn.Embedding(vocab_size, hidden_dim)
        self.model = get_model_type(model_type, hidden_dim, num_layers)
        if linear_lm_head:
            self.lm_head = torch.nn.Linear(in_features=hidden_dim, out_features=vocab_size)
        else:
            self.lm_head = torch.nn.Sequential(
                torch.nn.Linear(in_features=hidden_dim, out_features=512),
                torch.nn.GELU(),
                torch.nn.Linear(512, 512),
                torch.nn.GELU(),
                torch.nn.Linear(512, vocab_size)
            )
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


class RNNforward(ABC):
    token_embedding: Any
    model: Any
    lm_head: Any
    def forward(self, x):
        x = self.token_embedding(x)
        x, hidden = self.model(x)
        x = self.lm_head(x)
        return x
class QuietStarLanguageModel(ABC):
    @abstractmethod
    def get_logits_and_hidden_states_and_log_prob_hidden_states_dist(self, x) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        ...


class QuietStarLanguageModelLSTM(torch.nn.Module, SampleMixinRnnForward, QuietStarLanguageModel, RNNforward):
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

class QuietStarLanguageModelrGRU(SampleMixinRnnForward, torch.nn.Module, QuietStarLanguageModel, RNNforward):
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
    def __init__(self, vocab_size, reasoner_interpreter_vocab_size, base_lm_hidden_dim, reasoner_hidden_dim, interpreter_hidden_dim, use_base_lm, use_reasoner, mix_interpeter_base_lm, simple_lm_head, weight_groups, share_lm_head, linear_lm_head, max_reasoning_len, start_of_thought_token, variable_len, pad_token, stage_wise_downproject_interpreter_rep):
        super().__init__()
        self.use_base_lm = use_base_lm
        self.use_reasoner = use_reasoner
        self.max_reasoning_len = max_reasoning_len
        self.start_of_thought_token = start_of_thought_token
        self.simple_lm_head = simple_lm_head
        self.variable_len = variable_len
        self.pad_token = pad_token
        self.reasoner_interpreter_vocab_size = reasoner_interpreter_vocab_size
        self.mix_interpeter_base_lm = mix_interpeter_base_lm
        self.stage_wise_downproject_interpreter_rep = stage_wise_downproject_interpreter_rep
        if self.mix_interpeter_base_lm != 0:
            assert self.simple_lm_head, "The option to mix them only makes sense in simple head setting because with the complex head they are already mixed."

        self.start_of_reasoner_embedding_modifier = torch.nn.Embedding(5, embedding_dim=reasoner_hidden_dim)
        self.start_of_interpreter_embedding_modifier = torch.nn.Embedding(5, embedding_dim=reasoner_hidden_dim)
        self.dropout_index = 0
        self.base_language_model_token_embedding = torch.nn.Embedding(vocab_size, embedding_dim=base_lm_hidden_dim, padding_idx=pad_token)
        self.base_language_model_dropout = torch.nn.Dropout(p=0.0)
        self.base_language_model = torch.nn.GRU(input_size=base_lm_hidden_dim, hidden_size=base_lm_hidden_dim, batch_first=True)
        if weight_groups[1] == weight_groups[0]:
            self.base_reasoner_token_embedding = self.base_language_model_token_embedding
            self.base_reasoner = self.base_language_model        
        else:
            self.base_reasoner_token_embedding = torch.nn.Embedding(vocab_size, reasoner_hidden_dim, padding_idx=pad_token)
            self.base_reasoner = torch.nn.GRU(input_size=reasoner_hidden_dim, hidden_size=reasoner_hidden_dim, batch_first=True)        

        self.reasoner = LanguageModelLSTM(reasoner_interpreter_vocab_size, reasoner_hidden_dim, 1, model_type='gru')
        
        if weight_groups[2] == weight_groups[0]:
            self.reasoner.token_embedding = self.base_language_model_token_embedding
            self.reasoner.model = self.base_language_model
        elif weight_groups[2] == weight_groups[1]:
            self.reasoner.token_embedding = self.base_reasoner_token_embedding
            self.reasoner.model = self.base_reasoner
        if linear_lm_head == False:
            self.reasoner.lm_head = torch.nn.Sequential(
                    torch.nn.Linear(self.reasoner.model.hidden_size, 512),
                    torch.nn.GELU(),
                    torch.nn.Linear(512, 512),
                    torch.nn.GELU(),
                    torch.nn.Linear(512, reasoner_interpreter_vocab_size)
                )
        if share_lm_head is not None:
            if self.reasoner.model.hidden_size == base_lm_hidden_dim:
                self.reasoner.lm_head = share_lm_head
            else:
                self.reasoner.lm_head = torch.nn.Sequential(
                    torch.nn.Linear(self.reasoner.model.hidden_size, base_lm_hidden_dim),
                    share_lm_head
                )
        
        
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
            self.interpreter_token_embedding = torch.nn.Embedding(reasoner_interpreter_vocab_size, interpreter_hidden_dim, padding_idx=pad_token)
            self.interpreter = torch.nn.GRU(input_size=interpreter_hidden_dim, hidden_size=interpreter_hidden_dim, batch_first=True)
        self.downproject_interpreter_rep = torch.nn.Linear(self.interpreter.hidden_size, base_lm_hidden_dim)
        assert not stage_wise_downproject_interpreter_rep, "this isn't implemented. Didn't work for solving the problem I wanted to solve of large reasoner dimensions not working."
        # if stage_wise_downproject_interpreter_rep:
        #     log2interpreterrep = math.log2(self.interpreter.hidden_size)
        #     log2basehiddendim = math.log2(base_lm_hidden_dim)
        #     assert abs(log2interpreterrep - int(log2interpreterrep)) < 0.001 and \
        #            abs(log2basehiddendim - int(log2basehiddendim)) < 0.001, "only powers of 2."

            # self.downproject_interpreter_rep = torch.nn.Sequential(
            #     *(layer for layers in 
            #       ((torch.nn.GELU(), torch.nn.Linear(2**k, 2 ** (k-1))) for k in range(int(log2interpreterrep), int(log2basehiddendim), -1)) 
            #       for layer in layers)
            # )
            # self.downproject_interpreter_rep = torch.nn.Sequential(
            #     torch.nn.Linear(256, 64), # hardcode test.
            #     torch.nn.GELU(),
            #     torch.nn.Linear(64,32),
            # )


        self.mixer_norm_interpeter = torch.nn.LayerNorm(self.interpreter.hidden_size)
        self.mixer_norm_base_lm = torch.nn.LayerNorm(self.base_language_model.hidden_size)
        if self.mix_interpeter_base_lm == 1:
            self.mixer = torch.nn.Sequential( # I don't like the gating idea. I just think that it is missing too much. Will just do an MLP.
                torch.nn.Linear(self.interpreter.hidden_size + self.base_language_model.hidden_size, 512),
                torch.nn.GELU(),
                torch.nn.Linear(512, 1),
            )
            self.mixer[-1].weight.data.zero_()
            self.mixer[-1].bias.data.fill_(0) 
        elif self.mix_interpeter_base_lm == 2:
            self.mixer = torch.nn.Sequential( # I don't like the gating idea. I just think that it is missing too much. Will just do an MLP.
                torch.nn.Linear(self.interpreter.hidden_size + self.base_language_model.hidden_size, 512),
                torch.nn.GELU(),
                torch.nn.Linear(512, 1),
                torch.nn.Sigmoid(),
            )
            self.mixer[-2].weight.data.zero_()
            # -2 makes the sigmoid ~0.11, so derivative will be 0.01
            # -4 makes ~0.017, so derivative 0.000289 (just making sure it looks like it will be numerically stable enough?)
            self.mixer[-2].bias.data.fill_(0) 
        else:
            assert self.mix_interpeter_base_lm == 0, 'only 0, 1, and 2 implemented for mixing head.'
        self.ema_mixing_coef = 0.0
        self.ema_factor = torch.nn.Buffer(torch.tensor([1.0]))
        self.combining_base_lm_and_interpreter_residual = torch.nn.Sequential( # I don't like the gating idea. I just think that it is missing too much. Will just do an MLP.
            torch.nn.Linear(self.interpreter.hidden_size + self.base_language_model.hidden_size, 512),
            torch.nn.GELU(),
            torch.nn.Linear(512, self.base_language_model.hidden_size),
        )

        self.reasoning_len_ema_factor = torch.nn.Buffer(torch.tensor([1.0]))
        self.reasoning_len_mixing = 10 
        # self.additive_bias_for_base_lm_hidden_rep = torch.nn.Parameter(torch.zeros((self.base_language_model.hidden_size,)))
        # self.gating_for_additive_bias_for_base_lm_rep = torch.nn.Sequential( 
        #     torch.nn.Linear(self.base_language_model.hidden_size, 512),
        #     torch.nn.GELU(),
        #     torch.nn.Linear(512, 1),
        #     # torch.nn.Sigmoid(),
        # )
        # mixing coefficient should be dependant on the words, but I will just do one that is the same for every word.
        # or I could do ema weights for the mixer. (this would be odd?) typically an EMA net has parameters which are the ema of another network which is learning with typical SGD.
    def forward(self, x, hx=None, return_dict=False):
        if self.training is False:
            self.dropout_index = 0
        base_lm_embedding = self.base_language_model_token_embedding(x)
        base_reasoner_embedding = self.base_reasoner_token_embedding(x)
        if hx is not None:
            base_lm_hidden_rep, base_lm_final_hidden_rep = self.base_language_model(base_lm_embedding, hx[0])
            base_reasoner_hidden_rep, base_reasoner_final_hidden_rep = self.base_reasoner(base_reasoner_embedding, hx[1])
        else:
            base_lm_hidden_rep, base_lm_final_hidden_rep = self.base_language_model(base_lm_embedding)
            base_reasoner_hidden_rep, base_reasoner_final_hidden_rep = self.base_reasoner(base_reasoner_embedding)
        base_lm_hidden_rep = self.base_language_model_dropout(base_lm_hidden_rep)
        base_lm_final_hidden_rep = self.base_language_model_dropout(base_lm_final_hidden_rep) # this just in case I want to sample from a training model, which in itself is not something I think I ever do, so I should raise an error lol?? This declares a dependancy on GRU and one layer
        # this is some shit I am going to try....
        # well, more than just an additive bias. There is a weighting between the bias and the base representation determined by a 2 layer net kind of like a 
        # well, at least I know what the thing is doing now. I will convert the code back and understand this as a great reason to use hidden dim = 1 as a strong baseline.
        # temp_gating_coef = self.gating_for_additive_bias_for_base_lm_rep(base_lm_hidden_rep)
        # # print(temp_gating_coef)
        # base_lm_hidden_rep = base_lm_hidden_rep * (1-temp_gating_coef) + self.additive_bias_for_base_lm_hidden_rep * temp_gating_coef

        hidden_rep_seed_reasoning = base_reasoner_hidden_rep.reshape(-1, base_reasoner_hidden_rep.size(-1))
        start_and_end_prompt_tokens = torch.full_like(hidden_rep_seed_reasoning[..., 0], fill_value=self.start_of_thought_token).long()
        if self.variable_len and hasattr(self, 'reasoning_len_mixing') and self.reasoning_len_mixing != self.max_reasoning_len and self.max_reasoning_len != 0:
            temp_max_reasoning_len = (self.reasoning_len_mixing * self.reasoning_len_ema_factor + self.max_reasoning_len * (1-self.reasoning_len_ema_factor)).round().int().item()
            if self.training:
                assert x.size(0) % 10 == 0, "this only works for trice 10 right now. I could fix this, but later. if this works."
                self.reasoning_len_ema_factor = self.reasoning_len_ema_factor * 0.99996 ** (x.size(0) / 10) # scaled so 128 would yeild 0.995 divide by trice samples unfortunately need to pass this in if I want to do it correctly.
            print(f"{temp_max_reasoning_len=}")
        else:
            temp_max_reasoning_len = self.max_reasoning_len

        if self.max_reasoning_len == 0:
            reasonings = start_and_end_prompt_tokens.new_full((hidden_rep_seed_reasoning.size(0), 0), fill_value=0)
            reasoning_logits = base_reasoner_hidden_rep.new_full((hidden_rep_seed_reasoning.size(0), 0, self.reasoner_interpreter_vocab_size), fill_value=0)
        else:
            hidden_rep_seed_reasoning = hidden_rep_seed_reasoning + self.start_of_reasoner_embedding_modifier(torch.full_like(start_and_end_prompt_tokens, self.dropout_index))
            reasonings, reasoning_logits = self.reasoner.sample(start_and_end_prompt_tokens[:,None], hidden_rep_seed_reasoning[None,:,:], max_gen_length=temp_max_reasoning_len, greedy=False, return_logits=True, variable_len=self.variable_len, start_and_end_of_thought_token=self.start_of_thought_token, pad_token=self.pad_token)
        # the reasonings have the end of thought token appended and prepended, and have pad tokens after the eot till max_reasoning_len + 2
        if not self.variable_len or self.max_reasoning_len == 0:
            prompt_adjusted_reasonings = torch.concat([start_and_end_prompt_tokens[:,None], reasonings, start_and_end_prompt_tokens[:,None]], dim=-1)
        else:
            prompt_adjusted_reasonings = reasonings
            reasonings = reasonings[:, 1:temp_max_reasoning_len+1] # remove sot and remove cap if it is added which would be pad and eot, but won't be trained over. unless the cap is less than the max_reasoning_len which can be the case if we explicitly want the model to train on producing the end of thought tokens.
        interpreter_embedding = self.interpreter_token_embedding(prompt_adjusted_reasonings)
        lens = None
        if self.variable_len and self.max_reasoning_len != 0:
            lens = torch.full_like(prompt_adjusted_reasonings[:, 0], fill_value=prompt_adjusted_reasonings.size(-1))
            rows_with_ends, end_indices = torch.where(reasonings == self.start_of_thought_token)
            lens[rows_with_ends] = end_indices + 1 + 1 # plus the extra one for start of thought
            interpreter_embedding = pack_padded_sequence(interpreter_embedding, lens.cpu(), batch_first=True, enforce_sorted=False)
        start_of_interpreter_embedding_modifier = self.start_of_interpreter_embedding_modifier(torch.full_like(start_and_end_prompt_tokens, self.dropout_index))
        _, interpreter_rep = self.interpreter(interpreter_embedding, start_of_interpreter_embedding_modifier[None,...]) # dimension should be ??
        interpreter_rep = interpreter_rep.reshape(*base_lm_hidden_rep.shape[:-1], -1)
        reasonings = reasonings.reshape(*base_lm_hidden_rep.shape[:-1], -1)
        reasoning_logits = reasoning_logits.reshape(*reasonings.shape, reasoning_logits.size(-1))

        if self.simple_lm_head or self.stage_wise_downproject_interpreter_rep:
            interpreter_rep_for_modeling = self.downproject_interpreter_rep(interpreter_rep)
        else:
            interpreter_rep_for_modeling = interpreter_rep

        if self.use_base_lm and self.use_reasoner:
            if self.simple_lm_head:
                if bool(self.mix_interpeter_base_lm): # check if its not 0
                    interpreter_rep_for_modeling_normed = self.mixer_norm_interpeter(interpreter_rep) # don't use down projected because too many MLPs can be bad.
                    base_lm_hidden_rep_normed = self.mixer_norm_base_lm(base_lm_hidden_rep)
                    mixing_coef = self.mixer(torch.concat([base_lm_hidden_rep_normed, interpreter_rep_for_modeling_normed], dim=-1))
                    combined_base_lm_and_intepreter_residual = self.combining_base_lm_and_interpreter_residual(torch.concat([base_lm_hidden_rep_normed, interpreter_rep_for_modeling_normed], dim=-1))
                    interpreter_rep_for_modeling = combined_base_lm_and_intepreter_residual
                    if self.mix_interpeter_base_lm == 2:
                        # mixing_coef = mixing_coef.clamp(min=0.01, max=0.99) # clamping seems to need to be 0.5 or higher to get the model to not fully ignore its reasoning.
                        # if they are too small they may be clamped, and become dead. This is a worry early on.
                        # why ?? maybe some learning signal is passed through, and if the signal is too low, then the reasoning won't learn fast enough to not be ignored.
                        # shifting it down, but now making it so that the mixing cant change so fast, because the mixing seemed to be going up, but then when too far.
                        mixing_coef2 = self.ema_mixing_coef * self.ema_factor + (1-self.ema_factor) * mixing_coef
                        print("sigmoid ema:mean live:mean max",mixing_coef2.mean(), mixing_coef.mean(), mixing_coef.max(), mixing_coef.view(-1))# would be interesting to document the mean of this to see how much ignoring is going on.
                        with torch.no_grad():
                            if self.training:
                                # self.ema_mixing_coef = self.ema_mixing_coef * (0.95) + 0.05 * mixing_coef.mean()
                                # self.ema_factor = self.ema_factor * 0.995 # ema factor degrades slowly over time giving control from ema_mixing coef to the model's mixing head.
                                assert mixing_coef.size(0) % 10 == 0, "this only works for trice 10 right now. I could fix this, but later. if this works."
                                self.ema_factor = self.ema_factor * 0.99996 ** (mixing_coef.size(0) / 10) # scaled so 128 would yeild 0.995 divide by trice samples unfortunately need to pass this in if I want to do it correctly.
                        mixing_coef = mixing_coef2
                    elif self.mix_interpeter_base_lm == 1:
                        print("mixinghead min mean max", mixing_coef.min(), mixing_coef.mean(), mixing_coef.max(), mixing_coef.view(-1))
                    rep_for_modeling = base_lm_hidden_rep * (1 - mixing_coef) + interpreter_rep_for_modeling * (mixing_coef)
                else:
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
            # if # checking if length 0 potentially to adjust the logits for seemless operation later. 
            distribution = reasoning_logits.softmax(-1)
            log_distribution = reasoning_logits.log_softmax(-1)
            # if variable_len, then distribution should only be calculateed on non pad tokens.
            log_prob_hidden = log_distribution.gather(-1, reasonings[...,None])[...,0]
            if self.variable_len and self.max_reasoning_len != 0:
                lens = cast(torch.Tensor, lens)
                # can't just do == pad because the model could generate pad tokens.?? 
                # why tho? Why would 0 vector be chosen for pad. 
                # This would be bad as it is set as the output linear layer as well right???
                # this design choice seems a bit concerning because of Hewitt's adding embeddings blog idea right? I will ignore it tho lol.
                pad_mask = (torch.arange(reasonings.size(-1), device=reasonings.device)[None,None,:] >= (lens.reshape(reasonings.shape[:-1])[...,None] - 1))
                # need to determine if pad mask is being calculated correctly for the logits so they don't ignore the eot token.
                log_prob_hidden[pad_mask] = 0 # the pad tokens are deterministic and also shouldn't try to recieve gradients.
                onehot_pad = distribution.new_zeros(distribution.size(-1)) # , dtype=bool
                onehot_pad[self.pad_token] = 1
                # distribution_clone = distribution
                # distribution[pad_mask[...,None] & onehot_pad[None,None,None,:]] = 1 
                # distribution[pad_mask[...,None] & ~onehot_pad[None,None,None,:]] = 0 
                with torch.no_grad():
                    distribution.data[pad_mask] = onehot_pad # get around inplace checking for gradient correctness.
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

class ReasonerInterpreterGRULM(DiscreteLM, SampleMixinRnnForward, torch.nn.Module, QuietStarLanguageModel, RNNforward):
    def __init__(self, vocab_size, reasoner_interpreter_vocab_size, base_lm_hidden_dim, reasoner_hidden_dim, interpreter_hidden_dim, use_base_lm, use_reasoner, mix_interpeter_base_lm, simple_lm_head, linear_lm_head, weight_groups, share_lm_head, variable_len, pad_token, stage_wise_downproject_interpreter_rep, infer_pretrained_base, infer_pretrained_reasoner, infer_pretrained_base_reasoner, parameter_groups,max_reasoning_len=10, start_of_thought_token=0):
        super().__init__()
        assert interpreter_hidden_dim == reasoner_hidden_dim, "There is no reason I can think of right now for these two (interpreter_hidden_dim == reasoner_hidden_dim) to be different. This is a catch to make sure I don't make them different on accident."
        assert use_base_lm or use_reasoner, "must use one of either use_reasoner or use_base_lm"
        self.variable_len = variable_len
        self.pad_token = pad_token
        self.start_of_thought_token = start_of_thought_token
        self.token_embedding = torch.nn.Identity()
        self.mix_interpeter_base_lm = mix_interpeter_base_lm
        self.infer_pretrained_base = infer_pretrained_base
        self.infer_pretrained_reasoner = infer_pretrained_reasoner
        self.parameter_groups = parameter_groups
        self.interpreter_hidden_dim = interpreter_hidden_dim
        self.base_lm_hidden_dim  = base_lm_hidden_dim
        assert parameter_groups == 0, "Other groups are not implemented. create a function called get_param_groups"
        create_lm_head = None
        effective_input = 0
        additive_input = 32
        if simple_lm_head: 
            effective_input = base_lm_hidden_dim + additive_input
            if linear_lm_head:
                create_lm_head = lambda: torch.nn.Linear(in_features=effective_input, out_features=vocab_size)
            else:
                create_lm_head = lambda: torch.nn.Sequential(
                    torch.nn.Linear(in_features=effective_input, out_features=512),
                    torch.nn.GELU(),
                    torch.nn.Linear(512, 512),
                    torch.nn.GELU(),
                    torch.nn.Linear(512, vocab_size)
                )
        else:
            effective_interpreter_dim = base_lm_hidden_dim if stage_wise_downproject_interpreter_rep else interpreter_hidden_dim
            effective_input = int(use_reasoner) * effective_interpreter_dim + int(use_base_lm) * base_lm_hidden_dim
            effective_input += additive_input
            if linear_lm_head:
                create_lm_head = lambda: torch.nn.Linear(in_features = effective_input, out_features=vocab_size)
            else:
                create_lm_head = lambda: torch.nn.Sequential(
                    torch.nn.Linear(in_features = effective_input, out_features=512),
                    torch.nn.GELU(),
                    torch.nn.Linear(512, 512),
                    torch.nn.GELU(),
                    torch.nn.Linear(512, vocab_size)
                )
        self.lm_head_weights = create_lm_head()
        self.lm_head_modifiers = torch.nn.Embedding(5, additive_input) # this is a strange thing, that i want to do to test dropout, and don't want to make the code messier
        self.model = ReasonerInterpreterGRU(vocab_size, reasoner_interpreter_vocab_size, base_lm_hidden_dim, reasoner_hidden_dim, interpreter_hidden_dim, use_base_lm=use_base_lm, use_reasoner=use_reasoner, mix_interpeter_base_lm=mix_interpeter_base_lm, simple_lm_head=simple_lm_head, weight_groups=weight_groups, share_lm_head=(self.lm_head if share_lm_head else None), linear_lm_head=linear_lm_head, max_reasoning_len=max_reasoning_len, start_of_thought_token=start_of_thought_token, variable_len=variable_len, pad_token=pad_token, stage_wise_downproject_interpreter_rep=stage_wise_downproject_interpreter_rep)
        if bool(infer_pretrained_base):
            infer_pretrained_base = self.get_model_checkpoint(infer_pretrained_base, GRU_type="base lm")
            temp_model = torch.load(infer_pretrained_base)
            self.lm_head_weights.load_state_dict(temp_model.lm_head_weights.state_dict())
            self.lm_head_modifiers.load_state_dict(temp_model.lm_head_modifiers.state_dict())
            self.model.base_language_model_token_embedding.load_state_dict(temp_model.model.base_language_model_token_embedding.state_dict())
            self.model.base_language_model.load_state_dict(temp_model.model.base_language_model.state_dict())
            # for p in self.model.interpreter.parameters():
            #     p.data.zero_()
            # we will assume a path to model.pt right now, and just load everything incl
        if bool(infer_pretrained_reasoner):
            assert infer_pretrained_base is False, "don't know how to reconsile the pretrained base model with the different LM head yet...???"
            infer_pretrained_reasoner = self.get_model_checkpoint(infer_pretrained_reasoner, GRU_type="reasoner")
            temp_model = torch.load(infer_pretrained_reasoner)
            self.model.reasoner.token_embedding.load_state_dict(temp_model.token_embedding.state_dict())
            self.model.reasoner.model.load_state_dict(temp_model.model.state_dict())
            self.model.reasoner.lm_head.load_state_dict(temp_model.lm_head.state_dict())
        if bool(infer_pretrained_base_reasoner):
            assert infer_pretrained_base is False, "don't know how to reconsile the pretrained base model with the different LM head yet...???"
            infer_pretrained_base_reasoner = self.get_model_checkpoint(infer_pretrained_base_reasoner, GRU_type="base reasoner")
            temp_model = torch.load(infer_pretrained_base_reasoner)
            self.model.base_reasoner_token_embedding.load_state_dict(temp_model.token_embedding.state_dict())
            self.model.base_reasoner.load_state_dict(temp_model.model.state_dict())

    def get_model_checkpoint(self, infer_pretrained, GRU_type):
        if infer_pretrained == True:
            if GRU_type == "reasoner":
                assert self.interpreter_hidden_dim == 256
                infer_pretrained = "/nethome/jbjorner3/dev/hallucination-fun/quiet_star_replicate/quiet_star_replicate_runs/+run_modifier=[GRUPretrainReasonerExperiment]_base_lm_hidden_dim=256_info=\[gru\ pretrain\ v0\]_linear_lm_head=False_seed=0_2025-03-08/17-21-43/model.pt"
            elif GRU_type == "base reasoner":
                assert self.interpreter_hidden_dim == 256
                infer_pretrained = "/nethome/jbjorner3/dev/hallucination-fun/quiet_star_replicate/quiet_star_replicate_runs/+run_modifier=[GRUPretrainReasonerExperiment]_base_lm_hidden_dim=256_info=\[gru\ pretrain\ v0\]_linear_lm_head=False_seed=0_2025-03-08/17-21-43/model.pt"
            elif GRU_type == "base lm":
                # interpreter_hidden_base_hidden = (self.interpreter_hidden_dim, self.base_lm_hidden_dim)
                if self.interpreter_hidden_dim == 256:
                    if self.base_lm_hidden_dim == 32:
                        infer_pretrained = "/nethome/jbjorner3/dev/hallucination-fun/quiet_star_replicate/quiet_star_replicate_runs/+run_modifier=[GRUConcatExperiment]_base_lm_hidden_dim=32_info=\[gru base models v0\]_max_reasoning_len=0_reasoner_hidden_dim=256_seed=0_2025-03-07/21-21-45/model.pt"
                    elif self.base_lm_hidden_dim == 2:
                        infer_pretrained = "/nethome/jbjorner3/dev/hallucination-fun/quiet_star_replicate/quiet_star_replicate_runs/+run_modifier=[GRUConcatExperiment]_base_lm_hidden_dim=2_info=\[gru\ base\ models\ v0\]_max_reasoning_len=0_reasoner_hidden_dim=256_seed=0_2025-03-07/22-30-04/model.pt"
                    elif self.base_lm_hidden_dim == 4:
                        infer_pretrained = "/nethome/jbjorner3/dev/hallucination-fun/quiet_star_replicate/quiet_star_replicate_runs/+run_modifier=[GRUConcatExperiment]_base_lm_hidden_dim=4_info=\[gru\ base\ models\ v0\]_max_reasoning_len=0_reasoner_hidden_dim=256_seed=0_2025-03-07/22-30-04/model.pt"
                    elif self.base_lm_hidden_dim == 8:
                        infer_pretrained = "/nethome/jbjorner3/dev/hallucination-fun/quiet_star_replicate/quiet_star_replicate_runs/+run_modifier=[GRUConcatExperiment]_base_lm_hidden_dim=8_info=\[gru\ base\ models\ v0\]_max_reasoning_len=0_reasoner_hidden_dim=256_seed=0_2025-03-07/22-30-04/model.pt"
                    elif self.base_lm_hidden_dim == 16:
                        infer_pretrained = "/nethome/jbjorner3/dev/hallucination-fun/quiet_star_replicate/quiet_star_replicate_runs/+run_modifier=[GRUConcatExperiment]_base_lm_hidden_dim=16_info=\[gru\ base\ models\ v0\]_max_reasoning_len=0_reasoner_hidden_dim=256_seed=0_2025-03-07/22-30-04/model.pt"
                    elif self.base_lm_hidden_dim == 1:
                        infer_pretrained = "/nethome/jbjorner3/dev/hallucination-fun/quiet_star_replicate/quiet_star_replicate_runs/+run_modifier=[GRUConcatExperiment]_base_lm_hidden_dim=1_info=\[gru\ base\ models\ v0\]_max_reasoning_len=0_reasoner_hidden_dim=256_seed=0_2025-03-07/23-47-22/model.pt"
                    elif self.base_lm_hidden_dim == 128:
                        infer_pretrained = "/nethome/jbjorner3/dev/hallucination-fun/quiet_star_replicate/quiet_star_replicate_runs/+run_modifier=[GRUConcatExperiment]_base_lm_hidden_dim=128_info=\[gru\ base\ models\ v0\]_max_reasoning_len=0_reasoner_hidden_dim=256_seed=0_2025-03-08/11-41-23/model.pt"
        assert isinstance(infer_pretrained, str)
        return infer_pretrained

    def lm_head(self, x):
        if self.training is False:
            self.model.dropout_index = 0
        # import ipdb; ipdb.set_trace()
        additive_input_embedding = torch.full_like(x[...,0], fill_value=self.model.dropout_index, dtype=torch.long)
        additive_input_embedding = self.lm_head_modifiers(additive_input_embedding)
        return self.lm_head_weights(torch.concat((x, additive_input_embedding), dim=-1))
        
    def forward(self, x):
        x, _ = self.model(x)
        x = self.lm_head(x)
        return x
    def get_logits_and_hidden_states_and_log_prob_hidden_states_dist(self, x) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        ret_dict = self.model(x, return_dict=True)
        logits = self.lm_head(ret_dict['output'])
        return logits, ret_dict['hidden_states'], ret_dict['log_prob_hidden'], ret_dict['dist']


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
                prompt_adjusted_reasonings = torch.concat([start_and_end_prompt_tokens[:,None], reasonings, start_and_end_prompt_tokens[:,None]], dim=-1) 
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
            # if use_base_lm is false, then I should force the reasoning to be the only thing which is used. 
            # I can maintain that the start of thought token is the first thing seen,
            # but instead of taking the hidden representation produced after the terminal of sample I should re run the language model on all generated tokens with zero starting hidden representation.
            prompt_adjusted_reasonings = torch.concat([start_and_end_prompt_tokens[:,None], reasonings, start_and_end_prompt_tokens[:,None]], dim=-1) 
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


class QuietStarDiscreteGRULM(DiscreteLM, SampleMixinRnnForward, torch.nn.Module, QuietStarLanguageModel, RNNforward):
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


def get_position_ids_from_attn_mask(attention_mask, batch_size):
    '''Specifically for thoughts'''
    new_position_ids = (attention_mask[0,0].long().sum(-1) - 1).repeat(batch_size, 1)
    return new_position_ids
def prepare_output_for_next_input(attention_mask, batch_size):
    '''Assumes new input is one token per batch element, so if that is wrong oops'''
    seq_len = attention_mask.shape[-2]
    new_attention_matrix = torch.concatenate([attention_mask, torch.eye(seq_len, device=attention_mask.device, dtype=attention_mask.dtype)[None,None,...]], dim=-1)
    new_position_ids = get_position_ids_from_attn_mask(new_attention_matrix, batch_size)
    return new_position_ids, new_attention_matrix
# just piecing together a function to use as a sampling thought alternative to asking the generate function to be good at this, which it is just not.
def sample_thoughts(model, input_ids: torch.Tensor, start_of_thought_token_id: int, max_reasoning_len: int, preprocess_position_ids, first_model = None):
    """Sample thoughts for every token in parallel. 
    inputs
        model: PreTrainedModel # this type because implements GenerationMixin, and allows for callable with custom 4d attention mask
        inputs
        start_of_thought_token_id
        max_reasoning_len
        first_model: A model to derive the hidden representation for the base langauage over if none, the 'model' will be used.
    returns
        chosen token indices, 
        log prob of chosen tokens,
        prob distribution over thought tokens,
        kv_cache, # importantly with gradient information kept this is for predicting next tokens/next few tokens.
        last_attention_mask, # this needs to be augmented to serve as the next input.
    if you want to generate multiple thoughts, you need to kind of handle that on your own. This function isn't going to help you at all. 
    I'll wrap it with something like repeat_interleave. The standard.
    Doing simple thing first, copying quiet-star's implementation. 
    Can do other faster fancier memory efficient things later."""
    # bare minimum implementation. 
    # Just anything that is necessary to get the model generating 
    # like quiet-star just to see if it is possible first, 
    # with back prop.
    # skipping logit preprocessor, which could be a more general implementation of forcing terminating and starting thoughts.
    # skipping stopping criteria, which could be used to make some thoughts terminate for interesting reasons.
    # model.generation_config.bos_token_id
    # model.generation_config.eos_token_id
    # model.generation_config.pad_token_id # this isn't defined necessarily.
    # ignoring sending the token ids to tensors on cuda, (Should eventually do this to prevent cpu to gpu communications during inference...)
    # ignoring multiGPU for now.
    batch_size = input_ids.shape[0]
    seq_len = input_ids.shape[1]
    logits_list = []
    chosen_token_indices = []
    attention_mask = torch.tril(torch.ones(seq_len,seq_len, device=input_ids.device, dtype=torch.long))[None, None, ...]  # covers: [batch, n_heads, seq, seq]
    # skipping unfinished sequences accounting.
    position_ids = get_position_ids_from_attn_mask(attention_mask, batch_size) # [batch, seq]
    # don't need to preprocess the first set of position ids, because they just feed in the raw inputs
    # skip implementing an efficient mechanism for multi thought through cuda? I think this would be possible, but I don't know cuda stuff.
    
    def sample_next_tokens_from_logits(logits: torch.Tensor):
        # this doesn't have a gradient. We don't want to do gumbel or anything funny.
        with torch.no_grad(): # just to make doubly sure.
            sampled_next_tokens = logits.softmax(-1).view(-1,logits.size(-1)).multinomial(num_samples=1).reshape(batch_size, seq_len) 
        return sampled_next_tokens
    temp_model = model
    if first_model is not None:
        model = first_model

    past_key_values = None # starts None then gets updated after the first forward pass.
    num_forward_passes = 1 + 1 + max_reasoning_len  # plus one for the first pass on the tokens, plus one for the start of thought token
    for i in range(num_forward_passes): 
        outputs = model(input_ids=input_ids, past_key_values=past_key_values, attention_mask=attention_mask, position_ids=position_ids, use_cache=True)
        past_key_values = outputs.past_key_values
        position_ids, attention_mask = prepare_output_for_next_input(attention_mask, batch_size)
        # need to preprocess these position ids to ensure they have properly encode the fact we are in a thought.
        position_ids = preprocess_position_ids(position_ids, is_in_thought=True) # just using the seq len in batch right now.
        if i == 0: # for first token we force the model to think by feeding the model with the start of thought token.
            input_ids = outputs.logits.new_full((batch_size, seq_len), fill_value=start_of_thought_token_id, dtype=torch.long)
            # don't record this in the logits list because we don't want to make it more likely
            
            # this to remove the logits over the initial tokens, taken from transformers.generation.utils line 3268
            # which we don't use. (not sure if this really matter for us becasue we retain 
            #                      the computation graph for back prop of a much larger graph.)
            del outputs 
            if first_model is not None:
                model = temp_model
        elif i == num_forward_passes - 1: # last iteration
            del outputs # we don't look at these logits. They are eventually replaced with eot.
            ... 
            # we don't want to record anything on the last iteration, 
            # because we don't want to apply gradient on the tokens which are going to be over written 
            # to end of thought token.
        else:
            input_ids = sample_next_tokens_from_logits(outputs.logits) # [batch, seq_len]

            logits_list.append(outputs.logits) # batch, seq_len, voc
            chosen_token_indices.append(input_ids) 
    if max_reasoning_len != 0:
        logits = torch.stack(logits_list, dim=2) # batch, seq_len, thought_len, vocab_size
        chosen_token_indices = torch.stack(chosen_token_indices, dim=-1) # batch, seq_len, thought_len

        log_prob_tokens = logits.log_softmax(-1)
        log_prob_chosen_tokens = log_prob_tokens.gather(dim=-1, index=chosen_token_indices.unsqueeze(-1)).squeeze(-1)
        prob_dist_over_thought_tokens = logits.softmax(-1)
    else:
        chosen_token_indices = input_ids.new_zeros((batch_size, 1, 1))
        log_prob_chosen_tokens = torch.zeros((batch_size, 1, 1), dtype=torch.float32, device=input_ids.device)
        prob_dist_over_thought_tokens = torch.ones((batch_size, 1, 1, 1), dtype=torch.float32, device=input_ids.device) # i don't know vocab size, so just set it to 1.

    
    kv_cache = past_key_values
    last_attention_mask = attention_mask
    return chosen_token_indices, log_prob_chosen_tokens, prob_dist_over_thought_tokens, (kv_cache, last_attention_mask)

def prepare_shifted_inputs_for_thought_sampling_and_next_token_prediction_for_loss(inputs, num_tokens_ahead):
    # assuming there is already a start of thought token appended.
    
    # remove some of the tokens at the end which wouldn't be able to get the full loss because they don't have enough space for the num_tokens_ahead for token prediction.
    # note this isn't good for evaluation or generation, because you might want to generate all the way to the end of the sentence to predict the next token.
    inputs_for_thought_sampling = inputs[:, :-num_tokens_ahead].contiguous() # essentially, only need these logits equivalent

    inputs_for_next_token_prediction = inputs[:, 1:].contiguous()
    return inputs_for_thought_sampling, inputs_for_next_token_prediction


def use_thoughts_to_get_loss_over_next_tokens(model, labels: torch.Tensor, vocab_size, thought_augmented_key_values, attention_mask_for_thoughts_and_next_token: torch.Tensor, end_of_thought_token_id: int, num_tokens_ahead: int, base_inputs, use_residual, preprocess_position_ids, add_last_context_token, add_surogate_loss_to_last_context_token, increment_pos_id_for_last_context_token):
    batch_size = labels.size(0)
    input_seq_len = attention_mask_for_thoughts_and_next_token.size(-2) # knowable because of the attention_mask and thought_augmented_key_values
    input_ids = attention_mask_for_thoughts_and_next_token.new_full((batch_size, input_seq_len), fill_value=end_of_thought_token_id) # should be [batch, input_seq_len]
    past_key_values = thought_augmented_key_values
    attention_mask = attention_mask_for_thoughts_and_next_token
    position_ids = get_position_ids_from_attn_mask(attention_mask, batch_size)
    position_ids = preprocess_position_ids(position_ids, is_in_thought=True)
    loss_over_last_context_token = None

    if use_residual:
        base_output_logits = model(base_inputs).logits


    if add_last_context_token:
        # should run the input_ids for end of thought through and then prep the model to feed in the last_context token.
        outputs = model(input_ids=input_ids, past_key_values=past_key_values, attention_mask=attention_mask, position_ids=position_ids, use_cache=True)
        past_key_values = outputs.past_key_values
        position_ids, attention_mask = prepare_output_for_next_input(attention_mask, batch_size)
        position_ids = preprocess_position_ids(position_ids, is_in_thought=False) - 1 + int(increment_pos_id_for_last_context_token) # the minus one because we want the thought to have the same position id as it did before. I should try just making it without minus 1 maybe its not so bad...?
        input_ids = base_inputs[:, : input_seq_len]

        if add_surogate_loss_to_last_context_token:
            if use_residual:
                raise NotImplementedError("you cannot produce logits over the first token because for bos you have nothing to condition on.")
            loss_over_last_context_token = torch.nn.functional.cross_entropy(
                outputs.logits.view(-1, vocab_size), 
                input_ids.view(-1), 
                reduction='none').reshape(input_ids.shape)

    
    log_prob_next_tokens_list = []
    for i in range(num_tokens_ahead):
        outputs = model(input_ids=input_ids, past_key_values=past_key_values, attention_mask=attention_mask, position_ids=position_ids, use_cache=True)
        past_key_values = outputs.past_key_values
        position_ids, attention_mask = prepare_output_for_next_input(attention_mask, batch_size)
        position_ids = preprocess_position_ids(position_ids, is_in_thought=False) - int(add_last_context_token) + int(increment_pos_id_for_last_context_token)
        logits = outputs.logits
        if use_residual:
            logits += base_output_logits[:, i:i+input_seq_len]

        log_prob_next_tokens_list.append(
            torch.nn.functional.cross_entropy(
                logits.view(-1, vocab_size), 
                labels[:, i: i + input_seq_len].view(-1), 
                reduction='none')
            .reshape(input_ids.shape))
        if i == num_tokens_ahead - 1: # can't feed the model anything more to predict after this point, so should stop.
            ...
        else:
            # here we load up the next inputs to feed into the model
            input_ids = labels[:, i: i + input_seq_len] # [batch, seq_len]
    log_prob_next_tokens = torch.stack(log_prob_next_tokens_list, dim=2)# batch, seq_len, num_tokens_ahead
    
    # kv_cache = past_key_values
    # last_attention_mask = attention_mask
    if add_surogate_loss_to_last_context_token:
        return loss_over_last_context_token, log_prob_next_tokens
    return log_prob_next_tokens # , kv_cache, last_attention_mask

class ReasonerInterpreterTransformer():
    """
    Composed of four models. 
    One for base language modeling.
    One for base reasoning modeling ie kv cache init for reasoning model
    One for reasoning modeling (this has an LM head, and must be able to be called by sample)
    One for interpreting modeling being able to take in tokens from the reasoner and produce a hidden representation for the terminal token
    simple_lm_head would be the default, and then I would have to implement weight groups to ablate each difference.
    """
    def __init__(self, ):
        ...

class QuietStarDiscreteTransformer(torch.nn.Module):
    '''Single model for everything the ReasonerInterpreter does, also allow for multiple tokens ahead!'''
    def __init__(self, vocab_size, hidden_dim, n_layer, n_head, use_reasoner, use_residual, max_reasoning_len, max_seq_len, start_of_thought_token, end_of_thought_token, add_last_context_token, add_surogate_loss_to_last_context_token, increment_pos_id_for_last_context_token):
        super().__init__()
        self.base_lm_base_reasoner_reasoner_interpreter = CustomGPT2LMHeadModel(GPT2Config(vocab_size=vocab_size,
                                                                                           n_embd=hidden_dim,
                                                                                           n_layer=n_layer,
                                                                                           n_head=n_head,
                                                                                           resid_pdrop=0.0, 
                                                                                           embd_pdrop=0.0, 
                                                                                           attn_pdrop=0.0))
        self.start_of_thought_token = start_of_thought_token
        self.end_of_thought_token = end_of_thought_token
        self.max_seq_len = max_seq_len
        self.max_reasoning_len = max_reasoning_len
        self.vocab_size = vocab_size
        self.use_reasoner = use_reasoner
        self.use_residual = use_residual
        self.add_last_context_token = add_last_context_token
        self.add_surogate_loss_to_last_context_token = add_surogate_loss_to_last_context_token
        self.increment_pos_id_for_last_context_token = increment_pos_id_for_last_context_token
    def forward(self, x):
        # QuietStarGRULM: vocab_size, hidden_dim, use_base_lm, use_reasoner, max_reasoning_len, start_of_thought_token, debug_cfg
        # ReasonerInterpreterGRULM: vocab_size, reasoner_interpreter_vocab_size, base_lm_hidden_dim, reasoner_hidden_dim, interpreter_hidden_dim, use_base_lm, use_reasoner, simple_lm_head, weight_groups, share_lm_head, max_reasoning_len=10, start_of_thought_token=0

        # if collapse:
        #     assert n_tokens_ahead == 1, "only able to collapse when the number of tokens ahead is 1."
        # This must be used for inference, and will simply by default return the logits over the next tokens, so will be just n_tokens_ahead=1 by default.
        # the return for n_tokens_ahead=1 will be collapsed by default for compatability as compared to what is available for training.
        # the shape will typically be [batch, seq, n_tokens_ahead, vocab_size], but will be [batch, seq, vocab_size] for n_tokens_ahead=1 when collapse is True
        chosen_token_indices, log_prob_chosen_tokens, prob_dist_over_thought_tokens, repeated_last_fwd_pass_info = sample_thoughts(
            self.call_model, x, self.start_of_thought_token, self.max_reasoning_len, self.preprocess_position_ids)

        thought_augmented_key_values = repeated_last_fwd_pass_info[0]
        attention_mask_for_thoughts_and_next_token = repeated_last_fwd_pass_info[1]
        batch_size = x.size(0)
        seq_len = x.size(1)

        end_of_thought_input_ids = attention_mask_for_thoughts_and_next_token.new_full((batch_size, seq_len), fill_value=self.end_of_thought_token) # should be [batch, input_seq_len]
        position_ids = get_position_ids_from_attn_mask(attention_mask_for_thoughts_and_next_token, batch_size)
        position_ids = self.preprocess_position_ids(position_ids, is_in_thought=True)
        if self.add_last_context_token:
            outputs = self.call_model(input_ids=end_of_thought_input_ids, past_key_values=thought_augmented_key_values, attention_mask=attention_mask_for_thoughts_and_next_token, position_ids=position_ids, use_cache=True)
            thought_augmented_key_values = outputs.past_key_values
            position_ids, attention_mask_for_thoughts_and_next_token = prepare_output_for_next_input(attention_mask_for_thoughts_and_next_token, batch_size)
            position_ids = self.preprocess_position_ids(position_ids, is_in_thought=False) - int(self.add_last_context_token) + int(self.increment_pos_id_for_last_context_token)
            pre_output_input_ids = x
        else:
            pre_output_input_ids = end_of_thought_input_ids
        outputs = self.call_model(input_ids=pre_output_input_ids, past_key_values=thought_augmented_key_values, attention_mask=attention_mask_for_thoughts_and_next_token, position_ids=position_ids, use_cache=True)
        if self.use_residual:
            base_output_logits = self.base_lm_base_reasoner_reasoner_interpreter(x).logits 
            outputs.logits += base_output_logits
        # past_key_values = outputs.past_key_values
        # _, attention_mask = prepare_output_for_next_input(attention_mask_for_thoughts_and_next_token, batch_size)
        if not self.use_reasoner:
            outputs = self.base_lm_base_reasoner_reasoner_interpreter(x)
        return outputs.logits # , (past_key_values, attention_mask)
    def get_loss_and_hidden_states_and_log_prob_hidden_states_and_dist(self, input_ids, labels, n_tokens_ahead):
        if self.use_residual:
            base_input_ids = input_ids
            input_ids = input_ids.clone()[:, :-1]
        else:
            base_input_ids = input_ids
        hidden_states, log_prob_hidden_states, prob_dist_hidden_states, last_fwd_pass_info = sample_thoughts(
            self.call_model, input_ids, self.start_of_thought_token, self.max_reasoning_len, self.preprocess_position_ids)
        thought_augmented_key_values, attention_mask_for_thoughts_and_next_token = last_fwd_pass_info

        # technically labels don't have to be next tokens, but for every case I will use labels for they will be.
        losses = use_thoughts_to_get_loss_over_next_tokens(
            self.call_model, labels, self.vocab_size, thought_augmented_key_values, attention_mask_for_thoughts_and_next_token, self.end_of_thought_token, n_tokens_ahead, base_input_ids, self.use_residual, self.preprocess_position_ids, self.add_last_context_token, self.add_surogate_loss_to_last_context_token, self.increment_pos_id_for_last_context_token)
        loss_over_last_context_token = None
        if self.add_surogate_loss_to_last_context_token:
            loss_over_last_context_token, loss_over_labels = cast(tuple[torch.Tensor, torch.Tensor], losses)
        else:
            loss_over_labels = cast(torch.Tensor, losses)
        if not self.use_reasoner:
            outputs = self.base_lm_base_reasoner_reasoner_interpreter(input_ids)
            loss_over_labels = torch.nn.functional.cross_entropy(outputs.logits.view(-1, self.vocab_size), labels.view(-1), reduction='none')
            loss_over_labels = loss_over_labels.reshape(input_ids.shape).unsqueeze(-1)
            log_prob_hidden_states = torch.zeros_like(log_prob_hidden_states) # ensure that no gradient is given to the reasoning even in the case that we have numerical instability in the instance - mean for trice.
        if self.add_surogate_loss_to_last_context_token:
            return (cast(torch.Tensor, loss_over_last_context_token), loss_over_labels), hidden_states, log_prob_hidden_states, prob_dist_hidden_states
        return loss_over_labels, hidden_states, log_prob_hidden_states, prob_dist_hidden_states
    def preprocess_position_ids(self, position_ids, is_in_thought):
        new_position_ids = position_ids.clone()
        if is_in_thought:
            new_position_ids += self.max_seq_len
        else:
            new_position_ids -= (self.max_reasoning_len + 2) # [0 + 12 + 1, 1 + 12 + 1, 2 + 12 + 1, 3 + 12 + 1, 4 + 12 + 1] - 12 = [1, 2, 3, 4, 5] then minus 1 in the get loss function? yes loss function so later I can do CE and it won't look weird.
        return new_position_ids
    def call_model(self, *args, **kwargs):
        if len(args) > 2:
            attention_mask = args[2]
            attention_mask = invert_and_maxfloat_attn_mask(attention_mask, dtype=self.base_lm_base_reasoner_reasoner_interpreter.transformer.wte.weight.dtype)
            args = tuple(arg if i != 2 else attention_mask for i, arg in enumerate(args))
        elif 'attention_mask' in kwargs:
            attention_mask = kwargs.pop('attention_mask')
            attention_mask = invert_and_maxfloat_attn_mask(attention_mask, dtype=self.base_lm_base_reasoner_reasoner_interpreter.transformer.wte.weight.dtype)
            kwargs['attention_mask'] = attention_mask
        return self.base_lm_base_reasoner_reasoner_interpreter(*args, **kwargs)

class QuietStarDiscreteTransformerLM(QuietStarDiscreteTransformer): # here I subclass the model because it would otherwise be a wrapper. More important to subclass nn.Module for Reasoner Interpreter.
    '''Produces probability distribution over the langauge passed in. 
    But also has an interface for training which allows for the log 
    probability of multiple tokens ahead to be augmented based on some reasoning.'''
    # TODO: implement sample for qualitative comparison at some point. 
    # putting off because would probably have to support caching in the forward pass, 
    # and this isn't so important right now, but might take some time.
    # def sample(self, num_samples):
        # this function could technically be implemented in the Transformer (not LM) 
        # but I am leaving an indirection to allow for easier compatability with Reasoner Interpreter.
    ...


def get_trice_loss_ret_dict_from_rewards(repeated_reward, repeated_reward_n_ahead, original_batch_size, trice_samples, train_nll_num, repeated_log_p_hidden_states, only_positive, nll_loss_beta, policy_loss_beta, dist, print_stuff, entropy_encouragement_coef):
    trice_baseline_reward_n_ahead = repeated_reward_n_ahead.reshape(original_batch_size, trice_samples, -1)
    repeated_reward_n_ahead_minus_baseline = (repeated_reward_n_ahead.view(original_batch_size, trice_samples, -1) - trice_baseline_reward_n_ahead.mean(1, keepdim=True)).view(*repeated_reward_n_ahead.shape)
    if only_positive:
        repeated_reward_n_ahead_minus_baseline = repeated_reward_n_ahead_minus_baseline.clamp(min=0)
    # # trying some bounding shit: dint wrk
    # repeated_reward_n_ahead_minus_baseline = repeated_reward_n_ahead_minus_baseline.clamp(-0.00001, 0.00001)
    quiet_star_policy_loss = - (repeated_reward_n_ahead_minus_baseline.detach() * repeated_log_p_hidden_states.sum(-1)).sum()
    quiet_star_policy_loss_numel = repeated_reward_n_ahead_minus_baseline.numel()
    
    dist_entropy = dist.clone()
    dist_entropy[dist > 1e-7] = -dist_entropy[dist > 1e-7] * dist_entropy[dist > 1e-7].log()
    dist_entropy[dist < 1e-7] = 0
    dist_entropy = dist_entropy.sum(-1)
    # dist_entropy = torch.distributions.Categorical(dist).entropy()
    # dist_entropy = -(torch.where(dist > 1e-7, dist * dist.log(), torch.zeros_like(dist))).sum(-1) # numerically stable entropy calculation should be done. Taking the log of 0 doesn't work nicely.
    if dist_entropy.numel() > 0:
        non_pad_mask = (dist[:,:-1,:,-1] != 1)
        non_pad_tokens = non_pad_mask.float().sum(-1) + 0.00001 # assumes pad token is the last entry. This is risky business.
        entropy_encouragement_loss = -(dist_entropy[:,:-1][non_pad_mask]).sum() / non_pad_tokens.sum() * quiet_star_policy_loss_numel  # entropy per reasoning token is what we care about, but per reasoning token which is not the end actually..
        # there is something with the per token calculation that I am not thinking through right now. re;ated to accumulation
    else:
        entropy_encouragement_loss = 0

    nll_loss = (-repeated_reward.reshape(original_batch_size, trice_samples, -1)[:, :train_nll_num]).sum() * trice_samples / train_nll_num # quick fix to make the num el correct, eventually they divide by num el so you want to scale the loss
    nll_loss_numel = repeated_reward.numel() 
    loss = nll_loss_beta * nll_loss # mean allowed because no pad tokens.
    if policy_loss_beta != 0.0:
        loss += policy_loss_beta * quiet_star_policy_loss
    if entropy_encouragement_coef != 0.0:
        loss += entropy_encouragement_coef * entropy_encouragement_loss

    quiet_star_policy_loss = float(quiet_star_policy_loss) / quiet_star_policy_loss_numel
    nll_loss = nll_loss.item() / nll_loss_numel
    entropy_encouragement_loss = entropy_encouragement_loss / quiet_star_policy_loss_numel
    sentence_entropy = dist_entropy.mean(-1)
    # if print_stuff:
    #     print()
        # print()
        # print(f"{quiet_star_policy_loss = }")
        # print(f"{nll_loss = }")
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
        # print("distribution entropy min (0), mean, max (4.18):", dist_entropy.min().item(), dist_entropy.mean().item(), dist_entropy.max().item())
        # print("sentence entropy per token min, mean, max", sentence_entropy.min().item(), sentence_entropy.mean().item(), sentence_entropy.max().item())
        # if loss.isnan().any() or loss > 10:
        #     import ipdb; ipdb.set_trace()
    assert quiet_star_policy_loss_numel == nll_loss_numel, "need to have the same number of elements for the summed loss to have the correct meaning."
    ret_dict = {"loss": loss,
                "numel": quiet_star_policy_loss_numel,
                # "example_hidden_state": ,
                "nll_loss": float(nll_loss),
                "quiet_star_policy_loss": float(quiet_star_policy_loss)
                }
    if dist_entropy.numel() > 0:
        assert quiet_star_policy_loss_numel == dist_entropy[:,:-1,0].numel(), "needs the same number of elements for division to make sense in the main loop."
        ret_dict.update({
                    "entropy_encouragement_loss": float(entropy_encouragement_loss),
                    "reasoning_entropy_min": float(dist_entropy.min()),
                    "reasoning_entropy_mean": float(dist_entropy.mean()),
                    "reasoning_entropy_max": float(dist_entropy.max()),
                    "sentence_reasoning_entropy_min": float(sentence_entropy.min()),
                    "sentence_reasoning_entropy_mean": float(sentence_entropy.mean()),
                    "sentence_reasoning_entropy_max": float(sentence_entropy.max())
                    })
    return ret_dict

def get_quiet_star_loss_reason_once(model, inputs: torch.Tensor, policy_loss_beta=1e6, nll_loss_beta=1.0, last_context_loss_beta=1.0, trice_samples=2, train_nll_num=2, n_tokens_ahead=1, only_positive=False, print_stuff=True, entropy_encouragement_coef=0.0):
    '''The fundamental difference is that this loss function is designed to work nicely with reasoning only in one
    location, and then having a variable number of tokens that reasoning can impact based on the n_tokens_ahead.
    '''
    ret_dict = dict()
    original_batch_size = inputs.size(0)
    repeated_inputs = inputs.repeat_interleave(trice_samples, dim=0) # every example shows up twice, this for trice!
    repeated_inputs_for_thought_sampling, repeated_inputs_for_next_token_prediction = prepare_shifted_inputs_for_thought_sampling_and_next_token_prediction_for_loss(repeated_inputs, n_tokens_ahead)
    if model.use_residual:
        repeated_inputs_for_thought_sampling = repeated_inputs
    losses, repeated_hidden_states, repeated_log_prob_hidden_states, repeated_prob_dist_hidden_states = model.get_loss_and_hidden_states_and_log_prob_hidden_states_and_dist(repeated_inputs_for_thought_sampling, repeated_inputs_for_next_token_prediction, n_tokens_ahead)
    repeated_last_context_loss = None
    if model.add_surogate_loss_to_last_context_token:
        repeated_last_context_loss, repeated_next_tokens_loss = cast(tuple[torch.Tensor, torch.Tensor], losses)
        last_context_loss = repeated_last_context_loss.mean()
        ret_dict['last_context_loss'] = float(last_context_loss)
    else:
        repeated_next_tokens_loss = cast(torch.Tensor, losses)

    repeated_next_tokens_reward = - repeated_next_tokens_loss.squeeze(-1)
    
    
    ret_dict.update(get_trice_loss_ret_dict_from_rewards(repeated_next_tokens_reward,
                                repeated_next_tokens_reward,
                                original_batch_size,
                                trice_samples,
                                train_nll_num,
                                repeated_log_prob_hidden_states,
                                only_positive,
                                nll_loss_beta,
                                policy_loss_beta,
                                repeated_prob_dist_hidden_states,
                                print_stuff,
                                entropy_encouragement_coef))
    if 'last_context_loss' in ret_dict:
        repeated_last_context_loss = cast(torch.Tensor, repeated_last_context_loss)
        # going to assume only one next token for now to not deal with the strange loss scaling.
        assert repeated_last_context_loss.numel() == ret_dict['numel']
        ret_dict['loss'] += repeated_last_context_loss.sum() * last_context_loss_beta

    
    return ret_dict
    # this one actually should expect the model to have handled the shifting??? 
    # (why would it do that? Because I see it as wasteful to generate a bunch of thoughts when not needed, 
    # but it is still technically around just 1/128th to n_ahead/128th of the cost)
    # I can handle the shifting outside of this function, I just need to access the loss function computation myself.
    # Although repeated logits is returned from the function call to the model right now, 
    # I should instead have this be adaptable to the loss function rather than the model.
    # This would allow for a kind of schedule of many or few next token aheads. 
    # And repeated logits only kind of makes sense when the hidden representation or 
    # reasoning changes every token down stream, which is a cool thing, 
    # but I am not at the stage where that makes sense to try out.



def get_quiet_star_loss_reason_multiple(model: Any, inputs: torch.Tensor, policy_loss_beta:float=1e6, nll_loss_beta:float=1, trice_samples:int=2, train_nll_num=2, n_tokens_ahead=1, punish_unfinished=0.0, length_penalty=0.0, only_positive=False, print_stuff=True, entropy_encouragement_coef=0.0, test_mem=False):
    if n_tokens_ahead != 1:
        raise Exception("The n_tokens_ahead implementation here doesn't work for discrete GRU quiet-star, only for stochastic GRU. If ever I consider all the generated thoughts as a sequence, then perhaps this will become a relevant implementation again")
    # this loss consists of nll for the base model? 
    # well given that we don't parameterize a basemodel,
    # we will ignore that part of the loss for now.
    # getting nll for thoughts is still important tho, for training the lm head.
    n_tokens_ahead = min(inputs.size(1)-1, n_tokens_ahead)
    labels = inputs.clone()
    original_batch_size = inputs.size(0)
    repeated_inputs = inputs.repeat_interleave(trice_samples, dim=0) # every example shows up twice, this for trice!
    repeated_labels = labels.repeat_interleave(trice_samples, dim=0) # every example shows up twice, this for trice!
    # if hasattr(model.model, "mix_interpeter_base_lm"):
    # just quick and dirty test this entropy idea.
    is_var_len = hasattr(model, "variable_len") and model.variable_len
    if test_mem and is_var_len: 
        model.model.variable_len = False
    repeated_logits, repeated_hidden_states, repeated_log_p_hidden_states, dist = model.get_logits_and_hidden_states_and_log_prob_hidden_states_dist(repeated_inputs)
    if test_mem and is_var_len: # ensure that a full length reasoning is generated, but then convert back to original state.
        model.model.variable_len = True
    # to encourage entropy in the distribution should get the logits over next thought tokens and put a reward term on the entropy. add this to the loss
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

    if print_stuff:
        print(repeated_hidden_states[0,:10])
    # for variable length models should penalize not ending the thought. Encourages model to control its length by itself instead of degenerating to easy to ignore low entropy
    if hasattr(model, "variable_len") and model.variable_len and punish_unfinished != 0.0:
        repeated_reward_n_ahead -= (1 - (repeated_hidden_states[:, :-1] == model.start_of_thought_token).sum(dim=-1)) * punish_unfinished
    if hasattr(model, "variable_len") and model.variable_len and length_penalty != 0.0:
        repeated_reward_n_ahead -= (((repeated_hidden_states[:, :-1] != model.pad_token) & (repeated_hidden_states[:, :-1] != model.start_of_thought_token)).sum(dim=-1)) * length_penalty
    # no baseline to regress from, but can still average to create a baseline
    ret_dict = get_trice_loss_ret_dict_from_rewards(repeated_reward,
                                repeated_reward_n_ahead,
                                original_batch_size,
                                trice_samples,
                                train_nll_num,
                                repeated_log_p_hidden_states[:, :-1], # the implication that you can apply all these hidden states to the labels.
                                only_positive,
                                nll_loss_beta,
                                policy_loss_beta,
                                dist,
                                print_stuff,
                                entropy_encouragement_coef)
    if hasattr(model, "variable_len") and model.variable_len and repeated_hidden_states.numel() != 0:
        repeated_hidden_states2d = repeated_hidden_states.reshape(-1, repeated_hidden_states.size(-1))
        lens = torch.full_like(repeated_hidden_states2d[:, 0], fill_value=repeated_hidden_states2d.size(-1))
        rows_with_ends, end_indices = torch.where(repeated_hidden_states2d == model.start_of_thought_token)
        lens[rows_with_ends] = end_indices
        
        ret_dict['avg_reasoning_len'] = float(lens.float().mean())
        ret_dict['frac_empty_reasonings'] = int((repeated_hidden_states[...,0] == model.start_of_thought_token).sum()) / repeated_hidden_states[...,0].numel()
    return ret_dict

def get_nll_from_logits_and_labels(logits, labels):
    shifted_logits = logits[:, :-1].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    return torch.nn.CrossEntropyLoss(reduction='none')(shifted_logits.view(-1, logits.size(-1)), shifted_labels.view(-1))
def get_nll(model, inputs, print_stuff=False, **kwargs):
    labels = inputs.clone()
    model_return = model(inputs)
    if isinstance(model_return, torch.Tensor):
        logits = model_return
    else:
        assert hasattr(model_return, "logits"), "we assume we are dealing with a huggingface model"
        logits = model_return.logits
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
