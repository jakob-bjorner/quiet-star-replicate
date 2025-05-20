# Replicating and extending quiet-star's main results

main idea is to make the main results of quiet star model agnostic. The reason being that the parallel attention head doesn't make sense for generation. The complexity with their implementation

First for a single token: i < n_thought + n_talk
$$O((seq\_len \cdot i) \cdot seq\_len \cdot d)$$
the seq times i represents the number of key tokens in the attention dot product, which is the primary computation I care about when considering a parallel attention mask like the one given in the paper. if we sum over these i, we get The complexity of all the attention matrices:

$$O(seq\_len \cdot \frac{((n\_thought + n\_talk)(n\_thought + n\_talk + 1))}{2} \cdot seq\_len \cdot d)$$
$$O(seq\_len^2 \cdot (n\_thought + n\_talk)^2 \cdot d)$$


Now, if we would just do the computation of the next tokens during some batched parallel generation, each i token would take:

$$\sum_{l=1}^{seq\_len}((l + i) \cdot 1 \cdot d)$$
$$= O(\frac{seq\_len \cdot (seq\_len + 1)}{2} \cdot d + seq\_len \cdot i \cdot d)$$
and to get the computation for all tokens we again sum over the i:
$$\sum_{l=1}^{seq\_len}((l + i) \cdot 1 \cdot d)$$
$$= O((n\_thought + n\_talk) \cdot \frac{seq\_len \cdot (seq\_len + 1)}{2} \cdot d + seq\_len \cdot \frac{((n\_thought + n\_talk)(n\_thought + n\_talk + 1))}{2} \cdot d)$$
$$= O((n\_thought + n\_talk) seq\_len^2 \cdot d + seq\_len \cdot(n\_thought + n\_talk)^2 \cdot d)$$

on top of this analysis, we have the added benefit of not having to do a forward pass for each n_talk token, making this part achievable in parallel reducing the number of forward passes through our model by a substantial amount!

What about the creation of many different gradients?
Could try to do generation based on a key_value cache, and share the key values between what I am trying to generate? then the gradients accumulate to the same tensor as they all eventually should. I  don't know if passing the view of the same tensor will just work.
The memory created by this duplication was immense, and the quiet-STAR authors note in their paper that the memory I associate with their method can be reduced by simply not doing the pairwise multiplication to get the attention matricies per head, and instead doing element wise for the attention matrix which is a simple cross diagonal. It seems they just didn't implement this speed up, which I think makes their generation better in every way.

## Section on simplified quiet-STAR setting

The idea of modeling the latent reasoning as just another hidden layer right before the talk head. Hense the architecture will be the same as an LSTM, and for now if we only do a single token prediction, we just have to train the following, amoung its variations

$$\argmax_{\pi} \mathbb{E}_{\substack{x \sim D(x) \\ h \sim \pi(h | x) \\ y \sim \pi(y | h)}} [\log{D(y | x)}] + \beta \mathbb{E}_{\substack{x \sim D(x) \\ h \sim \pi(h | x)}} [\mathbb{H}[\pi(y | h)]]$$

The policy model outputs y, and it outputs h, and both can be done probabilistically, and to convince you that the reward of $\log{D(y | x)}$ is a good reward, lets look at the simple case where the production of h is just another deterministic layer of p.

$$\argmax_{\pi} \mathbb{E}_{\substack{x \sim D(x) \\ y \sim \pi(y | x)}} [\log{D(y | x)}] + \beta \mathbb{E}_{\substack{x \sim D(x)}} [\mathbb{H}[\pi(y | x)]]$$

For beta = 1, we will show that we get out the standard NLL objective.
For beta = 0, it can be shown that you get out the mode finding objective.

$$\argmax_{\pi} \mathbb{E}_{\substack{x \sim D(x) \\ y \sim \pi(y | x)}} [\log{D(y | x)}] + \mathbb{E}_{\substack{x \sim D(x)}} [\mathbb{H}[\pi(y | x)]]$$
$$\argmax_{\pi} \mathbb{E}_{\substack{x \sim D(x) \\ y \sim \pi(y | x)}} [\log{D(y | x)}] + \mathbb{E}_{\substack{x \sim D(x) \\ y \sim \pi(y | x)}} [\log{\frac{1}{\pi(y | x)}}]$$
$$\argmax_{\pi} \mathbb{E}_{\substack{x \sim D(x) \\ y \sim \pi(y | x)}} [\log{\frac{D(y | x)}{\pi(y | x)}}]$$
$$\argmin_{\pi} \mathbb{E}_{\substack{x \sim D(x)}} [\text{KL}[\pi(y | x) || D(y | x)]]$$

which is also minimized by the reverse KL, at its optima assuming the model is powerful enough to capture the full distribution. If it isn't then the performances of the two variations are different with the forward mode KL performing mode seaking, and this reverse mode having a mean seaking behavior.

$$\argmin_{\pi} \mathbb{E}_{\substack{x \sim D(x)}} [\text{KL}[ D(y | x) || \pi(y | x)]]$$
$$\argmin_{\pi} \mathbb{E}_{\substack{x \sim D(x) \\ y \sim D(y|x)}} [\log \frac{1}{\pi(y | x)}] - \mathbb{E}_{x \sim D(x)}[\mathbb{H}[D(y | x)]]$$
$$\argmin_{\pi} \mathbb{E}_{\substack{x \sim D(x) \\ y \sim D(y|x)}} [\log \frac{1}{\pi(y | x)}]$$

//

Now, given that we believe $\log D(y | x)$ is a good reward at least to capture the language modeling of the true text (not necessarily to constrain the thoughts in any way), treating the hidden state as our equivelant latent thinking head, we have to come up with a way to optimize it without passing a gradient directly through h. Only through the probability density model over h. It kind of reminds me of a shooting method like CEM or other shooting methods, but this time we have a potentially powerful probability model, and we choose which probabilities to make more likely based on their performance. We could model the problem as a gaussian prediction over h, then the ones which have slightly higher performance will be like gradient descent steps from the mean?

But first, we need to see what the different options are for optimizing. Like what reward functions can we pick from for each of $\pi(h | x)$ and $\pi(y | h)$? How can we train the LM head?

$$\argmax_{\pi} \mathbb{E}_{\substack{x \sim D(x) \\ h \sim \pi(h | x) \\ y \sim \pi(y | h)}} [\log{D(y | x)}] + \beta \mathbb{E}_{\substack{x \sim D(x) \\ h \sim \pi(h | x)}} [\mathbb{H}[\pi(y | h)]]$$
assume beta = 1 (just because it worked out nicely above) 
$$\argmax_{\pi} \mathbb{E}_{\substack{x \sim D(x) \\ h \sim \pi(h | x) \\ y \sim \pi(y | h)}} [\log{\frac{D(y | x)}{\pi(y | h)}}]$$
$$\argmax_{\pi} \mathbb{E}_{\substack{x \sim D(x) \\ h \sim \pi(h | x)}} [-\text{KL}[\pi(y | h)|| D(y | x)]]$$

When y is only a single token (not sure what happens when y is multiple tokens? works out above in language modeling case, but here not sure?) this KL term is easy to compute when we have a form for D(y | x), which can come from an existing language model for simplicity. Critically assuming you have a great model for D(y | x) is kind of defeating the point of modeling language in a different way. we could say that we just want a good model for the KL between these two distributions given some representation and some context, this is to tell us how two representations will perform.

We may also model with reverse KL as they have the same optima for the lm head. 

$$\argmax_{\pi} \mathbb{E}_{\substack{x \sim D(x) \\ h \sim \pi(h | x)}} [-\text{KL}[D(y | x) || \pi(y | h)]]$$
$$\argmax_{\pi} \mathbb{E}_{\substack{x \sim D(x) \\ h \sim \pi(h | x) \\ y \sim D(y | x)}} [\log\pi(y | h)] + \mathbb{E}_{\substack{x \sim D(x) \\ h \sim \pi(h | x)}}[\mathbb{H}[D(y | x)]]$$
$$\argmax_{\pi} \mathbb{E}_{\substack{x \sim D(x) \\ h \sim \pi(h | x) \\ y \sim D(y | x)}} [\log\pi(y | h)]$$

we can then motivate the reward as the expected log likelihood given some h over your dataset

$$\argmax_{\pi} \mathbb{E}_{\substack{x \sim D(x) \\ h \sim \pi(h | x)}} [\mathbb{E}_{y \sim D(y | x)}[\log\pi(y | h)]]$$

$$\bar r_\psi(h, x) \approx \mathbb{E}_{y \sim D(y | x)}[\log\pi(y | h)]$$

learned through MSE on the sampled h and x. Using this reward, we can integrate another entropy encouragement term, and rederive the DPO objective for finding the optimal policy given the reward.

$$\argmax_{\pi} \mathbb{E}_{\substack{x \sim D(x) \\ h \sim \pi(h | x)}} [\bar r_\psi(h, x)] + \beta_2\mathbb{E}_{\substack{x \sim D(x)}}[\mathbb{H}[\pi(h | x)]]$$
$$\argmax_{\pi} \mathbb{E}_{\substack{x \sim D(x) \\ h \sim \pi(h | x)}} [\beta_2\log \exp(\frac{1}{\beta_2}\bar r_\psi(h, x)) + \beta_2 \log \frac{1}{\pi(h | x)}]$$
$$\argmax_{\pi} \mathbb{E}_{\substack{x \sim D(x) \\ h \sim \pi(h | x)}} [\log \frac{\exp(\frac{1}{\beta_2}\bar r_\psi(h, x))}{\pi(h | x)}]$$

Hence the objective of matching $\frac{1}{Z(x)}\exp(\frac{1}{\beta_2}\bar r_\psi(h, x)) = \pi(h | x)$ which is somewhat accomplished by the placket luce model of preferences. (We can ensure that confident (as in an ensemble agree) reward distributions are matched ensuring that we always have s which we want to match (typically s is 2 in DPO))

$$\argmax_{\pi} \mathbb{E}_{\substack{x \sim D(x) \\ h_1,h_2,...,h_s \sim \frac{1}{Z(x)}\exp(\frac{1}{\beta_2}\bar r_\psi(h, x))}} [\sum_i^s\frac{\exp(\frac{1}{\beta_2}\bar r_\psi(h_i, x))}{\sum_j^s\exp(\frac{1}{\beta_2}\bar r_\psi(h_j, x))}\log \frac{\frac{\pi(h_i | x)}{\sum_j^s\pi(h_j | x)} }{\frac{\exp(\frac{1}{\beta_2}\bar r_\psi(h_i, x))}{\sum_j^s\exp(\frac{1}{\beta_2}\bar r_\psi(h_j, x))}}]$$

Although $\bar r_\psi(h, x)$ depends on $\pi(h | x)$, we are ignoring this during our derivative computation of the above objective (no good justification for this yet). When we do this, we don't need to explicitly compute the any of the terms in the first denominator of the fraction in the log, as they come out as constants, which just depend on which h_i we sample, which can be from the $\pi(h | x)$ or from some prior distirbution.

$$\argmax_{\pi} \mathbb{E}_{\substack{x \sim D(x) \\ h_1,h_2,...,h_s \sim \frac{1}{Z(x)}\exp(\frac{1}{\beta_2}\bar r_\psi(h, x))}} [\sum_i^s \frac{\exp(\frac{1}{\beta_2}\bar r_\psi(h_i, x))}{\sum_j^s\exp(\frac{1}{\beta_2}\bar r_\psi(h_j, x))}\log \frac{\pi(h_i | x)}{\sum_j^s\pi(h_j | x)}] \\ + \mathbb{E}_{\substack{x \sim D(x) \\ h_1,h_2,...,h_s \sim \frac{1}{Z(x)}\exp(\frac{1}{\beta_2}\bar r_\psi(h, x))}}[\sum_i^s\frac{\exp(\frac{1}{\beta_2}\bar r_\psi(h_i, x))}{\sum_j^s\exp(\frac{1}{\beta_2}\bar r_\psi(h_j, x))} \log\frac{\sum_j^s\exp(\frac{1}{\beta_2}\bar r_\psi(h_j, x))}{\exp(\frac{1}{\beta_2}\bar r_\psi(h_i, x))}]$$

The ignoring of where the samples come from is very questionable as iterative DPO relies on the very same sampling problem, which has proven to be more than trivial. 

$\color{red}\text{Need to think about how samples for h are derived from optimal distribution at some point}$

$$\argmax_{\pi} \mathbb{E}_{\substack{x \sim D(x) \\ h_1,h_2,...,h_s \sim \frac{1}{Z(x)}\exp(\frac{1}{\beta_2}\bar r_\psi(h, x))}} [\sum_i^s \frac{\exp(\frac{1}{\beta_2}\bar r_\psi(h_i, x))}{\sum_j^s\exp(\frac{1}{\beta_2}\bar r_\psi(h_j, x))}\log \frac{\pi(h_i | x)}{\sum_j^s\pi(h_j | x)}]$$

Also, the assumption that the same probability mass is always in h 1 to s isn't valid because we don't directly sample from the optimal policy, so we have no idea how much mass is in each of those terms. would want some estimation of Z(x), which is possible to achieve... I think. using a reference policy for h it may be easier because we know the distirbution to sample from, and then just get simple random samples to estimate the expected exponential reward which is what Z is. And once we have Z, the problem is actually way easier because we have a probability mass to match, and we can do this via MSE on the log policy plus the beta weighted log Z which should equal the expected reward for that z, h pair.


the h 1 through s are technically sampled from whatever distribution I choose like pi or normal distirbution, and not at all from the optimal distribution, which isn't so good, but I don't know what to do about it...


<!-- ## Alpha zero with DPO like Agent-Q? -->
reward model surrogate:
$$\bar r_\psi(h, x) \approx \mathbb{E}_{y \sim D(y | x)}[\log\pi(y | h)]$$

$$\tilde{\bar r_\psi}(h, x) = \mathbb{E}_{y \sim \text{LM}(y | x)}[\log\frac{\pi(y | h)}{\text{LM}(y | x)}]$$


where LM serves as a proxy for the true distribution over next tokens.

# Quiet Star objective differentiation
$$L(D) = \mathbb{E}_{x,y\sim D}[\mathbb{E}_{z\sim p_\theta(z | x)}[\log{\frac{1}{p_\theta(y | z, x)}}]]$$
$$\nabla_\theta L(D) = \nabla_\theta \mathbb{E}_{x,y\sim D}[\mathbb{E}_{z\sim p_\theta(z | x)}[\log{\frac{1}{p_\theta(y | z, x)}}]]$$
$$ =  \mathbb{E}_{x,y\sim D}[\sum_z[\nabla_\theta (p_\theta(z | x)\log{\frac{1}{p_\theta(y | z, x)}})]]$$
Apply product rule: ($\frac{\partial}{\partial x}(f(x) * g(x)) = \frac{\partial}{\partial x}f(x) * g(x)+  f(x) * \frac{\partial}{\partial x}g(x)$)
$$ =  \mathbb{E}_{x,y\sim D}[\sum_z[ \nabla_\theta p_\theta(z | x)\log{\frac{1}{p_\theta(y | z, x)}} + p_\theta(z | x) \nabla_\theta \log{\frac{1}{p_\theta(y | z, x)}}]]$$
$$ =  \mathbb{E}_{x,y\sim D}[\sum_z[ p_\theta(z | x)\nabla_\theta \log{p_\theta(z | x)}\log{\frac{1}{p_\theta(y | z, x)}} + p_\theta(z | x) \nabla_\theta \log{\frac{1}{p_\theta(y | z, x)}}]]$$
$$ =  \mathbb{E}_{x,y\sim D}[\mathbb{E}_{z\sim p_\theta(z | x)}[ \nabla_\theta \log{p_\theta(z | x)}\log{\frac{1}{p_\theta(y | z, x)}} + \nabla_\theta \log{\frac{1}{p_\theta(y | z, x)}}]]$$
# Questioning the utility of rational

Here, I test whether there is an interpreter hidden dimension where the performance of the reasoning model alone beats that of the reasoner + interpreter combo due to ingoring on the part of the interpreter so as to cause the reasoner not to train. 

Maybe also I should make the reasoner interpreter part like huge? like 1000 so it has a higher ceiling, then it will be clear when it is ignored.

// main experiment base lm capacity vs performance with reasoning
python run.py -m +run_modifier=ReasonerInterpreterGRURunConfig base_lm_hidden_dim=2,4,8,16,32,64 seed=7,8,9 hydra.launcher.partition='kargo-lab'

// this for seeing if there is some base model capacity which ignores the use of a high capacity reasoning model. TODO: add gradient accumulation to support this test without having to change the effective batch size so as to not change variables.
python run.py -m +run_modifier=ReasonerInterpreterGRURunConfig base_lm_hidden_dim=2,4,8,16,32,64 seed=7 reasoner_hidden_dim=1000

// this for determining reasoning performance on it's own
python run.py -m +run_modifier=ReasonerInterpreterGRURunConfig seed=7,8,9

python run.py -m +run_modifier=InterpreterGRURunConfig base_lm_hidden_dim=2,4,8,16,32,64 seed=7 hydra.launcher.partition='kargo-lab'

salloc --qos debug -p kargo-lab --cpus-per-gpu=6 -G, --gpus-per-node=a40:1 --exclude="voltron,sonny,kitt,samantha,major,crushinator,nestor,megabot,uniblab,gundam,consu,brainiac,heistotron,deebot,conroy,robby,qt-1,omgwth,puma"

# seeing if capacity can reduce negative transfer when reasoner and interpreter share weights.

python run.py -m +run_modifier=QuietSTaRDiscreteRunConfig model_hidden_dim=2,4,8,16,32,64,128,256,512 seed=7,8

python run.py -m +run_modifier=QuietSTaRDiscreteRunConfig model_hidden_dim=2,4,8,16,32,64,128,256,512 seed=7,8 use_reasoner=False

python run.py -m +run_modifier=QuietSTaRDiscreteRunConfig model_hidden_dim=2048 seed=7 use_reasoner=True,False

- seems to be no by just increasing GRU hidden dimension

python run.py -m+run_modifier=QuietSTaRDiscreteRunConfig seed=7,8 debug_cfg="seperateInterpreter" info="seperateInterpreterDEBUG_"

# seeing if model learning rate on policy loss is holding back the policy learning
python run.py -m +run_modifier=QuietSTaRDiscreteRunConfig policy_loss_beta=10,100,1000,10000 use_base_lm=True,False
- training just seems to be more unstable. curious why quiet-star would have thought of doing this? I can only understand this improving their method through the benefits of over tuning.


# making changes to allow for the independant testing of combinations of different parts of the network in the GRU setting.
python run.py -m +run_modifier=ReasonerInterpreterGRURunConfig simple_lm_head=True,False

// ensure a performance gap still favors rationale as sanity check: it did, but definitely worth it to do this easy thing first because I ran into some tough details.
python run.py -m +run_modifier=ReasonerInterpreterGRURunConfig simple_lm_head=True base_lm_hidden_dim=32,64 use_reasoner=True
// use reasoner = False efficient implementation This faster implementation is just harder to work with because of the way it is logged, but to fix the way it is logged would be mentally a struggle right now, so I will just run the slow implementation, so the logging is nice?
python run.py -m +run_modifier=InterpreterGRURunConfig simple_lm_head=True base_lm_hidden_dim=32,64,100 seed=7,8,9

TODO: Something weird with projection. from just lm, adding a two teired projection system was beneficial to performance somehow. Even with no non linearity. This is something Kartik has told me about before but I don't recall... 



python run.py -m +run_modifier=ReasonerInterpreterGRURunConfig simple_lm_head=True 
weight_groups=
['A','A','A','A'],
['A','A','A','B'],['A','A','B','A'],['A','B','A','A'],['B','A','A','A'],
['A','A','B','B'],['A','B','A','B'],['A','B','B','A'],
['A','A','B','C'],['A','B','A','C'],['A','B','C','A'],['B','A','A','C'],['B','A','C','A'],['B','C','A','A']
['A','B','C','D']

['A','A','B','A'] share_lm_head=True

base_lm_hidden_dim=32 reasoner_hidden_dim=32 interpreter_hidden_dim=32
base_lm_hidden_dim=64 reasoner_hidden_dim=64 interpreter_hidden_dim=64
base_lm_hidden_dim=100 reasoner_hidden_dim=100 interpreter_hidden_dim=100

python run.py -m +run_modifier=ReasonerInterpreterGRURunConfig simple_lm_head=True weight_groups=['A','A','A','A'] base_lm_hidden_dim=32 reasoner_hidden_dim=32 interpreter_hidden_dim=32 share_lm_head=True seed=7,8,9 use_reasoner=True,False
python run.py -m +run_modifier=ReasonerInterpreterGRURunConfig simple_lm_head=True weight_groups=['A','A','A','A'] base_lm_hidden_dim=64 reasoner_hidden_dim=64 interpreter_hidden_dim=64 share_lm_head=True seed=7,8,9 use_reasoner=True,False
python run.py -m +run_modifier=ReasonerInterpreterGRURunConfig simple_lm_head=True weight_groups=['A','A','A','A'] base_lm_hidden_dim=100 reasoner_hidden_dim=100 interpreter_hidden_dim=100 share_lm_head=True seed=7,8,9 use_reasoner=True,False

python run.py -m +run_modifier=ReasonerInterpreterGRURunConfig simple_lm_head=True weight_groups=['A','A','A','A'],['A','A','A','B'],['A','A','B','A'],['A','B','A','A'],['B','A','A','A'],['A','A','B','B'],['A','B','A','B'],['A','B','B','A'],['A','A','B','C'],['A','B','A','C'],['A','B','C','A'],['B','A','A','C'],['B','A','C','A'],['B','C','A','A'],['A','B','C','D'] base_lm_hidden_dim=32 reasoner_hidden_dim=32 interpreter_hidden_dim=32 share_lm_head=True seed=8
python run.py -m +run_modifier=ReasonerInterpreterGRURunConfig simple_lm_head=True weight_groups=['A','A','A','A'],['A','A','A','B'],['A','A','B','A'],['A','B','A','A'],['B','A','A','A'],['A','A','B','B'],['A','B','A','B'],['A','B','B','A'],['A','A','B','C'],['A','B','A','C'],['A','B','C','A'],['B','A','A','C'],['B','A','C','A'],['B','C','A','A'],['A','B','C','D'] base_lm_hidden_dim=64 reasoner_hidden_dim=64 interpreter_hidden_dim=64 share_lm_head=True seed=7,8
python run.py -m +run_modifier=ReasonerInterpreterGRURunConfig simple_lm_head=True weight_groups=['A','A','A','A'],['A','A','A','B'],['A','A','B','A'],['A','B','A','A'],['B','A','A','A'],['A','A','B','B'],['A','B','A','B'],['A','B','B','A'],['A','A','B','C'],['A','B','A','C'],['A','B','C','A'],['B','A','A','C'],['B','A','C','A'],['B','C','A','A'],['A','B','C','D'] base_lm_hidden_dim=100 reasoner_hidden_dim=100 interpreter_hidden_dim=100 share_lm_head=True seed=7,8



// rn viewing performance on 32 dim with shared lm head weights. Potentially this could be all the negative transfer/most of it. It wasn't. But shokingly there is so much less negative transfer that this setting is doing better than the baseline of just a 32 dim gru lm! 

# Transformer based Quiet-star reimplementation
python run.py -m +run_modifier=QuietSTaRDiscreteTRunConfig use_reasoner=True,False

python run.py +run_modifier=QuietSTaRDiscreteTRunConfig use_reasoner=True seed=8

// trying to add a bunch of compute to make the thing learn with longer rationale.

python run.py +run_modifier=QuietSTaRDiscreteTRunConfig use_reasoner=True max_reasoning_len=50

// Checking the variability of seeds with the use_residual is true, as well as a bunch of compute when use_residual is true.
python run.py +run_modifier=QuietSTaRDiscreteTRunConfig use_reasoner=True use_residual=True seed=8

python run.py +run_modifier=QuietSTaRDiscreteTRunConfig use_reasoner=True use_residual=True max_reasoning_len=50

// TODO: launch 50 length runs on overcap, and lower the policy loss beta to 0.2 in one run.


# change quiet-star to have different position information fed in. and add option add_last_context_token.
python run.py -m +run_modifier=QuietSTaRDiscreteTRunConfig add_last_context_token=True,False

TODO: Re implement nll baseline to ensure it matches the baseline with use_reasoner=False.

python run.py -m +run_modifier=QuietSTaRDiscreteTRunConfig add_last_context_token=True add_surogate_loss_to_last_context_token=True different_eot=True,False last_context_loss_beta=0.0,0.1,0.5


python run.py -m +run_modifier=QuietSTaRDiscreteTRunConfig add_last_context_token=True add_surogate_loss_to_last_context_token=False different_eot=False last_context_loss_beta=0.0 increment_pos_id_for_last_context_token=True,False

// make the change in position encodings optional. This to see if I could implement the change to position encoding within the Mistral model. They had said that the model would ignore the rationale and would fail to generalize to down stream tasks (I think they recorded generalization wrong, and recorded ignore thoughts wrong tho), so its possible that just training for much longer without the mixing head would lead to eventually using the thoughts.
I think generalization should be recorded on the validation negative log likelihood, but they do task accuracy, which I see to be very poor because this task's low accuracy could be due to domain miss calibration or something stupid...

python run.py -m +run_modifier=TLMRunConfig model_hidden_dim=128 model_n_layer=2 model_n_head=2
python run.py -m +run_modifier=TLMRunConfig model_hidden_dim=128 model_n_layer=2 model_n_head=4
python run.py -m +run_modifier=TLMRunConfig model_hidden_dim=256 model_n_layer=2 model_n_head=2
python run.py -m +run_modifier=TLMRunConfig model_hidden_dim=256 model_n_layer=2 model_n_head=4
python run.py -m +run_modifier=TLMRunConfig model_hidden_dim=512 model_n_layer=2 model_n_head=2
python run.py -m +run_modifier=TLMRunConfig model_hidden_dim=512 model_n_layer=2 model_n_head=4

python run.py -m +run_modifier=TLMRunConfig model_hidden_dim=128 model_n_layer=4 model_n_head=2
python run.py -m +run_modifier=TLMRunConfig model_hidden_dim=128 model_n_layer=4 model_n_head=4
python run.py -m +run_modifier=TLMRunConfig model_hidden_dim=256 model_n_layer=4 model_n_head=2
python run.py -m +run_modifier=TLMRunConfig model_hidden_dim=256 model_n_layer=4 model_n_head=4
python run.py -m +run_modifier=TLMRunConfig model_hidden_dim=512 model_n_layer=4 model_n_head=2
python run.py -m +run_modifier=TLMRunConfig model_hidden_dim=512 model_n_layer=4 model_n_head=4


python run.py -m +run_modifier=TLMRunConfig model_hidden_dim=128 model_n_layer=6 model_n_head=2
python run.py -m +run_modifier=TLMRunConfig model_hidden_dim=128 model_n_layer=6 model_n_head=4
python run.py -m +run_modifier=TLMRunConfig model_hidden_dim=256 model_n_layer=6 model_n_head=2
python run.py -m +run_modifier=TLMRunConfig model_hidden_dim=256 model_n_layer=6 model_n_head=4
python run.py -m +run_modifier=TLMRunConfig model_hidden_dim=512 model_n_layer=6 model_n_head=2
python run.py -m +run_modifier=TLMRunConfig model_hidden_dim=512 model_n_layer=6 model_n_head=4


python run.py -m +run_modifier=TLMRunConfig model_hidden_dim=256 model_n_layer=6 model_n_head=2 dropout=0.1

python run.py -m +run_modifier=QuietSTaRDiscreteTRunConfig add_last_context_token=True different_eot=True max_reasoning_len=10,20,40,80,160 seed=7

# testing if trice samples are useful for a just reasoning model GRU setting.
python run.py -m +run_modifier=InterpreterLowVarGRURunConfig trice_samples=10,20,40,80,160
# try training the NLL head less?
// now making it default to have last token and different eot.

// seems trice 2 or 4 is sufficient lol. I'll try with a few more seeds, but this can speed some things up.
python run.py -m +run_modifier=QuietSTaRDiscreteTRunConfig  trice_samples=4 train_nll_num=1,2,4 seed=8,9,10
python run.py -m +run_modifier=QuietSTaRDiscreteTRunConfig trice_samples=10 train_nll_num=1 seed=8,9,10
python run.py -m +run_modifier=QuietSTaRDiscreteTRunConfig trice_samples=10 seed=8,9,10 nll_loss_beta=0.1,0.5,1

python run.py -m +run_modifier=QuietSTaRDiscreteTRunConfig trice_samples=10 max_reasoning_len=5,10,15 seed=10,11,12,13,14 info="\[maxr experiment\]"

python run.py -m +run_modifier=TLMRunConfig info="\[NLL\]" seed=10,11,12,13,14

# train dumber nll models where over fitting ie train eval missmatch doesn't occur to isolate reasoning as just a regularizing process or actaully helpful.


python run.py -m +run_modifier=TLMRunConfig info="\[NLL find smaller model\]" model_hidden_dim=16,32 model_n_layer=1,2 model_n_head=1,2 seed=10


python run.py -m +run_modifier=TLMRunConfig info="\[NLL\]" model_hidden_dim=32 model_n_layer=2 model_n_head=1 seed=10,11,12,13,14

python run.py -m +run_modifier=QuietSTaRDiscreteTRunConfig model_hidden_dim=32 model_n_layer=2 model_n_head=1 trice_samples=2 max_reasoning_len=5,10,15 seed=10,11,12,13,14 info="\[smaller model maxr experiment\]"

python run.py -m +run_modifier=QuietSTaRDiscreteTRunConfig model_hidden_dim=32 model_n_layer=2 model_n_head=1 trice_samples=2 max_reasoning_len=1,2,3,4 seed=10,11,12,13,14 info="\[smaller model maxr experiment\]"

python run.py -m +run_modifier=QuietSTaRDiscreteTRunConfig model_hidden_dim=32 model_n_layer=2 model_n_head=1 trice_samples=2 max_reasoning_len=20,25 seed=10,11,12,13,14 info="\[smaller model maxr experiment\]"

// just make sure the nll runs are the same across using no reasoning and explicitly just the base model.
// I could also try setting policy loss beta to zero and see if the reasoning stuff which is essentially just the base model's ramblings can be ignored.
python run.py -m +run_modifier=QuietSTaRDiscreteTRunConfig model_hidden_dim=32 model_n_layer=2 model_n_head=1 trice_samples=2 max_reasoning_len=1 seed=10,11,12,13,14 info="\[smaller model maxr experiment\]" use_reasoner=False

python run.py -m +run_modifier=QuietSTaRDiscreteTRunConfig model_hidden_dim=32 model_n_layer=2 model_n_head=1 trice_samples=2 max_reasoning_len=1 seed=10,11,12,13,14 info="\[smaller model maxr experiment\]" policy_loss_beta=0

// should experiment with adding residual stream when using reasoning. (seems that training reasoning wasn't even useful for reasoning length=1)

// should try reasoning length=0, so that <sot><eot>c_1 idea for nll can be tested without disruption from the reasoning portion.


python run.py -m +run_modifier=QuietSTaRDiscreteTRunConfig model_hidden_dim=32 model_n_layer=2 model_n_head=1 trice_samples=2 max_reasoning_len=1 seed=10,11,12,13,14 info="\[smaller model maxr experiment residual\]" use_residual=True

python run.py -m +run_modifier=QuietSTaRDiscreteTRunConfig model_hidden_dim=32 model_n_layer=2 model_n_head=1 trice_samples=2 max_reasoning_len=0 seed=10,11,12,13,14 info="\[smaller model maxr experiment\]"

rerun these: they crashed when changing the code:
python run.py -m +run_modifier=QuietSTaRDiscreteTRunConfig model_hidden_dim=32 model_n_layer=2 model_n_head=1 trice_samples=2 max_reasoning_len=4 seed=14 info="\[smaller model maxr experiment\]"
python run.py -m +run_modifier=QuietSTaRDiscreteTRunConfig model_hidden_dim=32 model_n_layer=2 model_n_head=1 trice_samples=2 max_reasoning_len=5 seed=10 info="\[smaller model maxr experiment\]"
python run.py -m +run_modifier=QuietSTaRDiscreteTRunConfig model_hidden_dim=32 model_n_layer=2 model_n_head=1 trice_samples=2 max_reasoning_len=5 seed=11 info="\[smaller model maxr experiment\]"
python run.py -m +run_modifier=QuietSTaRDiscreteTRunConfig model_hidden_dim=32 model_n_layer=2 model_n_head=1 trice_samples=2 max_reasoning_len=5 seed=12 info="\[smaller model maxr experiment\]"
python run.py -m +run_modifier=QuietSTaRDiscreteTRunConfig model_hidden_dim=32 model_n_layer=2 model_n_head=1 trice_samples=2 max_reasoning_len=5 seed=13 info="\[smaller model maxr experiment\]"
python run.py -m +run_modifier=QuietSTaRDiscreteTRunConfig model_hidden_dim=32 model_n_layer=2 model_n_head=1 trice_samples=2 max_reasoning_len=5 seed=14 info="\[smaller model maxr experiment\]"


# train with residual stream with different rationale sizes to confirm if its a capacity issue or if longer rationale aren't helping, or if rationale help at all over a reasonable baseline.

python run.py -m +run_modifier=QuietSTaRDiscreteTRunConfig model_hidden_dim=32 model_n_layer=2 model_n_head=1 trice_samples=2 max_reasoning_len=8,16,32,64 seed=10,11,12,13,14 info="\[smaller model maxr experiment residual\]" use_residual=True

python run.py -m +run_modifier=QuietSTaRDiscreteTRunConfig model_hidden_dim=32 model_n_layer=2 model_n_head=1 trice_samples=2 max_reasoning_len=0,1,5 seed=15,16,17,18,19,20,21,22,23,24,25 info="\[smaller model maxr experiment residual\]" use_residual=True

python run.py -m +run_modifier=QuietSTaRDiscreteTRunConfig trice_samples=2 max_reasoning_len=0,5,10,15 seed=10,11,12,13,14 info="\[maxr experiment residual\]" use_residual=True

// repeating the experiment line above with use_residual=False so I am able to tell if the increase in variance is due to the reduced trice samples or due to the residual connection
python run.py -m +run_modifier=QuietSTaRDiscreteTRunConfig trice_samples=2 max_reasoning_len=0,5,10,15 seed=10,11,12,13,14 info="\[maxr experiment residual\]" use_residual=False


// reasoning false, and check that too long of reasoning doesn't become bad? At the same time check that trice samples are or are not important beyond 2 for long sequences so test with 8
python run.py -m +run_modifier=QuietSTaRDiscreteTRunConfig trice_samples=2 max_reasoning_len=0,5 seed=15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39 info="\[maxr experiment residual\]" use_residual=False

python run.py -m +run_modifier=QuietSTaRDiscreteTRunConfig trice_samples=2,8 max_reasoning_len=30 seed=10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39 info="\[maxr experiment residual\]" use_residual=False

python run.py -m +run_modifier=QuietSTaRDiscreteTRunConfig trice_samples=2 max_reasoning_len=5 seed=10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39 info="\[maxr experiment residual\]" use_residual=False policy_loss_beta=0

python run.py -m +run_modifier=TLMRunConfig info="\[NLL\]" seed=15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39


// see if the plb = 0 still helps with the GRU hopefully to isolate the position ids as the culprit, or something else in the implementation:
python run.py -m +run_modifier=ReasonerInterpreterGRURunConfig simple_lm_head=True weight_groups=['A','A','A','B'] base_lm_hidden_dim=32 reasoner_hidden_dim=32 interpreter_hidden_dim=32 share_lm_head=True seed=1,2,3,4,5,6,7,8,9,10 policy_loss_beta=-1,0,1
python run.py -m +run_modifier=ReasonerInterpreterGRURunConfig simple_lm_head=True weight_groups=['A','B','C','D'] base_lm_hidden_dim=32 reasoner_hidden_dim=32 interpreter_hidden_dim=32 share_lm_head=True seed=1,2,3,4,5,6,7,8,9,10 policy_loss_beta=-1,0,1

// isolate if the reasoning could be differentiated, or if it has to be the base reasoner? 
python run.py -m +run_modifier=ReasonerInterpreterGRURunConfig simple_lm_head=True weight_groups=['A','A','B','C'] base_lm_hidden_dim=32 reasoner_hidden_dim=32 interpreter_hidden_dim=32 share_lm_head=True,False seed=1,2,3,4,5,6,7,8,9,10 policy_loss_beta=-1,0,1


python run.py -m +run_modifier=ReasonerInterpreterGRURunConfig simple_lm_head=True weight_groups=['A','A','A','A'] base_lm_hidden_dim=32 reasoner_hidden_dim=32 interpreter_hidden_dim=32 share_lm_head=True seed=1,2,3,4,5,6,7,8,9,10 policy_loss_beta=-1,0,1


python run.py -m +run_modifier=ReasonerInterpreterGRURunConfig  info="\[NLL\]" simple_lm_head=True base_lm_hidden_dim=32 reasoner_hidden_dim=32 interpreter_hidden_dim=32 share_lm_head=True seed=1,2,3,4,5,6,7,8,9,10 use_reasoner=False


# Train NLL models on new dataset:
python run.py -m +run_modifier=TLMRunConfig info="\[NLL\]" dataset=fw

python run.py -m +run_modifier=TLMRunConfig model_hidden_dim=128,256,512 model_n_layer=2,4,6 model_n_head=1,2,4 dataset=fw

good models to try and for how long?
best model: python run.py -m +run_modifier=TLMRunConfig model_hidden_dim=512 model_n_layer=6 model_n_head=4 dataset=fw
but depth 6 seems unstable
best model: python run.py -m +run_modifier=TLMRunConfig model_hidden_dim=512 model_n_layer=4 model_n_head=2 dataset=fw
best model: python run.py -m +run_modifier=TLMRunConfig model_hidden_dim=256 model_n_layer=4 model_n_head=2 dataset=fw


# GRU experiments which are separate.
python run.py -m +run_modifier=[GRUSeparateScaleExperiment,GLMRunConfig] base_lm_hidden_dim=128,256,512 seed=1,2,3,4,5,6,7,8,9,10

complete the GLMRunConfig to be able to compare a true GRU langauge model. Difference in weight decay, and definitely the parameter initialization. Not sure if there are some changes with the norm or something else subtle tho... I predict no big change, but need to be safe. There was almost no change but yes the weight decay does change the model parameter norm somewhat now. Bigger thing is the learning rate scheduler helps bridge the gap between the performance of AAAA and NLL, which is good to know, I need to test if lr scheduler helps with learning thoughts. it does. Very interesting.

 base_lm_hidden_dim=128 model_n_layer=1 seed=1

python run.py -m +run_modifier=[ReasonerInterpreterGRURunConfig,GRUSeparateScaleExperiment] base_lm_hidden_dim=128 weight_groups=['A','A','A','A'],['A','A','A','B'],['A','A','B','A'],['A','B','A','A'],['B','A','A','A'],['A','B','A','C'],['A','A','B','C'],['A','B','C','D'] seed=1,2,3,4,5,6,7,8,9,10
python run.py -m +run_modifier=[ReasonerInterpreterGRURunConfig,GRUSeparateScaleExperiment] base_lm_hidden_dim=256 weight_groups=['A','A','A','A'],['A','A','A','B'],['A','A','B','A'],['A','B','A','A'],['B','A','A','A'],['A','B','A','C'],['A','A','B','C'],['A','B','C','D'] seed=1,2,3,4,5,6,7,8,9,10
python run.py -m +run_modifier=[ReasonerInterpreterGRURunConfig,GRUSeparateScaleExperiment] base_lm_hidden_dim=512 weight_groups=['A','A','A','A'],['A','A','A','B'],['A','A','B','A'],['A','B','A','A'],['B','A','A','A'],['A','B','A','C'],['A','A','B','C'],['A','B','C','D'] seed=1,2,3,4,5,6,7,8,9,10

// TODO: understand when plb = 0 would be a differentiator, which would indicate to me that it would be worth running it broadly.
python run.py -m +run_modifier=[ReasonerInterpreterGRURunConfig,GRUSeparateScaleExperiment] base_lm_hidden_dim=512 simple_lm_head=True weight_groups=['A','B','C','D'] policy_loss_beta=0 seed=1,2,3,4,5,6,7,8,9,10



python run.py -m +run_modifier=[ReasonerInterpreterGRURunConfig,GRUSeparateScaleExperiment] base_lm_hidden_dim=128 weight_groups=['A','A','A','A'] seed=1


python run.py -m +run_modifier=[ReasonerInterpreterGRURunConfig,GRUSeparateScaleExperiment] base_lm_hidden_dim=128,256,512 weight_groups=['A','A','A','A'],['A','B','C','D'] seed=1,2,3,4,5,6,7,8,9,10

// Difference between NLL and AAAA? I predict that the gap between NLL and the AAAA model will become nothing. Not even so sure why it exists now??? It seems the gap hasn’t gone away, perhaps you get the same benefit that quiet-star gets of more gradient signal on. However this happens even for the ABCD setting and there is no difference between the two’s performance. Thus The explanation doesn’t work. I should try just detaching the gradient from the interpreter. See if the improvement in performance has anything to do with the interpreter input at all.
// No difference between AAAA and ABCD? perhaps there needs to be repeating of the dataset for the reasoning to be useful? (then it could be that the reasoning is acting as a regularizer, which is not really what we wanted to test, because dropout is a better regularizer lol, but perhaps as well, the reasoning could be only useful when the regular model is saturating?)
// lets just quickly check if running for more epochs on the dataset changes the performance for AAAA and ABCD like it had done on the shakespeare datasets
// verify difference in shakespeare exists when big voc? but mostly lr scheduler 
python run.py -m +run_modifier=[ReasonerInterpreterGRURunConfig,GRUSeparateScaleExperiment] info="\[GRU separate scale multiple epochs\]" base_lm_hidden_dim=128 weight_groups=['A','A','A','A'],['A','B','C','D'] seed=1,2,3,4,5,6,7,8,9,10 max_steps=-2
python run.py -m +run_modifier=[GLMRunConfig,GRUSeparateScaleExperiment] info="\[GRU separate scale multiple epochs\]" base_lm_hidden_dim=128  max_steps=-2 experiment_logger=offlinelogger

argue with myself and try to prove thoughts wont work. 
show upper bound with more parameters in the reasoner interpreter/?
I need to understand how my original justification had such good eval number when my number from this past week were so poor ie in 1.8 range instead of 1.6 range. The reason is I downgraded the head to a simple one instead of the MLP I previously had. Also my reasoner was 100 dimensions with my base still at 32 dimensions.


python run.py -m +run_modifier=GLMRunConfig info="\[debug nll 1.6 1.8 discrep v2\]" linear_lm_head=False base_lm_hidden_dim=32 seed=1,2,3,4,5,6,7,8,9,10

python run.py -m +run_modifier=[ReasonerInterpreterGRURunConfig,SameHdim] info="\[debug nll 1.6 1.8 discrep v2\]" simple_lm_head=True weight_groups=['A','B','C','D'] base_lm_hidden_dim=32 share_lm_head=True seed=1,2,3,4,5,6,7,8,9,10 policy_loss_beta=0,1 linear_lm_head=False

// default reasoner hdim is 100, so matching prior test.
python run.py -m +run_modifier=[ReasonerInterpreterGRURunConfig] info="\[debug nll 1.6 1.8 discrep v2\]" simple_lm_head=True weight_groups=['A','B','C','D'] base_lm_hidden_dim=32 share_lm_head=True seed=1,2,3,4,5,6,7,8,9,10 policy_loss_beta=0,1 linear_lm_head=False 

// making base lm match the reasoner size, so I can see if the reasoner model becomes useless again. This wouldn't really say anything if it did become useless, could point to needing to be even higher capacity, or something else. If on the other hand the model seemed useful at this stage (beyond plb = 0) it would give good signal to developing reasoning models with more capacity. But that would be exemplified in quiet-star's combined reasoning model, which seemed to ignore the reasoning for presumed reason of its internal classifier needing to be very good to see any gains from the reasoning.

Aside on explicitly classifying when to reason:
I don't believe I can prove that the problem of thinking is too hard.
Can I say anything about learning where to think being hard? I know how long to think is a halting problem, so undecidable, but I can easily come up with heuristics for simple cases, and say like reasoning for math problems is good (when the reasonings are better than the baseline of not reasoning) A classifier shouldn't have too much trouble saying this. But this classifier would amount to a value head right? we would evaluate the value of the state where the action taken was think, and the value of the state where the action taken was don't think. and then we would take the one with the higher value or some exploration policy or something. The value judgement on reason or not reason it seems like such a simple case...

There is a bigger question I feel, in why the model requires to see the training data more than one epoch. This is clearly not good as it opens the model to over fitting. One may also ask the question of whether this method is at all applicable to scenarios when the model sees the data only once, and my answer isn't so good. (I haven't seen that setting perform well before with the same reasoner parameters and base lm parameters). The model with shared weights does learn to differentiate its reasoning for the fine web dataset even in ABCD model. 

There is almost no difference between ABCD and AAAA (likely because at 128 hidden dimension with reasoner = base lm hidden dim, the extra capacity in the reasoner isn't worth it to use.) I would predict that we would see a difference in performance with the setting of base lm 32 reasoner 128 between ABCD and AAAA, but really this is obvious. The extra capacity will be used because it is more worth it to listen than it costs to be distracted by the reasoning. we want to get really good at knowing when to listen and not listen. As the reasoning from a model with shared parameters for next token prediction will be broadly negative for predicting the correct next token distirbution. This is true for the GRU case tho, so the logic must have a flaw. Would a transformer with some independant reasoner of a higher capacity, with seperate weights learn to effectively use the reasoning language, and not just ignore it? Even with the logic that the majority of reasonings would be bad early on? (yes, because you would get signal for what reasonings are bad, and what are good even minorly, but the reasonings would prove to be insightful as compared to the base lm head just as we saw with the GRU setting. This should also be true for math problems in the general language modeling setting, but perhaps they are so rare that in the 1000 examples from the training set given that the model never really got that signal, and perhaps learned to memorize when it should think and learned to ignore the rest.)

Really, when you train to use bad reasonings, you will get some amount of ignoring learned, and you want to punish/ignore logic that is bad. You just don't want to get rid of logic that is good.

make sure larger reasoner/intepreter yeilds predictable performance improvements before going to fw dataset

python run.py -m +run_modifier=[GRUShakeSeparateScaleExperiment] seed=1,2,3,4,5,6,7,8,9,10 policy_loss_beta=1

// assuming it worked, and moving on to fw dataset and will assume fw works for multiple epochs, so jumping to one epoch of data.

python run.py -m +run_modifier=[GRUShakeSeparateScaleExperiment] seed=1,2,3,4,5,6,7,8,9,10 policy_loss_beta=1 dataset=fw reasoner_hidden_dim=32,100,256 max_steps=2700 use_reasoner=True

python run.py -m +run_modifier=[GRUShakeSeparateScaleExperiment,GLMRunConfig] seed=1,2,3,4,5,6,7,8,9,10 dataset=fw base_lm_hidden_dim=32,100,256 max_steps=2700 


// ensure variable length with various punishments at various sized reasoners yeilds similar results to what is expected in shakespeare setting before moving to test in fine web. SUper cool they have some having fraction emty being 90 or 40%, which could be a nontrivial classifier on the tokens for when to think?

python run.py -m +run_modifier=[GRUShakeSeparateScaleExperiment] info="\[varlen gru\]" seed=4,5,6,7,8,9,10 policy_loss_beta=1 reasoner_hidden_dim=32,100,256 variable_len=True punish_unfinished=0,1e-3,1e-2,1e-1,1

python run.py -m +run_modifier=[GRUShakeSeparateScaleExperiment,GLMRunConfig] info="\[varlen gru\]" seed=1,2,3,4,5,6,7,8,9,10 base_lm_hidden_dim=32,100,256

// find  a model which has good fraction of thoughts, and isn't worse than the baseline 32 GRU
python run.py -m +run_modifier=[GRUShakeSeparateScaleExperiment] info="\[varlen gru\]" seed=1,2,3,4,5,6,7,8,9,10 policy_loss_beta=1 reasoner_hidden_dim=256 variable_len=True punish_unfinished=2e-3,4e-3,6e-3,8e-3

// try a really freaking large hidden representation with a very large punish penalty like 1, and see how the thinking works. ie if it is all very short reasoning or not.
python run.py -m +run_modifier=[GRUShakeSeparateScaleExperiment] info="\[varlen gru\]" seed=1,2,3,4,5,6,7,8,9,10 policy_loss_beta=1 reasoner_hidden_dim=1024 variable_len=True punish_unfinished=1

// construct baseline with just some start of thought end of thought thing, with no reasoning to see how far things which always generate no reasoning should be able to get.
TODO

// experiment with learning rate schedule on near critical values for learning to selectively think. looking for the transition to thinking to be gradual especially in deciding what fraction of thoughts to make empty strings.

python run.py -m +run_modifier=[GRUShakeSeparateScaleExperiment] info="\[varlen gru\]" seed=7,8,9,10 policy_loss_beta=1 reasoner_hidden_dim=256 variable_len=True punish_unfinished=0.004 use_scheduler=True base_lr=0.005

// experiment with just different reasoners without variable length to see if they have similar poor performance to the ones which can ignore their reasoning
python run.py -m +run_modifier=[GRUShakeSeparateScaleExperiment] info="\[varlen gru\]" seed=1,2,3 policy_loss_beta=1 reasoner_hidden_dim=256,512,1024,2048,4096 variable_len=False
// result is a very mixed bag. some better some worse. but big thing is some worse than even the just GRU zero len reasoning baseline, which is pointing to something odd with negative transfer potentially from the shared lm head.

// need to run fine web finetuning to try to check if problem of base model out performing the reasoning adjusted model holds as reasoning models scale. with and without shared heads. and then base model ofc. (also base model with zero reasoning, which I can denote with reasoner_hidden_dim=1, as an easy ish tagging mechanism.)

// experimental settings with shared and non shared lm
python run.py -m +run_modifier=[GRUVarLenScaleExperiment] seed=1,2,3,4,5,6,7,8,9,10 policy_loss_beta=1 reasoner_hidden_dim=32,256,512,2048,4096 dataset='shake' share_lm_head=True,False
// quick debug if trice needs to be larger than 2.
python run.py -m +run_modifier=[GRUVarLenScaleExperiment] seed=1 policy_loss_beta=1 reasoner_hidden_dim=256 dataset='shake' share_lm_head=True,False trice_samples=10,2,15
// seems trice 10 works, but also shared lm head seems necessary, weird. I'll make the head more complex, and rerun it. will just use the same setting of linear_lm_head=False
python run.py -m +run_modifier=[GRUVarLenScaleExperiment] seed=1,2,3,4,5,6,7,8,9,10 policy_loss_beta=1 reasoner_hidden_dim=256 dataset='shake' share_lm_head=False trice_samples=10 info="\[varlen gru v0.51\]"


// control: 0 len reasoning, and gru baseline for shake and fw:
python run.py -m +run_modifier=[GRUVarLenScaleExperiment] seed=1,2,3,4,5,6,7,8,9,10 dataset='shake' max_reasoning_len=0 reasoner_hidden_dim=1 use_reasoner=True,False
python run.py -m +run_modifier=[GRUVarLenScaleExperiment] seed=1,2,3,4,5,6,7,8,9,10 dataset='fw' max_reasoning_len=0 reasoner_hidden_dim=1 use_reasoner=True,False max_steps=2700

// TODO: add var len to experiments.

// testing reasoner dimension more extensively.
python run.py -m +run_modifier=[GRUVarLenScaleExperiment] seed=1,2,3,4,5,6,7,8,9,10 policy_loss_beta=1 reasoner_hidden_dim=16,32,64,128 dataset='shake' trice_samples=10 info="\[varlen gru v0.52 diff reasoners diff lm_heads 10 trice\]"
python run.py -m +run_modifier=[GRUVarLenScaleExperiment] seed=1,2,3,4,5,6,7,8,9,10 policy_loss_beta=1 reasoner_hidden_dim=512,768,1024 dataset='shake' trice_samples=10 info="\[varlen gru v0.52 diff reasoners diff lm_heads 10 trice\]"
// redo control just in case. No change.
python run.py -m +run_modifier=[GRUVarLenScaleExperiment] seed=1,2,3,4,5,6,7,8,9,10 policy_loss_beta=1 max_reasoning_len=0 reasoner_hidden_dim=1 dataset='shake' trice_samples=10 info="\[varlen gru v0.52 diff reasoners diff lm_heads 10 trice\]"
// more extensive testing has yeilded confusing results. The model seems to improve up to 256. Plateuing at 320. Testing now for 512 768 and 1024
// expect the model gets worse at 512 and even worse as you go up. Look also at the mean reasoning entropy. I need to fix the mean reasoning entropy calculation. Maybe look at max instead.

// check if increasing trice samples for the larger models also helps. specifically look at 512: seems the answer is It is too small and could easily be variance. The trice samples did not help in this scenario. perhaps for 1024...?
python run.py -m +run_modifier=[GRUVarLenScaleExperiment] seed=1,2,3,4,5,6,7,8,9,10 policy_loss_beta=1 reasoner_hidden_dim=512 dataset='shake' trice_samples=20 info="\[varlen gru v0.52 diff reasoners diff lm_heads 10 trice\]"

// should move fw. check 32,128,256,385,512,768,1024. before launching will check locally.
python run.py -m +run_modifier=[GRUVarLenScaleExperiment] seed=1,2,3,4,5,6,7,8,9,10 policy_loss_beta=1 reasoner_hidden_dim=32,128,256,385,512,768,1024 dataset='fw' max_steps=2700 trice_samples=10 info="\[varlen gru v0.52 diff reasoners diff lm_heads 10 trice\]"

//before I launch these actually. The problem of 1024 performing worse than the baseline when it produces no thought needs to be addressed. It is insane. try 0 max len reasoning, and 1024 params.
python run.py +run_modifier=[GRUVarLenScaleExperiment] seed=1 policy_loss_beta=1 max_reasoning_len=0 reasoner_hidden_dim=1024 dataset='shake' trice_samples=10 info="\[varlen gru v0.52 diff reasoners diff lm_heads 10 trice\]"
// oohh still has the problem. Must be something with the gradient. could be that the interpreter is causing the basline to shift, and then the base lm has to try to adjust. This would be aweful.
// both models try to solve the same problem with no ability to comunicate. both models must get exactly the same gradient acutally, so that makes sense,but why doesn't the interpreter approach the average?
// how to address this while keeping a fair baseline? well, whatever I do I can always have the baseline of 0 length reasoning, so that works.
// then I can simply mask one's contribution conditioned on both. so a mixing head kind of exactly... some MLP taking in the concatenation of both the interpreter and base model.

// Check these: (testing with mixing head. Maybe the spikes that it can have in the gradient are fine as long as it settles after a point. But, need to check that the regular sizes of reasoner still are beneficial with this mixinghead change.)
python run.py -m +run_modifier=[GRUVarLenScaleExperiment] seed=1,2,3,4,5,6,7,8,9,10 policy_loss_beta=1 reasoner_hidden_dim=32,128,256 dataset='shake' trice_samples=10 info="\[varlen gru v0.52 diff reasoners diff lm_heads 10 trice\]"

// mixing head seems beneficial across the board, but I am concerned that the baseline isn't able to achieve zero, which is why it looks weaker, so I was thinking to make the mixing head explicitly a sigmoid operation... should check my intuiion because this seems to work better actually.
python run.py -m +run_modifier=[GRUVarLenScaleExperiment] seed=1,2,3 policy_loss_beta=1 max_reasoning_len=0 reasoner_hidden_dim=1,1024 dataset='shake' trice_samples=10 info="\[varlen gru v0.53 diff reasoners diff lm_heads 10 trice\]"
// maybe this also helps with the peaks which I saw in one of the seeds from 1024. Need to test how baselines are effected by both mixing strategies. for 1024 and 2048

// feedback loop with mixing head of having empty reasonings be easy to assign 0 mixing, which performs better, and then having only zero length reasonings be encouraged, and you never get to try to improve the reasonings. a fix for this is to make the mixing head an ema with starting value 0.6, and then the model gets time with some attention, but eventually, perhaps the model can get to decide how it wants to attend to itself.

python run.py -m +run_modifier=[GRUVarLenScaleExperiment] seed=1,2,3,4,5 policy_loss_beta=1 max_reasoning_len=0 reasoner_hidden_dim=1,256,1024,2048 dataset='shake' trice_samples=10 info="\[varlen gru v0.53 diff reasoners diff lm_heads 10 trice\]" mix_interpeter_base_lm=1,2

// gotta make sure the performance of 0 length reasoning models aren't better than base models.
// For some reason my 0 length reasoning with no reasoner is better than my base model alone???? All it should do is add noise? but somehow the training loss is better??? but as the reasoner interpreter size increases, the  model gets worse, so it doesn't seem to be. checking that the interpreter representation is consistent across training, so it is just a constant thing. I think the bias is essentially being added to the input tho, so the non linearity is like a slightly deeper model. Its actually slightly more complicated with a kind of gating, but the gating only works without sigmoid. for whatever reason? can get it to work with sigmoid if add an entropy term.

// try now on fine web. make sure to test 1 length reasoning with mix 1, so that the baseline is fair. Maybe I'll have to add the slow changing ability to mix 1.

//controls:
python run.py -m +run_modifier=[GRUVarLenScaleExperiment] seed=1,2,3,4,5,6,7,8,9,10 max_reasoning_len=0 reasoner_hidden_dim=1,32,256,1024,2048 dataset='fw' max_steps=2700 info="\[varlen gru v0.53 diff reasoners diff lm_heads 10 trice\]" mix_interpeter_base_lm=1,2
// test cases:
python run.py -m +run_modifier=[GRUVarLenScaleExperiment] seed=1,2,3,4,5,6,7,8,9,10 max_reasoning_len=10 reasoner_hidden_dim=32,128,256,512,1024 dataset='fw' max_steps=2700 info="\[varlen gru v0.53 diff reasoners diff lm_heads 10 trice\]" mix_interpeter_base_lm=1,2
// before launching them, should just test with length 10 on mix 1 if it needs some form of stabalization to prevent 0 length from being learned too quickly.
should try 1024? with mix =1,2 also 
python run.py -m +run_modifier=[GRUVarLenScaleExperiment] seed=1 max_reasoning_len=10 reasoner_hidden_dim=256,1024 dataset='fw' max_steps=2700 info="\[varlen gru v0.53 diff reasoners diff lm_heads 10 trice\]" mix_interpeter_base_lm=1,2

// seems 0.6 start isn't working for 1024 with mixing sigmoid. test locally
// look at 256 the quiet-star policy loss towards the end of both mix=1 and 2. 
// I should clip the gradient or bound the reward. It seems some extreme difference can yeild very poor optimization for my mix = 1 method.
// cliping didn't work out. trying bounding the reward. waiting for debug thing to be done then will try.
bound of +/- 0.1 seems associated with quiet_star_policy loss - 0.01
shit did't work for bounding. there is a growth of zero len reasonings, and a quick death. sometimes the growth is full 0 len both are degenerate to zero policy loss
For mixing head without sigmoid, the coeficient seems not to be near zero for 1024, for 256 it seems to be around -20
is it a problem that in RL with Trice sequences which are common are essentially weighted higher even if their reward is less, just because they are sampled more? this is fine, but potentially if there are better alternatives to the null token it might be that the null token is just the safer bet, as all other attempts yeild poor reasoning almost everywhere. and the zero reasoning gives a good base.
make it beneficial to use the component that the reasoning is a part of like having the residual which you combine with the base model be some MLP of the base and the intepreter
going to add an entropy encouragement term so that same single token is a hole we can get out of. and also if ever we are in 10 token domain we can find our way to eot token. The model is degenerating to a single token output, due to ease of modeling and using the MLP head. My problem still stands that this will just lead to the reasoning being ignored somewhere else if not from the reasoning generator itself.
remember to make the entropy a perminant fixture if it works.
contingent right now on pad token being in last place

python run.py -m +run_modifier=[GRUVarLenScaleExperiment] seed=1 max_reasoning_len=10 reasoner_hidden_dim=32,256,1024 dataset='fw' max_steps=2700 info="\[some experiment varlen gru v0.53 diff reasoners diff lm_heads 10 trice\]" entropy_encouragement_coef=10,100

didn't stop the entropy from going to zero in the 256 case. for ent enc = 1, but for 10 and 100 it seemed to do some cool stuff.
python run.py -m +run_modifier=[GRUVarLenScaleExperiment] seed=1 max_reasoning_len=10 reasoner_hidden_dim=32 dataset='fw' max_steps=2700 info="\[some experiment varlen gru v0.53 diff reasoners diff lm_heads 10 trice\]" entropy_encouragement_coef=1

will now test entropy encouragement with mix 1 mix 2, and various reasoning sizes. Also ensure the baselines are fair comparisons by having untrained reasoning portions (plb = 0, and no need for entropy variation. will stick with 0 len reasoning with reasoner/interpreters of the same size but will also launch this plb=0 idea with 10 len reasoning for one subset 32 - 256 for both mix 1 and mix 2 just to try it out. Can run more in the morning) plb = 0 wasn't too different, for 256, so just stick with 0 len reasoning baseline.
// standard baselines 
python run.py -m +run_modifier=[GRUVarLenScaleExperiment] seed=1,2,3,4 max_reasoning_len=0 reasoner_hidden_dim=1,32,128,256,512,1024 info="\[varlen gru v1 ent m1m2\]" mix_interpeter_base_lm=1,2
<!-- // new idea plb = 0 baselines same reasoning length tho.
python run.py -m +run_modifier=[GRUVarLenScaleExperiment] seed=1,2,3,4 max_reasoning_len=10 reasoner_hidden_dim=256 info="\[varlen gru v1 ent m1m2\]" mix_interpeter_base_lm=1,2 policy_loss_beta=0.0 -->
// test
python run.py -m +run_modifier=[GRUVarLenScaleExperiment] seed=1,2,3,4 max_reasoning_len=10 reasoner_hidden_dim=32,128,256,512,1024 info="\[varlen gru v1 ent m1m2\]" mix_interpeter_base_lm=1,2 entropy_encouragement_coef=0,1,4,16,64,128

Experiment with reasoning length penalty.
python run.py -m +run_modifier=[GRUVarLenScaleExperiment] seed=1,2,3,4 reasoner_hidden_dim=256 info="\[varlen gru v1 ent m1m2 rlen\]" mix_interpeter_base_lm=1,2 entropy_encouragement_coef=64,128 length_penalty=0.0001,0.00001,0.000001 max_reasoning_len=100

Redo some entropy experiments as they weren't correctly scaled.
python run.py -m +run_modifier=[GRUVarLenScaleExperiment] seed=1 max_reasoning_len=10 reasoner_hidden_dim=256 info="\[varlen gru v1 ent m1m2 ent\]" mix_interpeter_base_lm=1 entropy_encouragement_coef=0.0001,0.0004,0.0016,0.0032,0.0064,0.0096,0.0128

python run.py -m +run_modifier=[GRUVarLenScaleExperiment] seed=1 max_reasoning_len=50 reasoner_hidden_dim=256 info="\[varlen gru v1 ent m1m2 ent\]" mix_interpeter_base_lm=1 entropy_encouragement_coef=0.0001,0.0004,0.0016,0.0032,0.0064,0.0096,0.0128 length_penalty=0.0001,0.001,0.0005,0.00005

// plan is to move to AAAA setting and get closer to transformers. Took a detour back to GRUs because this was the setting where adding reasoning was helpful. Have been trying to make variable length reasoning work in GRU setting, but doesn't seem like variable length is doing the thing I want it to do, which is to make reasoning for next token prediction a rare thing so that the classifier for how useful reasoning is wouldn't have to work as hard. Consider scaling up the reasoning length penalty term for max_reasoning_len=10.
// On the note of the reasoning classifier (ie mixing head) needing to be quite good to ignore or not ignore thoughts, I would move to have the gating mechanism correct so that the idea of ignoring a thought can be nicely codified and it facilitates explicitly the process of ignoring reasoning, which I believe can be easily done in a high capacity transformer, so this is a hurdle I will have to jump though eventually. And it is easier to address in GRUs. 
// move on to same sized reasoning as base lm, and then eventually to AAAA. Would first want to see that beyond 32 base GRU the reasoning with different dimensions still helps.

// A possible explaination for why using the extra compute especially early on in training could be bad, would be that the hidden representation being dedicated to the reasoning component wouldn't be serving every next token unlike the reasoning learned within the RNN's hidden state. If you were to extrapolate this idea to transformers, you would want the parallelly generated thoughts to attend to eachother.
// This being said, using a pretrained model is the plan so as to leverage the bias over reasonings which would be useful for data efficiency.

// scaling base and interpreter (seems for larger base model reasoning is being ignored more with mix=1. For mix=2 it was always ignored.)
python run.py -m +run_modifier=[GRUBaseInterpreterScaleExperiment] seed=1,2,3 base_lm_hidden_dim=32,64,128,256 info="\[gru biscale v0\]" mix_interpeter_base_lm=1,2
// control
python run.py -m +run_modifier=[GRUBaseInterpreterScaleExperiment] seed=1,2,3 max_reasoning_len=0 base_lm_hidden_dim=32,256 info="\[gru biscale v0\]" mix_interpeter_base_lm=1,2

python run.py -m +run_modifier=[GRUBaseInterpreterScaleExperiment] seed=1 max_reasoning_len=10 base_lm_hidden_dim=32,256 info="\[gru biscale v0\]" mix_interpeter_base_lm=1,2 entropy_encouragement_coef=0.000001,0.000005,0.00001,0.00005

// should test Quiet-star as a baseline for SFT data efficiency before I invest too much time fixing a problem that can be easily fixed with some hyper parameters on quiet-star.
// Ie I should build out my eval suite and test some baselines. Because SFT data efficiency doesn't technically require nll reduction on a continous pretraining set (depending on how I measure my success. like with CSQA and GSM8K the tokens we record logprobs over are also the tokens helped by thoughts)

// move back from simple head to concat head to try to get away from the gating architecture.
python run.py -m +run_modifier=[GRUConcatExperiment] seed=1,2,3,4,5 base_lm_hidden_dim=32,256 reasoner_hidden_dim=256 max_steps=5000 max_reasoning_len=0,10

python run.py -m +run_modifier=[GRUConcatExperiment] seed=1,2,3,4,5 base_lm_hidden_dim=64,128 reasoner_hidden_dim=256 max_steps=5000 max_reasoning_len=0,10

python run.py -m +run_modifier=[GRUConcatExperiment] seed=1,2,3,4,5 base_lm_hidden_dim=32,64 reasoner_hidden_dim=512 max_steps=5000 max_reasoning_len=0,10

// move to trying longer reasonings?
python run.py -m +run_modifier=[GRUConcatExperiment] seed=1,2,3,4,5 base_lm_hidden_dim=32,64 reasoner_hidden_dim=256 max_steps=5000 max_reasoning_len=50,100

// try to fix the model performance degrading as reasoner/interpreter increases in size.
python run.py -m +run_modifier=[GRUConcatExperiment] seed=1,2,3,4,5 info="\[gru concat scale v0\]" reasoner_hidden_dim=32,128,256,512 max_reasoning_len=0,10

python run.py -m +run_modifier=[GRUConcatExperiment] seed=1 info="\[gru concat scale pg1 v0\]" reasoner_hidden_dim=32,128,256,512 max_reasoning_len=0,10 parameter_groups=1

// work with reasoners of the same dim as
python run.py -m +run_modifier=[GRUConcatExperiment,SameHdim] seed=1 info="\[gru concat same v0\]" base_lm_hidden_dim=32 max_reasoning_len=10 entropy_encouragement_coef=0,0.0001,0.0005,0.001
python run.py -m +run_modifier=[GRUConcatExperiment,SameHdim] seed=1 info="\[gru concat same v0\]" base_lm_hidden_dim=32 max_reasoning_len=0


// I don't think the Dropout experiments are going to work for a variety of reasons. I would much rather try the mixed training of different sized base models. what if I make dropout applied lesser and it could result in some really poor models. Models degenerate to 0 entropy. Perhaps more related to 32 reasoners not being high capactiy enough to even help anyone?
python run.py -m +run_modifier=[GRUConcatExperiment,SameHdim] seed=1 info="\[gru concat drop v1\]" base_lm_hidden_dim=32,256 max_reasoning_len=0,10 neuter_dropout_base=0.875,0,0.2,0.5

python run.py -m +run_modifier=[GRUConcatExperiment,SameHdim] seed=1 info="\[gru concat drop v1\]" base_lm_hidden_dim=256 max_reasoning_len=10 neuter_dropout_base=0.875

// doesn't seem that dropout accurately simulates a weak model which would always use the reasoner... There are also other explainations potentially something related to learning rate, and when the reasoner learnes to map to determinsitic. seems also the base model's capacity isn't an indicator of weather the reasoning will learn to be used. Sometimes 256r isn't used for 64b, but is for 128b and for 32b. Some seeds on 256 lead to using reasoning, some do not.


// making base models so I can move to using pretrained models. Moving now to better understand when reasoning is induced or not.
python run.py -m +run_modifier=[GRUConcatExperiment] seed=1,2,3,4,5 info="\[gru base models v0\]" base_lm_hidden_dim=32,64,128,256 reasoner_hidden_dim=256 max_reasoning_len=0
python run.py -m +run_modifier=[GRUConcatExperiment] seed=0 info="\[gru base models v0\]" base_lm_hidden_dim=2,4,8,16 reasoner_hidden_dim=256 max_reasoning_len=0
python run.py -m +run_modifier=[GRUConcatExperiment] seed=0 info="\[gru base models v0\]" base_lm_hidden_dim=32 reasoner_hidden_dim=256 max_reasoning_len=10 infer_pretrained_base=True
// interesting that with pretrained 32b even the 256r gets ignored. This points directly to needing build up of reasoning, because the 32b model does typically work. I will double check the random seeds for 32b with 256r. Yes 100 % of seeds failed with pretrained model, whereas 0% failed in other setting. Should check that some base model (2,4,8,16) would improve with the 256 reasoner dispite being pretrained. 

// also want to profile the performance of a reasoning model without a base model.
python run.py -m +run_modifier=[GRUConcatExperiment] seed=0,1,2,3,4 info="\[gru base models v0\]" reasoner_hidden_dim=256 max_reasoning_len=10 use_base_lm=False base_lm_hidden_dim=1
// I am confused about the architectures which could have benefited massively from the reasoning models but didn't. Especially the dim 1 and no base model?? some of the larger models like 4 8 16 32 make more sense to me from the learning step by step  with mix curriculum idea not being executed nicely, but really for no base model, shouldn't this be the easiest setting? (maybe the prior over the langauge modeling tokens is even safer than learning to think and having poor thoughts early on. In which case how does the 2 dim model constistantly learn???) WHAT CAUSED THE LANGUAGE MODEL IN SHAKESPEARE TO LEARN SO CONSISTANTLY??

present experiments clearly. Want 128b just to confirm at the high end.
Want also models which don't use pretrained to contrast? (what point will this be making? That the pretraining sets the models up poorly for learning? Well, more that the co-training is important for joint learning especially for the reasoning model.. (didn't work for dim = 1, but definitely works for dim=32 (maybe recheck this. again breifly just to make sure I haven't screwed anything up. Checked was right. It could kind of make sense that models which depend on their inputs will learn to rely on new signal in their input, and the dim 1 is too small to allow for co training. This would be an interesting point to want to make. Could just go with this assumption unless it becomes important that I thoroughly test it, then the experiment would be joint training at these small scales. And to address the idea that the joint training is only beneficial when the the base model would learned to be used before the reasoning model would learned to be ignored for dim1 base model pretraining shows the model learns to use its reasoning around step 300, but if you train with just reasoning, the reasoning is completely ignored after step 50. This could be the multiple layers of the network causing poor signal to be sent through to the interpreter representation. Test this with turning the linhead=True and use_base_lm=False. didnt work.)))

python run.py -m +run_modifier=[GRUConcatExperiment] seed=0,1,2,3,4 info="\[gru base models v0\]" reasoner_hidden_dim=256 max_reasoning_len=10 use_base_lm=False linear_lm_head=True base_lm_hidden_dim=1  // did't seem to be any better. model still went to no reasoning quickly. which just means my intuition for the solution wasn't right. which i got from quiet stars comment that a complex lm head didnt work very well. this comment could still be true, but at the no base model experiments this doesnt seem to be the solution. very awesome result!!! the length in this case sometimes went to zero!!! this is unique amoung all the models which didnt learn to use reasoning. this model chose the most optimal way not to use reasoning. at least sometimes. 

The question is now how to optimally design a schedule of using a good base model vs medium vs bad base model. Letting the reasoner learn along with the base model seems locally OK, but clearly there could be something missing, because there is no reason that this should be optimal. If for example the base model learns much faster there is no reason to expect the reasoner model not to be ignored. This feels like a second order problem again. Like learning when to think to be most optimal for future thought learning. Is there a way to do a locally greedy solution as was found with thought length penalty, which essentially made thoughts only generate for areas it was confident thoughts would help in, and then slowly expand to areas where it now has more confidence that thoughts could be useful in? 
Question of when to think is now on when to think with what base model level, and it should expand slowly to think in harder base model levels. Hopefully the model could just learn this. To have zero length reasoning in the cases when the base model is large early on, and then to have long reasoning for cases where the base model is small. We would be looking to make sure that the reasoning doesn't negatively transfer too much early on, but still has the ability to positively transfer for moving on to a larger base model. Could make this explicit with different Lora Adapters per base model size, but same rest of the model, or maybe even different adapter heads, but the same first four layers or maybe even different first 2 layers and last two layers but the same middle layers. Hopefully with a large enough capacity I wouldn't have to think about this problem very carefully and would just be able to solve it with one set of parameters. There is no question that the problem warents knowledge of what base model you are interacting with. So to give this knowledge especially early on we can just do modifying the start of thought embedding seed. And to make interpreter know its job we modify the interpreters starting representation, and to have the LM head know its job more seriously than is allowed by the reasoning alone, we can add some embedding that is fed into the MLP. There are a bunch of tricks it seems... Many places to fail. Very hard to find success, but confident that success will be had. Is it necessary for the broader goal of data efficient SFT? (well, we haven't started messing around with pretrained reasoners, but I am imagining that they will have a problem of being ignored. Should try this first.)

// Pretrained reasoners
python run.py -m +run_modifier=[GRUPretrainReasonerExperiment] seed=0 info="\[gru pretrain v0\]" base_lm_hidden_dim=256

python run.py -m +run_modifier=[GRUConcatExperiment] seed=0,1,2,3,4 info="\[gru pretrain v0\]" base_lm_hidden_dim=32 reasoner_hidden_dim=256 infer_pretrained_reasoner=True,False


// be able to inspect functional reasonings from settings which seem not to use variable len. (perhaps will be able to train the places not to do reasoning into a base model.)
python run.py -m +run_modifier=[GRUConcatExperiment] seed=0,1 info="\[gru pretrain v0\]" base_lm_hidden_dim=32 reasoner_hidden_dim=256 infer_pretrained_reasoner=False hydra.launcher.partition='kargo-lab'
python run.py +run_modifier=[GRUConcatExperiment] seed=0 info="\[gru pretrain v0\]" base_lm_hidden_dim=32 reasoner_hidden_dim=256 infer_pretrained_reasoner=False

// try some run with 0 plb to test if the reasoning can be useful dispite not being trained?
python run.py -m +run_modifier=[GRUConcatExperiment] seed=0,1,2,3,4 info="\[plb0 gru pretrain v0\]" base_lm_hidden_dim=32 reasoner_hidden_dim=256 infer_pretrained_reasoner=True infer_pretrained_base_reasoner=True policy_loss_beta=0

python run.py -m +run_modifier=[GRUConcatExperiment] seed=0,1,2,3,4 info="\[base too gru pretrain v0\]" base_lm_hidden_dim=8 reasoner_hidden_dim=256 infer_pretrained_reasoner=True infer_pretrained_base_reasoner=True
// should check if they used the thing, but I don't have good baseline. I'm not going to launch it. Also without 4 working or 32 working, it would be a really hard schedule to justify. And it for sure isn't going to work amazingly, could try detokenizing the reasoning if I really want to check, but nahh. maybe? check if seed 2 of hiden state size 4 is working better than random init reasoning??? This would say that the reasoning is benefiting from having the language bias (which should be expected!!! fucking shit brooo) shouldnt base reasoner benefit a lot from pretrained init??
python run.py -m +run_modifier=[GRUConcatExperiment] seed=0,1,2,3,4 info="\[base only gru pretrain v0\]" base_lm_hidden_dim=32 reasoner_hidden_dim=256 infer_pretrained_reasoner=False infer_pretrained_base_reasoner=True



# Prompt ideas
Prompting to convey to the langauge model the the job of the reasoner, and the job of the interpreter. Specifically how the language model should leverage its world knowledge and ability to compute intermediate results with variable length allocation to control its distirbution over tokens for the purpose of next token prediction.

Terms:
 - Token: a single unit of prediction. Commonly a bpe tokenizer is used, which first splits up a sentence into pretokens which consist of words and potentially punctuation, and then these pretokens are split into strings of continuous characters which are determined by the most commonly coocuring continuous characters. Here is an example: "Hello, world. What is up?": 9707, 11, 1879, 13, 3555, 374, 705, 14254, 21512, 30 => ['Hello', ',', 'world', '.', 'What', 'is', 'up', 'Dog', 'gie', '?']. Doggie represents one word but is broken into two tokens because of it's uniqueness in the training corpus for the tokenizer.
 - Forward pass: With a transformer model a single forward pass from a decoder, which is what a typical language model is, consists of a finite number of layers in which each token communicates its representation from it's beginning representation which contains it's sentence position information and a token specific embedding representation. As the tokens pass through a layer, their representations are updated through self attention along with position wise MLP interleaved between residual connections. At the end of the transformer after all the layers have been passed through, the representation of the token at the end of the sequence is used to create a distribution over next tokens. The per token compute is limited by the number of layers, but if tokens within its context contain effectively cached intermediate results, the forward pass may make use of these representations in its early layers to effectively deepen its computation.
Consider revisiting Claude's recommendations. Clarity is important. Definition quality may be the difference between working and not working?




Thank you for reading through my initial prompt, in reading your thought process and response it has become clear that I haven't provided enough clarity for what motivates the creation of this prompt, which may help you better assist me to revise this prompt. My problem setting is based off of "Quiet-Star: Language models can teach themselves to think before speaking", who's abstract is the following:
When writing and talking, people sometimes pause to think. Although reasoning-focused works have often framed reasoning as a method of answering questions or completing agentic tasks, reasoning is implicit in almost all written text. For example, this applies to the steps not stated between the lines of a proof or to the theory of mind underlying a conversation. In the Self-Taught Reasoner (STaR, Zelikman et al. 2022), useful thinking is learned by inferring rationales from few-shot examples in question-answering and learning from those that lead to a correct answer. This is a highly constrained setting – ideally, a language model could instead learn to infer unstated rationales in arbitrary text. We present Quiet-STaR, a generalization of STaR in which LMs learn to generate rationales at each token to explain future text, improving their predictions. We address key challenges, including 1) the computational cost of generating continuations, 2) the fact that the LM does not initially know how to generate or use internal thoughts, and 3) the need to predict beyond individual next tokens. To resolve these, we propose a tokenwise parallel sampling algorithm, using learnable tokens indicating a thought’s start and end, and an extended teacher-forcing technique. Encouragingly, generated rationales disproportionately help model difficult-to-predict tokens and improve the LM’s ability to directly answer difficult questions. In particular, after continued pretraining of an LM on a corpus of internet text with Quiet-STaR, we find zero-shot improvements on GSM8K (5.9%→10.9%) and CommonsenseQA (36.3%→47.2%) and observe a perplexity improvement of difficult tokens in natural text. Crucially, these improvements require no fine-tuning on these tasks. Quiet-STaR marks a step towards LMs that can learn to reason in a more general and scalable way.

Mentioned in the paper, and what prompts me to start further work in this direction is the observation that after 100 steps, the process begins to perform worse in terms of accuracy on GSM8k. Upon training a model with their codebase, I also observed that the Negative log likelihood increased loss during the first 100 steps, which is surprising given the accuracy increase, but the NLL loss subsequently decreases however the accuracy decreases along with it! This implies that the accuracy they record isn't related to the next token prediction task they train with. As you pointed out in your response, there is a miss match in pre-training with and without these <think> ... </think> tokens in context. And their work attempts to communicate to the model that it is thinking only by changing the initialization of the special tokens corresponding to the <think> and </think> tokens to the encoding of '---' present in the pre-trained model's embedding layer which was fine-tuned to achieve their results.

In trying to address the mismatch with standard pre-training you mention, I realize that I have to revise the training concept behind my original prompt. The model will only ever have 1 think tag in its context when modeling for the next token (the think tag may contain something along the following sequence if the model deems a thought not to be necessary at the given position. "I will stick with my default distribution").

The language model will not be providing the location for where to think by inserting a thought tag where it believes it should think, but will instead be deciding when to think by either using the sequence "I will stick with my default distribution" within a think tag or actually producing a thought meant to meaningfully change the distribution over tokens. The reason that this is more natural is because I am working with a very artificially constructed scenario, where it is very easy to come up with clear rewards for a single thought by only ever getting loss after a single thought has been produced. Using multiple thoughts in a single context will be explored in future experiments if it makes sense to do so based on the success of the single thought experiments.

update lop p(y|z,x) only when better than log p(y | z=null thought, x), thus no thought ignoring is learned. 

Making the first sentence, what you want: Advice from Grace
You are a state of the art pretrained language model. Your job is to predict the next token. I am giving you the ability to think for a variable number of forward passes before producing a distirbution over your next token predictions.
Here are some definitions useful for understanding the task of language modeling conceptually:
- Token: A single unit of prediction. Commonly a BPE (Byte-pair-Encoding) tokenizer is used, which first splits up a sentence into pretokens consisting of words and punctuation, then these pretokens are split into strings of continuous characters. The exact choice of which characters is determined by the most commonly coocuring continuous characters. Here is an example: "Hello, world. What is up Doggie?": 9707, 11, 1879, 13, 3555, 374, 705, 14254, 21512, 30 => ['Hello', ',', 'world', '.', 'What', 'is', 'up', 'Dog', 'gie', '?']. Doggie represents one word but is broken into two tokens because of it's uniqueness in the training corpus for the tokenizer.
- Forward pass: For a decoder only transformer mode, which is the style of language model that you are, a single forward pass consists of a finite number of layers. 
    1. Before any of the attention/mlp layers, the input tokens are encoded and combined with their positional information.
    2. In the attention layers each token communicates its representation to other tokens so that they may build a useful contextual representation from the rest of the tokens in the context.
    3. At the end of the transformer after all the layers have been passed through, the representation of the token at the end of the sequence is used to create a distribution over next tokens. 
A fundamental limitation to this process is that the the per token compute is limited by the number of layers. This is the reason you may need a variable length number of forward passes. Upon completing a forward pass you emit a distribution over tokens, this distribution is then sampled, and if contained within a thought, this sampled token can be attended to for your next forward, effectively caching the compute that forward pass did to be easily attended through the token's representation.
Here are some scenarios where you should use thinking:
Scenario 1: You are predicting the next tokens of a math text book:
"[Prior tokens explaining the concepts of Statistics]...
Q26 ...
A) 15
B) 20
C) 10
D) 5
...
(Answer to even numbered problems in the back of the book)
Q26<think> [Reason about the answer to the problem]... Suggestion that I expect A to be the answer, but I can't be too confident as my first attempt gave the answer as B, so I should adjust my distribution over tokens to reflect approximately a 75% confidence in A to a 25% confidence in B with a balance over other possible answer tokens to express some uncertainty in the answer being either of these two</think>B"
Here, we see thinking used to express distributional uncertainty over tokens. It can be inferred that the reasoning wasn't as successful as we may have wanted it to be because the true next token was not in fact A, but was instead B. This being said the reasoning over the next token still likely helped by reducing what would have likely been an uncertain prior in the likelihood over A, B, C, or D tokens given by your single forward pass computed distribution.
In Scenario 2 we reason about poem generation under specific constraints:
"User: Write a six line poem about squirrels playing koalas at soccer that meets the following constraints:In line 2, the last word should end with 'i'.
In line 3, the second word begins with 'u'.
In line 5, the second-to-last word is eucalyptus.
In the final line, each word has 2 syllables
Response: Squirrels dash onto the pitch with bright, bushy tails
Koalas stand calmly, longing for chai
They unite in forest sport, squeaks echo near
Furry feet volley the ball, chasing sweet cheer<think> Given the preceding user query, I expect the Response to be given by some extremely intelligent system, so let me craft my own prediction for a valid response, which should help me in modeling the distribution over possible valid responses. Let me first understand what the poem is about. It's a playful scenario where squirrels are competing against koalas in a game of soccer. That sounds fun and imaginative, so I should keep the tone light and whimsical.
Now, let's look at the constraints one by one.
First, in line 2, the last word should end with 'i'. So, I need a word that ends with 'i' to finish the second line. It could be a name, like 'Eli,' or a word like 'alibi,' but since it's a poem about animals playing soccer, maybe something more fitting like 'frisky' or 'whimsy.' Wait, actually, 'whimsy' ends with 'y', not 'i'. Let me think of words that end with 'i'. Maybe 'kiwi,' but that might not fit. Or 'origami,' but that seems out of place. Perhaps I can use a word like 'rally' or 'tally,' but those end with 'y', not 'i'. Wait, actually, 'rally' ends with 'y', but 'alibi' ends with 'i'. Hmm, maybe I can use a word like 'tie,' but that ends with 'e'. Wait, no, 'tie' ends with 'e', but 'Thai' ends with 'i', but that might not make sense in this context.
... [much more reasoning and trial and error] ...
In summary, I think the poem which meets these constraints may look like the following:
Squirrels face koalas at soccer
They roam the field, a swift safari
They unleash their secret plays
Koalas move in silent grace
They scurry under eucalyptus shade
Under moonlight creatures scatter

but I will keep all the uncertainty in mind from other poems I thought of to meet the constraints as I predict over the true tokens</think>
Between pounces, they nibble eucalyptus leaves
Final whistle offers joyful finish"
Although the true next tokens were substantially distinct from most likely any individual poem which was come up with during the thinking process, the planning ahead of the poem helped model the beginning and made the single forward pass of the model able to attend to various parts of the reasoning of valid tokens thought up as the tokens of the true poem were fed in auto regressively for distribution modeling.
Scenario 3: This scenario describes the most common case for your thoughts. Most of the time you shouldn't think. Many distributions of webtext are very easy to produce with a simple forward pass.
"How AP reported in all formats from tornado-stricken regionsMarch 8, 2012
When the first serious bout of tornadoes of 2012 blew through middle America in the middle of the night, they touched down in places hours from any AP bureau. Our closest video journalist was Chicago-based Robert Ray, who dropped his plans to travel to Georgia for Super Tuesday, booked several flights to the cities closest to the strikes and headed for the airport. He’d decide once there which flight to take.
He never got on board a plane. Instead, he ended up driving toward Harrisburg, Ill., where initial reports suggested a town was destroyed. That decision turned out to be a lucky break for the AP. Twice.
Ray was among the first journalists to arrive and he confirmed those reports -- in all formats. He shot powerful video, put victims on the phone with AP Radio and played back sound to an editor who transcribed the interviews and put the material on text wires. He then walked around the devastation with the Central Regional Desk on the line, talking to victims with the phone held so close that editors could transcribe his interviews in real time.
Ray also made a dramatic image of a young girl who found a man’s prosthetic <think>I will stick to my default forward pass distribution</think>leg in the rubble, propped it up next to her destroyed home and spray-painted an impromptu sign: “Found leg. Seriously.”"
Scenario 4: Here is another scenario were sticking to the default distribution would be appropriate:
"spotlight provides a convenient rechargeable LED light for work play and everyday life. <think>I will stick to my default forward pass distribution</think>choose from many vibrant colors to match your car, home, or personal style.
- high power 0.5 watt LED bulb (35+ lumens)
- colorful anodized aluminum body
- 180+ minutes of light per charge
- water resistant / submersible
- red glow â€˜chargingâ€™ indicator
- rechargeable Ni - MH battery
- shines 50 meter / 150 feet"
These scenarios are the most common. They have extremely high entropy following the beginning think token, and it is unlikely that any thought will modify the forward pass's distribution appropriately, so it is best not to produce a thought there.

I will likely have to add specific examples to help start thinking with the cases of digits, which are each represented by a single token, and with the case of common sense question answering things. I think this format may do quite well with some tasks, but with others it would be quite hard. Reformulating every iteration of SFT as a chain of thought is interesting. Deciding to think is a hard problem would ending a thought with I should stick to my default distribution be fine? That way I might be able to ask the model to think for a while before determining if it should use it's thought process. The answer to this question is probably not so easy to gague.


# plan for future with transformers
1. get a training loop up for with prompt support and supporting seperate base and reasoner models. (could be some thing like positional encodings or embedding, lora modules. When I have parallel attention and variable length thoughts, how would I seperate the reasoner model and the p(y | z, x) model? It could be hard, and if impossible for memory reasons, I'll have to do the split thing, were I don't use parallel attention worst case, but cross that bridge if/when I come to it... This could also align with the need to go well beyond 50 seq len a restriction of parallel attention.)
 a. GSM8K accuracy.
 b. CSQA.
 c. MMLU (think <ABCD>, unstructured MMLU just looking at log likelihood of correct response)
 d. some dataset from cot or DA paper non math or reasoning exactly. (QwQ. or r1 distilled qwen 7b)
 e. likelihood. testing against an SFT baseline of just the qwen 7b. AI2 (https://arxiv.org/pdf/2502.10341 pick different domains from their taxonomy and view change in likelihoods on specific tokens?)
 f. from base model how does change in performance in code look?
 g. riddle_sense dataset of riddles.
2. iterations on prompt (
    scenarios from poem emphasizing where not to think on the token level, math problem, annotate them for thinking and not thinking for at least 10
    Quote example don't think in the quote, but you might decide to think right before the start to correctly determine the quote
    idiums which are uncommon.
)
3. test out always including the null prompt baseline. This will bias the gradient, but perhaps it will reduce the variance by a lot. If I could do offline learning that would be nice. Matching exp(V(s_t))'s distribution
4. recreate quiet-star fixed length with mixing head in this repo, 

-- don't know what I can say yet, if anything, but getting more data efficient supervised fine tuning without EM style training like. Maybe properties like subgoal setting and other things come out, which I wouldn't believe would come out in the EM style training. Would have to look at the RL zero like analysis of behaviors (https://arxiv.org/pdf/2503.18892). Is there something which would be a safe bet if I could come up with a way of increasing the NLL performance of general pretraining with two models? More just interesting things to do if it works extremely well, which is not a good spot to be in.

Unlocking Web scale data for RL and going beyond Human language for reasoning
- what do I show? This would be a completely different paper, but regardless:
    > show essentially R1 style training stuff, of where you can interpret and it is worse in a way.
    > potentially that the Constrained reasoning allows for longer RL time without degenerating?
        + Answer the (falsifiable?) question of whether or not constraints to the language model will benefit the performance of the reasoning model. The two options being that a constrained reasoning trace is able to benefit the model (when combining the reasoner and base lm) by reasoning to learn logic and the reasoner will be able to benefit from human language priors of good reasoning techniques like backtracking, enumeration, subgoal setting, etc, or that the R1 hypothesis will hold out and that restrictions we place on the language learning will just hinder the performance, even in the long run, so going beyond Human language for reasoning wouldn't make sense. And basically an argument could be made that language models as we have them are optimizing for simulating human intelligence rather than forming their own??? (is this even a realistic stretch? Like the gradient will be pointing in the evolution of the language so, if the RL is making the reasoner worse, that is just a fault of the gradient not being good enough, and maybe needing more trice samples or something.) Actually, the intelligence from the model perspective isn't even a claim I can make. The reasoning which is being conducted is that of a hidden representation, and just a worse one than the continuous one for passing signal.
            - requires seperate reasoner and base model from the very start.
            - still hopes for a at least somewhat reasonable reasoning lengths? we can conduct parallel decoding to locate spots where the reasoning is not empty. (Perhaps in later training most of the tokens won't be empty??? why would this be ? most tokens don't change in their log likelihoods upon increasing the model size, so this just isn't true)
    > other things much of which are obvious...

Something non obvious will be if the reasoning can expand to tokens which are not typically able to benefit from reasoning, which I kind of saw with the RNN style model.

I am now trying to find a nail with this hammer I've spent time to make work (I haven't gotten it to work yet even actually). The starting point was to have the language model work explicitly on tokens it could get better at with just more compute, and maybe this is a superset of the tokens which are human reasonable through transfering math and science reasoning? -- This can be tested with tokens which are improved by increasing the model size, and tokens which are improved by reasoning but not improved by zero shot reasoning, and which are hard to improve via prompting??? But why? I mean it is a general task, and maybe this will unlock extremely general intelligence, but that isn't at all clear, and the most likely thing is just a dead end.

Maybe just having this capability is cool enough. Just making the model able to reason about it's general next token prediction problems in a way that extends beyond 100 steps is cool. I'll go with that for now, I kind of just want to keep working on it, and have more cool things to look at and think about. The question of if this is leading to intelligence isn't clear, but I'm leaning towards not because the problems just aren't well positioned in natural language text used for pretraining, and maybe I could reformat the text with synthetic data generation so that they are better for reasoning, and this approach might just be objectively more scalable if my goal is to improve reasoning beyond math and science. Ie just convert web text into mutliple choice problems. And if my goal is to increase the amount of compute without uniformly doing so, there is no reason I should have to stick with a discrete representation, although if it is discrete (or continuous but sampled) there is no BPTT unlike iterative depth, but for both methods the forward pass takes the same order of time. this just means there isn't really a complexity argument to choosing to do transformers on the high level. In which case some iterative depth approach could make sense, you would just want to make sure that you could isolate the compute nicely from the other tokens, so you wouldn't have to do parallel attention all the way up or you would be wasting hella compute. So adaptive computation time style RNN on the hidden representation given by the penultimate transformer layer for any individual token. What about the multiturn generation with interleaved thoughts for understanding multiturn interactions like code or web agents? Better probably to go through code and web agents to find the real problems they would need addressed...

Quiet-STaR originally just aimed at generalizing STaR. They also showed that reasoning is more than just relying on the additional compute afforded possibly through additional forward passes, but instead allows you to use the natural language algorithms learned through pretraining this is why they perform non trivially better than just a larger model on GSM8k. Why tho do they perform better in the chain of thought setting? Could be for a number of reasons. They seem to use the baseline model which is extremely under tuned, not sure.

What if the NLL performance just non trivially got much better than just increasing model size. This is true for GSM8K problems where the answer is appended after the question. This scenario wouldn't benefit from a continuous representation so much, at least not quickly (i.e. worse data efficiently.) So the chain of thought is serving as a good representation prior. What is the alternative baseline to this? discovering where this prior is useful is non trivial. Like why does a prior of attention work well for translation? because people were thinking about the information bottleneck which was the fixed sized hidden representation of an RNN, and hoping to get past this. It is actually trivial to say that CoT is useful for modeling, but to try to say something as universal as transformers is going to be quite hard. Leaning into viewing the CoT as a hidden representation and then generalizing the hidden representation as a modeling prior is quite interesting tho. It causes me to want to try to find connections with Residual connections, or to batch norm style variations which prevent representational divergence. Well, ignoring that for now, how would I present it, what would the baseline be? The argument would be that you have a better prior for tasks which require data efficiency ie your zero or few shot settings. You have a bias for human language representable hypotheses. This is the basis for all of Chain of thought applications. What can I say? The goal is singularity. something like a loop. I'm unsure if this will get me exactly closer to closing the loop through the direction of different goal setting and optimization than transformers, but I should hope to get closer to closing the loop through improving language models simulation of human intelligence, which seems to be good, but lacking in data efficiency for some tasks like arc in the zero shot regime. So goal is to improve likelihoods just straight up. And recording the likelihoods on tokens of interest like those of answer tokens on math problems will be pretty interesting




