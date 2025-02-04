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