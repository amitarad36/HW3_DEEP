r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers


def part1_rnn_hyperparams():
    hypers = dict(
        batch_size=0,
        seq_len=0,
        h_dim=0,
        n_layers=0,
        dropout=0,
        learn_rate=0.0,
        lr_sched_factor=0.0,
        lr_sched_patience=0,
    )
    # TODO: Set the hyperparameters to train the model.
    # ====== YOUR CODE: ======
    hypers["batch_size"] = 64
    hypers["seq_len"] = 50
    hypers["h_dim"] = 128
    hypers["n_layers"] = 2
    hypers["dropout"] = 0.2
    hypers["learn_rate"] = 0.002
    hypers["lr_sched_factor"] = 0.5
    hypers["lr_sched_patience"] = 3
    # ========================
    return hypers


def part1_generation_params():
    start_seq = ""
    temperature = 0.0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    start_seq = "ACT I."
    temperature = 0.5
    # ========================
    return start_seq, temperature


part1_q1 = r"""
**Your answer:**
Sliding-window attention is like a CNN: each token only looks at a local window, so cost drops from $O(n^2)$ to about $O(n \cdot w)$ and we keep the most relevant nearby context. Long-range info still gets through by stacking layers—each layer lets information hop one window farther—so depth expands reach even though any single layer is local. The trade-off is if the window is tiny or depth is shallow, faraway dependencies can be missed.
Moreover this keeps tensors fit in memory, enables parallelism (batching, passing forward a batch and not just one sequence at a time), and applies truncated BPTT so gradients remain stable. And lastly, using this method we also increase the number of training examples (we use overlapping windows).
"""

part1_q2 = r"""
**Your answer:**
The mechanism responsible for that is the encoding of information in the hidden state. This applies across timesteps and across contiguous batches (the sequenceBatchSampler preserves order), so the model can showcase context preservation for longer than the explicit sequence window.At training time we ensure passing last hidden state of batch j as initial hidden state of batch j+1. At generation time the hidden state keeps accumulating history, which extends memory beyond one sequence length.
"""

part1_q3 = r"""
**Your answer:**
As mentioned in the previous answer, we need contiguous ordering so that sample k in batch j continues sample k in batch j+1, allowing hidden states to act as proper memory, extending the capabilities of text generation to more than one sequence length. Shuffling would break sequence continuity and invalidate the carried hidden states, which will result in impaired training.
"""

part1_q4 = r"""
**Your answer:**
1. Lowering temperature sharpens the distribution graph, what makes sampling tend toward argmax which in turn displays a more coherent text, and more deterministic behavior.
2. Very high temperature flattens the distribution graph toward uniform, wich leads to a deviation toward randomness or gibberish generation.
3. Very low temperature makes sampling almost deterministic (since the distribution values are exaggerated). one can identify repeating patterns due to low diversity.
"""
# ==============


# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0, h_dim=0, z_dim=0, x_sigma2=0, learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    hypers["batch_size"] = 32
    hypers["h_dim"] = 1024
    hypers["z_dim"] = 128
    hypers["x_sigma2"] = 0.0009
    hypers["learn_rate"] = 0.0002
    hypers["betas"] = (0.9, 0.999)
    # ========================
    return hypers


part2_q1 = r"""
**Your answer:**

 $\sigma^2$ represents the variance of the gaussian likelihood.
 In the loss, it says how much weight is put  into the reconstruction term (MSE) against the KL regularization.

If we put small value in $\sigma^2$, the MSE is more important, so the model focuses on
getting the output closer the input, at the cost of getting a smooth latent
distribution close to the prior.

If we put large value in $\sigma^2$, MSE is weighted less, making the KL divergence more important. 
This forces the posterior closer to the $N(0,I)$ prior and leads the decoder to ignore
input details, getting almost the same picture everytime.
 
"""

part2_q2 = r"""
**Your answer:**

1.
As explained in previous answer, the reconstruction loss (MSE)
 punishes the model for reconstructing an image that doesn't look like the original image. 

The KL divergence loss punishes the model the further the approximate posterior is from a $N(0,I)$ distribution.
It acts as a regularization term in our case.

2. The KL loss forces the latent-space distribution to be close to a standard normal distribution,
making latent means move toward zero and variances toward one.

3. The benefit of this effect is that the latent space becomes more smooth and well-structured, which makes
sampling and interpolation meaningful: nearby points in the latent space give similar outputs, and sampling
from $N(0,I)$ gives realistic images.
"""

part2_q3 = r"""
**Your answer:**

We maximize $p(X)$ because learning a generative model means maximizing the
 likelihood of the observed data, and the VAE loss is a tractable lower bound on this objective (via Jensen’s inequality).


"""

part2_q4 = r"""
**Your answer:**

We model the log-variance because variance has to be positive
, while neural networks can output any real value. Predicting $\log\sigma^2$ and
exponentiating it guarantees a positive number without requiring additional constraints.

In addition, using log gives more numerical stability in optimization, especially
for very small/large values. It also makes the computation of the KL
divergence simpler, which depends on $\log\sigma^2$.

"""


def part3_transformer_encoder_hyperparams():
    hypers = dict(
        embed_dim = 0, 
        num_heads = 0,
        num_layers = 0,
        hidden_dim = 0,
        window_size = 0,
        droupout = 0.0,
        lr=0.0,
    )

    # TODO: Tweak the hyperparameters to train the transformer encoder.
    # ====== YOUR CODE: ======
    hypers["embed_dim"] = 256
    hypers["num_heads"] = 4
    hypers["num_layers"] = 2
    hypers["hidden_dim"] = 256
    hypers["window_size"] = 128
    hypers["droupout"] = 0.2
    hypers["lr"] = 0.0001
    # ========================
    return hypers


part3_q1 = r"""
**Your answer:**

Sliding-window attention reduces the computational complexity from quadratic $O(n^2)$ to linear $O(n \cdot w)$ by restricting each token to a local window of size $w$. While this makes processing long sequences feasible, a single layer is limited to seeing only its immediate neighbors.

Stacking encoder layers compensates for this limitation by expanding the **Effective Receptive Field**. In the first layer, a token aggregates information only from its local window. In the second layer, that same token attends to neighbors who have already gathered information from *their* own windows in the previous layer. This allows information to propagate outward, "hopping" further with each additional layer.

Much like stacking convolutional layers increases the receptive field size linearly with depth, stacking sliding-window attention layers allows the final layer to incorporate global context from the entire sequence. This enables the model to capture long-range dependencies despite using only computationally efficient, local operations.
"""

part3_q2 = r"""
**Your answer:**

**Proposed Variation: Block-Dilated Attention**

We propose a pattern that mixes local focus with distant "hops," constructed by concatenating blocks of valid tokens with blocks of skipped tokens.

**The Pattern:** We define a **locality size** ($l$) and a **dilation gap** ($d$). We repeat this pattern $n$ times.

- **Visual:** +++ -- +++ -- +++ (where +++ is length $l$, and -- is length $d$, we mark it here with d=2, l=3 for example).
- **Total Span:** The attention reaches back much further than a standard window.

**Complexity Analysis:** The time complexity stays $O(N \cdot w)$, where $w$ is the total number of active tokens ($w = n \cdot l$). Even though the window stretches further back due to the gaps ($d$), the model still computes the same fixed number of dot products per token. The computation cost depends only on the number of + s, not the - s.

**Global Context & Layers:** This method shares global information much faster than the standard sliding window. By adding the gaps ($d$), a single layer can "see" distinct parts of the sequence that are far apart.

- **Layers:** We would need fewer layers to cover the whole sequence because each layer's reach is multiplied by the spacing.
- **Limitations:** The main trade-off is "blind spots." If a critical word falls exactly in a --- gap, the current token can't see it directly. It has to wait for a deeper layer where that missing word has been aggregated by a neighbor. A potential fix can be to vary the parameters ($l$, $d$) across layers to ensure coverage, say we start with large d and small l and gradually decrease d while increasing l in deeper layers.
"""

# ==============
