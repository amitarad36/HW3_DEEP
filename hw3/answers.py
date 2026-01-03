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
We split the data into fixed-length sequences (windows) for several reasons. Firstly this allows pairing each sequence with labels (the same sequence shifted by one character for next-character prediction).
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
        batch_size=32, h_dim=1024, z_dim=128, x_sigma2=0.0009, learn_rate=0.0002, betas=(0.9, 0.999),
    ) # x_sigma2=0.0009
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    # (i tweaked them in the dict itself)
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
    pass
    # ========================
    return hypers


part3_q1 = r"""
**Your answer:**
"""

part3_q2 = r"""
**Your answer:**
"""

# ==============
