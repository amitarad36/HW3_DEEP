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
        batch_size=0, h_dim=0, z_dim=0, x_sigma2=0, learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    pass
    # ========================
    return hypers


part2_q1 = r"""
**Your answer:**
"""

part2_q2 = r"""
**Your answer:**
"""

part2_q3 = r"""
**Your answer:**
"""

part2_q4 = r"""
**Your answer:**
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
