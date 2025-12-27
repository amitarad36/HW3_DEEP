import re
import torch
import torch.nn as nn
import torch.utils.data
from torch import Tensor
from typing import Iterator

# Yuval
def char_maps(text: str):
    """
    Create mapping from the unique chars in a text to integers and
    vice-versa.
    :param text: Some text.
    :return: Two maps.
        - char_to_idx, a mapping from a character to a unique
        integer from zero to the number of unique chars in the text.
        - idx_to_char, a mapping from an index to the character
        represented by it. The reverse of the above map.

    """
    # TODO:
    #  Create two maps as described in the docstring above.
    #  It's best if you also sort the chars before assigning indices, so that
    #  they're in lexical order.
    # ====== YOUR CODE: ======
    unique_chars = sorted(set(text))
    char_to_idx = {char: idx for idx, char in enumerate(unique_chars)}
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    # ========================
    return char_to_idx, idx_to_char


def remove_chars(text: str, chars_to_remove):
    """
    Removes all occurrences of the given chars from a text sequence.
    :param text: The text sequence.
    :param chars_to_remove: A list of characters that should be removed.
    :return:
        - text_clean: the text after removing the chars.
        - n_removed: Number of chars removed.
    """
    # TODO: Implement according to the docstring.
    # ====== YOUR CODE: ======
    pattern = f"[{''.join(re.escape(char) for char in chars_to_remove)}]"
    text_clean = re.sub(pattern, '', text)
    n_removed = len(text) - len(text_clean)
    # ========================
    return text_clean, n_removed


def chars_to_onehot(text: str, char_to_idx: dict) -> Tensor:
    """
    Embed a sequence of chars as a a tensor containing the one-hot encoding
    of each char. A one-hot encoding means that each char is represented as
    a tensor of zeros with a single '1' element at the index in the tensor
    corresponding to the index of that char.
    :param text: The text to embed.
    :param char_to_idx: Mapping from each char in the sequence to it's
    unique index.
    :return: Tensor of shape (N, D) where N is the length of the sequence
    and D is the number of unique chars in the sequence. The dtype of the
    returned tensor will be torch.int8.
    """
    # TODO: Implement the embedding.
    # ====== YOUR CODE: ======
    N = len(text)
    V = len(char_to_idx)
    result = torch.zeros((N, V), dtype=torch.int8)
    if N == 0:
        return result

    # Fill one-hot rows according to mapping
    for i, ch in enumerate(text):
        idx = char_to_idx.get(ch)
        if idx is None:
            raise KeyError(f"Character '{ch}' not found in mapping.")
        result[i, idx] = 1
    # ========================
    return result


def onehot_to_chars(embedded_text: Tensor, idx_to_char: dict) -> str:
    """
    Reverses the embedding of a text sequence, producing back the original
    sequence as a string.
    :param embedded_text: Text sequence represented as a tensor of shape
    (N, D) where each row is the one-hot encoding of a character.
    :param idx_to_char: Mapping from indices to characters.
    :return: A string containing the text sequence represented by the
    embedding.
    """
    # TODO: Implement the reverse-embedding.
    # ====== YOUR CODE: ======
    if embedded_text.numel() == 0:
        return ""

    # Argmax over the one-hot dimension to recover indices
    idxs = embedded_text.argmax(dim=1).tolist()
    chars = [idx_to_char[i] for i in idxs]
    result = "".join(chars)
    # ========================
    return result


def chars_to_labelled_samples(text: str, char_to_idx: dict, seq_len: int, device="cpu"):
    """
    Splits a char sequence into smaller sequences of labelled samples.
    A sample here is a sequence of seq_len embedded chars.
    Each sample has a corresponding label, which is also a sequence of
    seq_len chars represented as indices. The label is constructed such that
    the label of each char is the next char in the original sequence.
    :param text: The char sequence to split.
    :param char_to_idx: The mapping to create and embedding with.
    :param seq_len: The sequence length of each sample and label.
    :param device: The device on which to create the result tensors.
    :return: A tuple containing two tensors:
    samples, of shape (N, S, V) and labels of shape (N, S) where N is
    the number of created samples, S is the seq_len and V is the embedding
    dimension.
    """
    # TODO:
    #  Implement the labelled samples creation.
    #  1. Embed the given text.
    #  2. Create the samples tensor by splitting to groups of seq_len.
    #     Notice that the last char has no label, so don't use it.
    #  3. Create the labels tensor in a similar way and convert to indices.
    #  Note that no explicit loops are required to implement this function.
    # ====== YOUR CODE: ======
    total_chars = len(text)
    V = len(char_to_idx)

    # Guard against invalid inputs
    if seq_len <= 0 or total_chars < 2:
        samples = torch.empty((0, seq_len, V), dtype=torch.int8, device=device)
        labels = torch.empty((0, seq_len), dtype=torch.long, device=device)
        return samples, labels

    # Number of full samples that can be labelled (each position needs a next char)
    N = (total_chars - 1) // seq_len 
    if N <= 0:
        samples = torch.empty((0, seq_len, V), dtype=torch.int8, device=device)
        labels = torch.empty((0, seq_len), dtype=torch.long, device=device)
        return samples, labels

    # We need the first N*seq_len + 1 characters (last char supplies the final label)
    use_chars = N * seq_len + 1
    # Map chars to indices (will raise KeyError if char missing)
    idx_list = [char_to_idx[ch] for ch in text[:use_chars]]
    indices = torch.tensor(idx_list, dtype=torch.long, device=device)

    # samples indices are all but last, reshaped to (N, seq_len)
    samples_idx = indices[:-1].view(N, seq_len)
    # labels are indices shifted by 1, reshaped to (N, seq_len)
    labels = indices[1:].view(N, seq_len)

    # Convert sample indices to one-hot (N, seq_len, V) and cast to int8
    samples = torch.nn.functional.one_hot(samples_idx, num_classes=V).to(dtype=torch.int8, device=device)

    # ========================
    return samples, labels


def hot_softmax(y, dim=0, temperature=1.0):
    """
    A softmax which first scales the input by 1/temperature and
    then computes softmax along the given dimension.
    :param y: Input tensor.
    :param dim: Dimension to apply softmax on.
    :param temperature: Temperature.
    :return: Softmax computed with the temperature parameter.
    """
    # TODO: Implement based on the above.
    # ====== YOUR CODE: ======
    scaled_y = y / temperature
    softmax_y = torch.nn.functional.softmax(scaled_y, dim=dim)
    return softmax_y
    # ========================


def generate_from_model(model, start_sequence, n_chars, char_maps, T):
    """
    Generates a sequence of chars based on a given model and a start sequence.
    :param model: An RNN model. forward should accept (x,h0) and return (y,
    h_s) where x is an embedded input sequence, h0 is an initial hidden state,
    y is an embedded output sequence and h_s is the final hidden state.
    :param start_sequence: The initial sequence to feed the model.
    :param n_chars: The total number of chars to generate (including the
    initial sequence).
    :param char_maps: A tuple as returned by char_maps(text).
    :param T: Temperature for sampling with softmax-based distribution.
    :return: A string starting with the start_sequence and continuing for
    with chars predicted by the model, with a total length of n_chars.
    """
    assert len(start_sequence) < n_chars
    device = next(model.parameters()).device
    char_to_idx, idx_to_char = char_maps
    out_text = start_sequence

    # TODO:
    #  Implement char-by-char text generation.
    #  1. Feed the start_sequence into the model.
    #  2. Sample a new char from the output distribution of the last output
    #     char. Convert output to probabilities first.
    #     See torch.multinomial() for the sampling part.
    #  3. Feed the new char into the model.
    #  4. Rinse and Repeat.
    #  Note that tracking tensor operations for gradient calculation is not
    #  necessary for this. Best to disable tracking for speed.
    #  See torch.no_grad().
    # ====== YOUR CODE: ======
    param_dtype = next(model.parameters()).dtype
    with torch.no_grad():
        # Step 1: Feed the start_sequence into the model
        input_seq = chars_to_onehot(start_sequence, char_to_idx).unsqueeze(0).to(device)  # Shape (1, S, V)
        # one-hot is int8; convert to model parameter dtype (usually float)
        input_seq = input_seq.to(dtype=param_dtype)
        output, hidden_state = model(input_seq)

        # Generate characters until reaching n_chars
        current_char = start_sequence[-1]
        for _ in range(len(start_sequence), n_chars):
            # Get the output for the last character
            last_output = output[0, -1]  # Shape (V,)

            # Step 2: Convert output to probabilities with temperature
            probs = hot_softmax(last_output, dim=0, temperature=T)

            # Sample a new char index from the distribution
            sampled_idx = torch.multinomial(probs, num_samples=1).item()
            sampled_char = idx_to_char[sampled_idx]
            out_text += sampled_char

            # Step 3: Feed the new char into the model
            new_input = chars_to_onehot(sampled_char, char_to_idx).unsqueeze(0).to(device)  # Shape (1, 1, V)
            new_input = new_input.to(dtype=param_dtype)
            output, hidden_state = model(new_input, hidden_state)
    # ========================

    return out_text


class SequenceBatchSampler(torch.utils.data.Sampler):
    """
    Samples indices from a dataset containing consecutive sequences.
    This sample ensures that samples in the same index of adjacent
    batches are also adjacent in the dataset.
    """

    def __init__(self, dataset: torch.utils.data.Dataset, batch_size):
        """
        :param dataset: The dataset for which to create indices.
        :param batch_size: Number of indices in each batch.
        """
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[int]:
        # TODO:
        #  Return an iterator of indices, i.e. numbers in range(len(dataset)).
        #  dataset and represents one  batch.
        #  The indices must be generated in a way that ensures
        #  that when a batch of size self.batch_size of indices is taken, samples in
        #  the same index of adjacent batches are also adjacent in the dataset.
        #  In the case when the last batch can't have batch_size samples,
        #  you can drop it.
        idx = None  # idx should be a 1-d list of indices.
        # ====== YOUR CODE: ======
        total_samples = len(self.dataset)
        n_batches = total_samples // self.batch_size
        idx = []
        for i in range(n_batches):
            for j in range(self.batch_size):
                idx.append(i + j * n_batches)
        # ========================
        return iter(idx)

    def __len__(self):
        return len(self.dataset)


class MultilayerGRU(nn.Module):
    """
    Represents a multi-layer GRU (gated recurrent unit) model.
    """

    def __init__(self, in_dim, h_dim, out_dim, n_layers, dropout=0):
        """
        :param in_dim: Number of input dimensions (at each timestep).
        :param h_dim: Number of hidden state dimensions.
        :param out_dim: Number of input dimensions (at each timestep).
        :param n_layers: Number of layer in the model.
        :param dropout: Level of dropout to apply between layers. Zero
        disables.
        """
        super().__init__()
        assert in_dim > 0 and h_dim > 0 and out_dim > 0 and n_layers > 0

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.h_dim = h_dim
        self.n_layers = n_layers
        self.layer_params=[]
        
        # ====== YOUR CODE: ======
        self.layer_params = nn.ModuleList() 

        for layer_index in range(n_layers): 
            input_dim = in_dim if layer_index == 0 else h_dim
            layer = nn.Module()
            layer.update_wx = nn.Linear(input_dim, h_dim, bias=False)
            layer.update_wh = nn.Linear(h_dim, h_dim, bias=False)
            layer.reset_wx = nn.Linear(input_dim, h_dim, bias=False)
            layer.reset_wh = nn.Linear(h_dim, h_dim, bias=False)
            layer.candidate_wx = nn.Linear(input_dim, h_dim, bias=False)
            layer.candidate_wh = nn.Linear(h_dim, h_dim, bias=False)
            layer.b_z = nn.Parameter(torch.zeros(h_dim))
            layer.b_r = nn.Parameter(torch.zeros(h_dim))
            layer.b_g = nn.Parameter(torch.zeros(h_dim))
            layer.dropout = nn.Dropout(dropout) if layer_index > 0 and dropout > 0 else nn.Identity()
            self.layer_params.append(layer)

        self.h2o = nn.Linear(h_dim, out_dim)
        # ========================
        
    def forward(self, input: Tensor, hidden_state: Tensor = None):
        """
        :param input: Batch of sequences. Shape should be (B, S, I) where B is
        the batch size, S is the length of each sequence and I is the
        input dimension (number of chars in the case of a char RNN).
        :param hidden_state: Initial hidden state per layer (for the first
        char). Shape should be (B, L, H) where B is the batch size, L is the
        number of layers, and H is the number of hidden dimensions.
        :return: A tuple of (layer_output, hidden_state).
        The layer_output tensor is the output of the last RNN layer,
        of shape (B, S, O) where B,S are as above and O is the output
        dimension.
        The hidden_state tensor is the final hidden state, per layer, of shape
        (B, L, H) as above.
        """
        batch_size, seq_len, _ = input.shape

        layer_states = []
        for i in range(self.n_layers):
            if hidden_state is None:
                layer_states.append(
                    torch.zeros(batch_size, self.h_dim, device=input.device)
                )
            else:
                layer_states.append(hidden_state[:, i, :])

        layer_input = input
        layer_output = None

        # Loop over timesteps and layers of the model
        activation_sigmoid, activation_tanh = nn.Sigmoid(), nn.Tanh()
        output_seq = torch.zeros((batch_size, seq_len, self.out_dim), device=input.device, dtype=input.dtype)

        for time_step in range(seq_len):
            current_input = layer_input[:, time_step]

            for layer_index in range(self.n_layers):
                layer = self.layer_params[layer_index]
                prev_state = layer_states[layer_index]

                if layer_index > 0:
                    current_input = layer_states[layer_index - 1]

                update_gate = activation_sigmoid(layer.update_wx(current_input) + layer.update_wh(prev_state) + layer.b_z)
                reset_gate = activation_sigmoid(layer.reset_wx(current_input) + layer.reset_wh(prev_state) + layer.b_r)
                candidate_state = activation_tanh(layer.candidate_wx(current_input) + layer.candidate_wh(reset_gate * prev_state) + layer.b_g)

                new_state = update_gate * prev_state + (1 - update_gate) * candidate_state
                layer_states[layer_index] = new_state

                if layer_index > 0:
                    current_input = layer.dropout(new_state)

            # project last layer hidden to output dim
            output_seq[:, time_step] = self.h2o(layer_states[-1])

        final_hidden_state = torch.stack(layer_states, dim=1)
        hidden_state = final_hidden_state
        layer_output = output_seq

        return layer_output, hidden_state
