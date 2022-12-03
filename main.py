import os, sys, torch, transformers
import numpy as np
from transformers import GPT2Tokenizer, GPT2Model
from scipy.stats import wasserstein_distance
from scipy.spatial import distance
from scipy.spatial import distance_matrix
from collections import defaultdict
import time


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def generate_stories(fn):
    stories = []
    f = open(fn)
    first_line = f.readline()
    assert first_line.strip() == "!ARTICLE"
    curr_story = ""

    for line in f:
        sentence = line.strip()
        if sentence == "!ARTICLE":
            stories.append(curr_story[:-1])
            curr_story = ""
        else:
            curr_story += line.strip() + " "

    stories.append(curr_story[:-1])
    return stories


def normalize_by_norms(tensor):
    norms = torch.norm(tensor, dim=-1)
    return norms/norms.sum(dim=-1, keepdims=True)


def normalized_ent(tensor):
    # the diagonal=-1 makes attn entropy conform to the original Ryu and Lewis (2021) formulation
    tensor = torch.tril(tensor, diagonal=-1)
    # renormalizes over 1..i-1
    tensor = tensor / tensor.sum(dim=-1).unsqueeze(-1)
    log_tensor = torch.tril(tensor, diagonal=-1)
    # norm_tensor divides entropy by log2(N)
    norm_vector = torch.arange(0, tensor.shape[0], dtype=torch.float).unsqueeze(0).t()
    norm_vector[0] += 0.0001
    norm_tensor = torch.log2(norm_vector.repeat(1, tensor.shape[0]))
    norm_tensor[1] += 0.0001
    idx = log_tensor != 0
    log_tensor[idx] = torch.log2(log_tensor[idx])
    return torch.sum(torch.div((-1 * tensor * log_tensor), norm_tensor), dim=-1)


def calc_norm_ent(tensor):
    norm_ent = defaultdict(float)
    # iterate over heads
    for i in range(tensor.shape[0]):
        ent_tensor = normalized_ent(tensor[i])
        # iterate over timesteps
        for j in range(ent_tensor.shape[0]-1):
            norm_ent[j] += ent_tensor[j+1].item()

    return norm_ent


def calc_delta_ent(tensor):
    delta_ent = defaultdict(float)
    # iterate over heads
    for i in range(tensor.shape[0]):
        ent_tensor = normalized_ent(tensor[i])
        for j in range(ent_tensor.shape[0]-1):
            delta_ent[j] += abs(ent_tensor[j+1].item() - ent_tensor[j].item())

    return delta_ent


def calc_md(tensor):
    md = defaultdict(float)
    # iterate over heads
    for i in range(tensor.shape[0]):
        q_tensor = tensor[i][:-1]
        p_tensor = tensor[i][1:]
        # iterate over tokens
        for j in range(p_tensor.shape[0]):
            md[j] += (distance.cityblock(q_tensor[j][:j + 2].detach().numpy(), p_tensor[j][:j + 2].detach().numpy()))

    return md


def calc_norm_emd(tensor):
    emd = defaultdict(float)
    # iterate over heads
    for i in range(tensor.shape[0]):
        q_tensor = tensor[i][:-1]
        p_tensor = tensor[i][1:]
        # iterate over tokens
        for j in range(p_tensor.shape[0]):
            # bins = np.arange(0, j+2)
            # normalized EMD; the bins range from 0 to 1 regardless of the number of bins
            bins = np.linspace(0, 1, num=j+2)
            emd[j] += wasserstein_distance(bins, bins, q_tensor[j][:j+2].detach().numpy(), p_tensor[j][:j+2].detach().numpy())

    return emd


def main():
    preds = ["word", "normentw", "normentn", "normentr", "deltaentw", "deltaentn", "deltaentr",
             "manhattanw", "manhattann", "manhattanr", "emdw", "emdn", "emdr"]
    print(" ".join(preds))
    stories = generate_stories(sys.argv[1])
    tokenizer = GPT2Tokenizer.from_pretrained(sys.argv[2])
    model = GPT2Model.from_pretrained(sys.argv[2], output_attentions=True)
    model.eval()
    ctx_size = model.config.n_ctx
    bos_id = model.config.bos_token_id

    batches = []
    words = []
    for story in stories:
        words.extend(story.split(" "))
        # GPT2Tokenizer does not automatically add <|endoftext|> to beginning
        tokenizer_output = tokenizer(story)
        ids = tokenizer_output.input_ids
        attn = tokenizer_output.attention_mask
        start_idx = 0

        # in case the text doesn't fit in one context window (sliding window with stride of 50%)
        while len(ids) > ctx_size:
            batches.append((transformers.BatchEncoding({"input_ids": torch.tensor([bos_id] + ids[:ctx_size-1]),
                                                       "attention_mask": torch.tensor([1] + attn[:ctx_size-1])}), start_idx))
            ids = ids[int(ctx_size/2):]
            attn = attn[int(ctx_size/2):]
            start_idx = int(ctx_size/2)-1

        batches.append((transformers.BatchEncoding({"input_ids": torch.tensor([bos_id] + ids),
                                                   "attention_mask": torch.tensor([1] + attn)}), start_idx))

    curr_word_ix = 0
    curr_word_vals = [0] * 12
    curr_word = ""

    for batch in batches:
        batch_input, start_idx = batch
        with torch.no_grad():
            model_output = model(**batch_input)

        toks = tokenizer.convert_ids_to_tokens(batch_input.input_ids)[1:]

        # attentions: Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, seq_length, seq_length).
        # projected_states: Tuple of torch.FloatTensor (one from final layer) of shape (batch_size, num_heads, seq_length, seq_length, num_heads*head_dim).
        # resln_states: Tuple of torch.FloatTensor (one from final layer) of shape (batch_size, num_heads, seq_length, seq_length, num_heads*head_dim).
        attn = model_output.attentions[-1].squeeze(0)  # num_heads, seq_length, seq_length
        projected_states = model_output.projected_states[-1].squeeze(0)  # num_heads, seq_length, seq_length, num_heads*head_dim
        resln_states = model_output.resln_states[-1].squeeze(0)  # num_heads, seq_length, seq_length, num_heads*head_dim

        norm_proj = normalize_by_norms(projected_states)
        norm_resln = normalize_by_norms(resln_states)

        del model_output, projected_states, resln_states

        normentw = calc_norm_ent(attn)
        normentn = calc_norm_ent(norm_proj)
        normentr = calc_norm_ent(norm_resln)

        deltaentw = calc_delta_ent(attn)
        deltaentn = calc_delta_ent(norm_proj)
        deltaentr = calc_delta_ent(norm_resln)

        manhattanw = calc_md(attn)
        manhattann = calc_md(norm_proj)
        manhattanr = calc_md(norm_resln)

        emdw = calc_norm_emd(attn)
        emdn = calc_norm_emd(norm_proj)
        emdr = calc_norm_emd(norm_resln)

        # MAIN LOOP
        for i in range(start_idx, len(toks)):
            curr_tok_vals = []
            for j in [normentw, normentn, normentr, deltaentw, deltaentn, deltaentr,
                      manhattanw, manhattann, manhattanr, emdw, emdn, emdr]:
                curr_tok_vals.append(j[i])

            # add predictor values together for multiple-token words
            curr_word_vals = [sum(x) for x in zip(curr_word_vals, curr_tok_vals)]

            # concatenate tokens together for multiple-token words
            curr_tok = toks[i].replace("Ġ", "", 1).encode("latin-1").decode("utf-8")
            curr_word += curr_tok
            words[curr_word_ix] = words[curr_word_ix].replace(curr_tok, "", 1)

            if words[curr_word_ix] == "":
                print(curr_word + " " + " ".join([str(val) for val in curr_word_vals]))
                curr_word_vals = [0] * 12
                curr_word = ""
                curr_word_ix += 1


if __name__ == "__main__":
    main()
