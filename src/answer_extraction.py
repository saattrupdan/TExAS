'''Answer extraction using beam search over cross-attention values'''

from torch import Tensor
import torch
from typing import List, Tuple


def extract_translated_answer(answer_token_idxs: List[int],
                              cross_attention_tensor: Tensor,
                              beam_width: int = 100,
                              beam_radius: int = 3,
                              max_attention_sum: int = 10
                              ) -> Tuple[List[int], float]:
    '''Extracts the location of the answer in the translated document.

    Args:
        answer_token_idxs (list of ints):
            The token indices of the answer in the original context.
        cross_attention_tensor (PyTorch Tensor):
            The first cross-attention tensor, of shape (attention_heads,
            num_target_tokens, num_source_tokens).
        beam_width (int, optional):
            The width of the beam, i.e., how many combinations will be kept in
            memory at every token step. Defaults to 100.
        beam_radius (int, optional):
            The radius of the beam, i.e., the maximal number of tokens between
            consecutive translated tokens. Defaults to 3.
        max_attention_sum (int, optional):
            The maximal sum of attention values that will be considered for
            the answer. Defaults to 10.

    Returns:
        pair of a list of int and a float:
            The first entry consists of indices of the tokens in the translated
            document that correspond to the answer, and the second entry is the
            attention score of the best translation.
    '''
    with torch.no_grad():

        # Ensure that `beam_width` is not larger than the number of target
        # tokens
        beam_width = min(beam_width, cross_attention_tensor.size(1))

        # Abbreviate wordy variable names
        s, e = min(answer_token_idxs), max(answer_token_idxs)

        # Extract all the cross-attention values for the answer.
        # This has shape (attention_heads, num_target_tokens, answer),
        # with answer being the number of tokens in the answer.
        att_values = cross_attention_tensor[:, :, s : e + 1]

        # Use the attention head with the highest value for each token.
        # This has shape (num_target_tokens, answer).
        att_values = att_values.max(dim=0)[0]

        # Set the attention values we're not interested in to 0.
        mask = [i not in answer_token_idxs for i in range(s, e + 1)]
        att_values[:, mask] = 0.

        # Remove the tokens in the answer which have a large sum of attention
        # values, as these tend to be irrelevant words like 'the' or 'a'.
        mask = att_values.sum(dim=0).gt(max_attention_sum)
        att_values[:, mask] = 0.

        # Start of beam search.
        # For the first token, we simply take the top `beam_width` best tokens,
        # i.e. the tokens with the highest cross-attention values.
        beam = att_values[:, 0].topk(k=beam_width, dim=0)

        # `beam_values` will have shape (beam_width,) and contain the sum
        # of the cross-attention values for the `beam_width` best token
        # combinations, and `beam_idxs` will have shape (beam_width, 1) and
        # contain the indices of those combinations.
        # The `beam_idxs` tensor will be replaced by larger tensors later,
        # ending up with shape (beam_width, answer).
        beam_values, beam_idxs = beam
        beam_idxs = beam_idxs.unsqueeze(dim=1)

        # We traverse each token in the answer, and keep the top `beam_width` best
        # combinations of target tokens.
        for token_idx in range(s + 1, e + 1):

            # Initialise tensors to hold all the `beam_width` ** 2 new
            # combinations, of which we will keep the `beam_width` best.
            all_beam_idxs = torch.zeros(beam_width ** 2,
                                        1 + token_idx - s,
                                        dtype=torch.long)
            all_beam_values = torch.zeros(beam_width ** 2, dtype=torch.float)

            # For each of the previous best combinations, we select the
            # `beam_width` best combinations of the next token.
            for i, beam_idx_tensor in enumerate(beam_idxs):

                # The last beam index of this combination dictates the
                # possible attention values that are relevant. Namely, we only
                # include the attention values for the tokens that are at most
                # five tokens away from that last token.
                last_beam_idx = beam_idx_tensor[-1]
                att_start = max(0, last_beam_idx)
                att_end = min(att_values.shape[0],
                              last_beam_idx + beam_radius + 1)

                # We select the top `beam_width` best attention values for the
                # next token.
                att_segment = att_values[att_start: att_end, token_idx - s]
                eff_beam_width = min(att_segment.shape[0], beam_width)
                sub_beam = att_segment.topk(k=eff_beam_width, dim=0)
                sub_beam_values, sub_beam_idxs = sub_beam
                sub_beam_idxs += att_start

                # If there were not enough attention values available, we pad
                # `sub_beam_idxs` with zeroes, to ensure that it is
                # `beam_width` long.
                trunc_length = sub_beam_idxs.shape[0]
                if trunc_length < beam_width:
                    sub_beam_idxs = torch.cat(
                        [sub_beam_idxs,
                         torch.zeros(beam_width - sub_beam_idxs.shape[0],
                                     dtype=torch.long)],
                        dim=0)

                # We compute new beam values and indices for these new
                # combinations
                combs = (beam_idx_tensor.repeat(beam_width, 1),
                         sub_beam_idxs.unsqueeze(dim=1))
                sub_beam_idxs = torch.cat(combs, dim=1)
                sub_beam_values = beam_values[i] + sub_beam_values

                # If we padded `sub_beam_idxs` with zeroes then we also pad
                # `sub_beam_values` with zeroes. We do this after adding
                # `beam_values[i]` above, to ensure that these combinations get
                # a zero value.
                if trunc_length < beam_width:
                    sub_beam_values = torch.cat(
                        [sub_beam_values,
                         torch.zeros(beam_width - sub_beam_values.shape[0],
                                     dtype=torch.float)],
                        dim=0)

                # We store these in the `all_beam_idxs` and `all_beam_values`
                # tensors
                start_sub_beam = i * beam_width
                end_sub_beam = (i + 1) * beam_width
                all_beam_idxs[start_sub_beam:end_sub_beam, :] = sub_beam_idxs
                all_beam_values[start_sub_beam:end_sub_beam] = sub_beam_values

            # We next keep only the `beam_width` best combinations of the
            # combinations.
            beam = all_beam_values.topk(k=beam_width, dim=0)
            beam_values = beam[0]
            beam_idxs = all_beam_idxs[beam[1]]

        # We now have the top `beam_width` best combinations of target tokens
        # stored in the `beam_idxs` tensor. We now extract the best combination
        # out of these, and return it.
        best_combination = beam_idxs[beam_values.argmax()]
        return best_combination.tolist(), beam_values.max()[0].item()
