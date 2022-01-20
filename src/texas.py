'''Implementation of the TExAS algorithm for translating QA datasets.

This is an implementation of the model described in the paper:
    TODO
'''

import json
from pathlib import Path
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import M2M100Tokenizer, M2M100ForConditionalGeneration
import spacy
import re
import torch
from answer_extraction import extract_translated_answer


class Texas:
    '''The TExAS algorithm for translating QA datasets.

    This is an implementation of the model described in the paper:
        TODO
    '''
    sentence_splitter = spacy.load('en_core_web_sm')
    translation_tokenizer = (M2M100Tokenizer
                             .from_pretrained('facebook/m2m100_1.2B'))
    translation_model = (M2M100ForConditionalGeneration
                         .from_pretrained('facebook/m2m100_1.2B')
                         .eval())
    cache = tuple()

    def translate_dataset(self,
                          dataset_id: str = 'squad_v2',
                          split: str = 'train',
                          target_language: str = 'da'):
        '''Translate a SQuAD-like QA dataset.

        This will store the translated dataset in
        `./{dataset_id}-{split}-{target_language}.jsonl`.

        Args:
            dataset_id (str, optional):
                The dataset to translate. Must be a HuggingFace dataset ID.
                Defaults to 'squad_v2'.
            split (str, optional):
                The split of the dataset to translate. Defaults to 'train'.
            target_language (str, optional):
                The target language of the translation. Defaults to 'da'.
        '''
        # Set up the target JSONL file
        path = Path(f'{dataset_id}-{split}-{target_language}.jsonl')

        # If the file already exists, raise an exception
        if path.exists():
            raise FileExistsError(f'File {path} already exists.')

        # Create the file
        path.touch()

        # Initialise the dataset streaming
        dataset = load_dataset(dataset_id, split=split)

        # Shortened variables for the translation model
        tokenizer = self.translation_tokenizer
        model = self.translation_model
        nlp = self.sentence_splitter

        # Iterate over the dataset
        desc = f'Translating the {split} split of {dataset_id}'
        for example in tqdm(dataset, desc=desc, total=len(dataset)):

            # Get the context
            ctx = example['context']

            # If the context has already been translated, load the
            # translated context
            cache = self.cache
            if len(cache) and ctx == cache[0]:
                translated_context = cache[1]
                tokens = cache[2]
                translated_tokens = cache[3]
                charmap = cache[4]
                translated_charmap = cache[5]

            # Otherwise, translate the context
            else:
                # First, split the context into sentences
                sentences = [str(sent) for sent in nlp(ctx).sents]

                # Initialise lists containing the translations
                all_translations = list()
                all_tokens = list()
                all_translated_tokens = list()

                # Translate all sentences
                charstart = 0
                translated_charstart = 0
                charmap = list()
                translated_charmap = list()
                desc = 'Translating context'
                for sent_idx, sentence in enumerate(tqdm(sentences,
                                                         desc=desc,
                                                         leave=False)):

                    # print()
                    # print(f'"{sentence}"')
                    # print()

                    # Tokenize the sentence
                    tokens = tokenizer(sentence, return_tensors='pt')
                    tokens = tokens['input_ids']

                    # Get a mapping between the token IDs and the character
                    # intervals over which the token ranges over
                    token_strings = (tokenizer
                                     .convert_ids_to_tokens(tokens[0]))
                    local_charmap = list()
                    if sent_idx == 0:
                        local_charmap.append((charstart, charstart))
                    for i, token_string in enumerate(token_strings[1:-1]):
                        if i == 0 and sent_idx == 0:
                            token_string = token_string.strip('▁')
                        start = charstart
                        if len(local_charmap) > 0:
                            start = max(start, local_charmap[-1][-1])
                        end = start + len(token_string)
                        local_charmap.append((start, end))
                    if sent_idx == len(sentences) - 1:
                        last_idx = local_charmap[-1][-1]
                        local_charmap.append((last_idx, last_idx))

                    # Translate the tokenized sentence
                    bos_token = tokenizer.get_lang_id(target_language)
                    params = dict(forced_bos_token_id=bos_token)
                    translated_tokens = model.generate(tokens, **params)

                    # Get a mapping between the translated token IDs and the
                    # character intervals over which the token ranges over
                    token_strings = (
                        tokenizer.convert_ids_to_tokens(translated_tokens[0])
                    )
                    translated_local_charmap = list()
                    if translated_charstart == 0:
                        nochars = (translated_charstart, translated_charstart)
                        translated_local_charmap.append(nochars)
                    for i, token_string in enumerate(token_strings[2:-1]):
                        if i == 0 and sent_idx == 0:
                            token_string = token_string.strip('▁')
                        start = translated_charstart
                        if len(translated_local_charmap) > 0:
                            start = max(start,
                                        translated_local_charmap[-1][-1])
                        end = start + len(token_string)
                        translated_local_charmap.append((start, end))
                    if sent_idx == len(sentences) - 1:
                        last_idx = translated_local_charmap[-1][-1]
                        translated_local_charmap.append((last_idx, last_idx))

                    # Decode the translated tokens
                    params = dict(skip_special_tokens=True)
                    translation = tokenizer.batch_decode(translated_tokens,
                                                         **params)[0]

                    # Append the translation to the context
                    all_translations.append(translation)
                    all_tokens.append(tokens[0])
                    all_translated_tokens.append(translated_tokens[0])
                    charmap += local_charmap
                    translated_charmap += translated_local_charmap

                    # Update `charstart`, to offset the character indices for
                    # the other sentences
                    charstart += len(sentence)
                    translated_charstart += len(translation)
                    if sent_idx > 0:
                        charstart += 1
                        translated_charstart += 1

                # Aggregate the translations and the tokens
                translated_context = ' '.join(all_translations)
                tokens = torch.cat(
                    [torch.tensor([all_tokens[0][0]])] +
                    [tok[1:-1] for tok in all_tokens] +
                    [torch.tensor([all_tokens[-1][-1]])]
                )
                translated_tokens = torch.cat(
                    [torch.tensor([all_translated_tokens[0][1]])] +
                    [tok[2:-1] for tok in all_translated_tokens] +
                    [torch.tensor([all_translated_tokens[-1][-1]])]
                )

                # Strip the translated context of leading and trailing
                # whitespace
                translated_context = translated_context.strip()

                # Store the translated context
                cache = (ctx,
                         translated_context,
                         tokens,
                         translated_tokens,
                         charmap,
                         translated_charmap)
                self.cache = cache

            # Get the cross attentions
            dec_in_ids = translated_tokens.unsqueeze(0)
            cross_attentions = (model.forward(tokens,
                                              decoder_input_ids=dec_in_ids,
                                              output_attentions=True)
                                     .cross_attentions[0][0])

            # Get the answer token IDs
            answers = dict(text=list(), answer_start=list())
            for answer, char_s in zip(example['answers']['text'],
                                      example['answers']['answer_start']):
                char_e = char_s + len(answer)
                token_s = [idx for idx, (start, end) in enumerate(charmap)
                           if start <= char_s and char_s < end][0]
                token_e = [idx for idx, (start, end) in enumerate(charmap)
                           if start <= char_e and char_e < end][0]

                # Extract the translated token IDs
                idxs = extract_translated_answer(
                    answer_token_idx_start=token_s,
                    answer_token_idx_end=token_e,
                    cross_attention_tensor=cross_attentions
                )

                # Attach extra tokens to the translated tokens, to "complete
                # the last word". This adds tokens until punctuation or a space
                # is met.
                regex = '(▁.*|[.,:;]+)'
                token_strings = (
                    tokenizer.convert_ids_to_tokens(translated_tokens)
                )
                while not (max(idxs) + 1 == len(token_strings) or
                           re.match(regex, token_strings[max(idxs) + 1])):
                    idxs.append(max(idxs) + 1)

                # Convert the token IDs to character IDs
                min_token_idx = min(idxs)
                max_token_idx = max(idxs)
                min_char_idx = translated_charmap[min_token_idx][0]
                max_char_idx = translated_charmap[max_token_idx][1]

                # Ensure that the answer does not start with a space
                if translated_context[min_char_idx] == ' ':
                    min_char_idx += 1

                # pred_toks = [ctx[s:e] for s, e in charmap]
                # true_toks = tokenizer.convert_ids_to_tokens(tokens)
                # print(list(zip(pred_toks, true_toks)))

                # pred_toks = [translated_context[s:e] for s, e in translated_charmap]
                # true_toks = tokenizer.convert_ids_to_tokens(translated_tokens)
                # print(list(zip(pred_toks, true_toks)))

                # Store the translated answer
                answer = translated_context[min_char_idx:max_char_idx]
                answers['text'].append(answer)
                answers['answer_start'].append(min_char_idx)

            # Translate the question
            question = example['question']
            tokens = tokenizer(question, return_tensors='pt')['input_ids']
            bos_token = tokenizer.get_lang_id(target_language)
            params = dict(forced_bos_token_id=bos_token)
            translated_tokens = model.generate(tokens, **params)
            params = dict(skip_special_tokens=True)
            translated_question = tokenizer.batch_decode(translated_tokens,
                                                         **params)[0]

            # Store the translated example
            new_example = dict(
                id=example['id'],
                title=example['title'],
                context=translated_context,
                question=translated_question,
                answers=answers
            )

            # Append the translated example to the JSONL file
            with path.open('a') as f:
                f.write(json.dumps(new_example) + '\n')


if __name__ == '__main__':
    texas = Texas()
    texas.translate_dataset(dataset_id='squad_v2',
                            split='train',
                            target_language='da')
