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
from typing import Union, List, Tuple, Callable, Optional
from answer_extraction import extract_translated_answer


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Texas:
    '''The TExAS algorithm for translating QA datasets.

    This is an implementation of the model described in the paper:
        TODO
    '''
    translation_tokenizer = (M2M100Tokenizer
                             .from_pretrained('facebook/m2m100_1.2B'))
    translation_model = (M2M100ForConditionalGeneration
                         .from_pretrained('facebook/m2m100_1.2B')
                         .eval()
                         .to(device))
    cache = tuple()

    def _translate(self,
                   doc: str,
                   target_language: str,
                   return_tokens: bool = False) -> Union[str, tuple]:
        '''Translate a single document.

        Args:
            doc (str):
                The document to translate.
            target_language (str):
                The target language.
            return_tokens (bool):
                Whether to also return the tokens of the input and the
                translation.

        Returns:
            str or triple:
                Either the translated document, if `return_tokens` is False,
                or a triple of the form (translation, input_tokens,
                translation_tokens).
        '''
        with torch.no_grad():
            tokenizer = self.translation_tokenizer
            model = self.translation_model
            tokens = tokenizer(doc, return_tensors='pt')['input_ids']
            bos_token = tokenizer.get_lang_id(target_language)
            params = dict(forced_bos_token_id=bos_token)

            try:
                translated_tokens = model.generate(tokens.to(device), **params)
                translated_tokens = translated_tokens.cpu()
            except:
                model.cpu()
                translated_tokens = model.generate(tokens, **params)
                model.to(device)

            params = dict(skip_special_tokens=True)
            translation = tokenizer.batch_decode(translated_tokens, **params)
            if return_tokens:
                return translation[0], tokens, translated_tokens
            else:
                return translation[0]

    def _extract_answer(self,
                        char_start_idx: int,
                        char_end_idx: int,
                        charmap: List[Tuple[int, int]],
                        translated_charmap: List[Tuple[int, int]],
                        translated_tokens: torch.Tensor,
                        translated_context: str,
                        cross_attentions: torch.Tensor,
                        return_token_indices: bool = False) -> Tuple[int, int]:
        '''Extract the answer from the translated context.

        Args:
            char_start_idx (int):
                The start index of the answer in the original context.
            char_end_idx (int):
                The end index of the answer in the original context.
            charmap (list of pairs of ints):
                The character mapping of the original context.
            translated_charmap (list of pairs of ints):
                The character mapping of the translated context.
            translated_tokens (torch.Tensor):
                The tokens of the translated context.
            translated_context (str):
                The translated context.
            cross_attentions (torch.Tensor):
                The cross attentions of the translated context.
            return_token_indices (bool):
                Whether to also return the indices of the tokens in the
                translated context.

        Returns:
            pair of ints:
                The start and end character index of the answer in the
                translated context.
        '''
        # Abbreviate `char_start_idx` and `char_end_idx`
        char_s, char_e = char_start_idx, char_end_idx

        # Use the character mapping to convert the character indices
        # to token indices
        try:
            token_s = [idx for idx, (start, end) in enumerate(charmap)
                       if start <= char_s and char_s <= end][0]
        except IndexError:
            closest_left = max([idx for idx, (_, end) in enumerate(charmap)
                                if end <= char_s])
            closest_right = min([idx for idx, (start, _) in enumerate(charmap)
                                 if char_s <= start])
            left_distance = char_s - charmap[closest_left][1]
            right_distance = charmap[closest_right][0] - char_s
            if left_distance <= right_distance:
                token_s = closest_left
            else:
                token_s = closest_right

        token_e = max([idx for idx, (start, _) in enumerate(charmap)
                       if start < char_e])

        # Extract the translated token IDs
        idxs = extract_translated_answer(
            answer_token_idxs=range(token_s, token_e + 1),
            cross_attention_tensor=cross_attentions
        )

        # Attach extra tokens to the translated tokens, to "complete
        # the last word". This adds tokens until punctuation or a space
        # is met.
        token_strings = (
            self.translation_tokenizer.convert_ids_to_tokens(translated_tokens)
        )
        regex = '(▁.*|[!?.,:;]+.*)'
        while not (max(idxs) + 1 == len(token_strings) or
                   re.match(regex, token_strings[max(idxs) + 1])):
            idxs.append(max(idxs) + 1)

        # Convert the token IDs to character IDs
        min_token_idx = min(idxs)
        max_token_idx = max(idxs)
        min_char_idx = max(0, min(translated_charmap[min_token_idx][0],
                                  len(translated_context) - 1))
        max_char_idx = max(0, min(translated_charmap[max_token_idx][1],
                                  len(translated_context) - 1))

        # Ensure that the answer does not end with punctuation or a
        # space
        while translated_context[max_char_idx - 1] in ' !?.,:;)("\'':
            max_char_idx -= 1

        # Ensure that `min_char_idx` not larger than `max_char_idx`
        min_char_idx = min(min_char_idx, max_char_idx)

        # Ensure that the answer does not start with a space or punctuation
        while (min_char_idx < max_char_idx and
               translated_context[min_char_idx] in ' !?.,:;)("\''):
            min_char_idx += 1

        if return_token_indices:
            return min_char_idx, max_char_idx, min_token_idx, max_token_idx
        else:
            return min_char_idx, max_char_idx

    def _compute_cross_attentions(self,
                                  tokens: torch.Tensor,
                                  translated_tokens: torch.Tensor
                                  ) -> torch.Tensor:
        '''Compute the cross attentions of the tokens and translated tokens.

        Args:
            tokens (torch.Tensor):
                The tokens of the original context.
            translated_tokens (torch.Tensor):
                The tokens of the translated context.

        Returns:
            torch.Tensor:
                The cross attentions of the tokens and translated tokens, of
                shape (num_attention_heads, num_translated_tokens, num_tokens).
        '''
        with torch.no_grad():
            model = self.translation_model
            dec_in_ids = translated_tokens.unsqueeze(0)
            try:
                outputs = model.forward(
                    tokens.to(device),
                    decoder_input_ids=dec_in_ids.to(device),
                    output_attentions=True
                )
            except:
                model.cpu()
                outputs = model.forward(
                    tokens,
                    decoder_input_ids=dec_in_ids,
                    output_attentions=True
                )
                model.to(device)
            cross_attentions = outputs.cross_attentions[0][0].cpu()
            return cross_attentions

    def translate_dataset(
            self,
            dataset_id: str = 'squad_v2',
            dataset_subset_id: Optional[str] = None,
            split: str = 'train',
            target_language: str = 'da',
            sentence_splitter: str = 'en_core_web_sm',
            context_fn: Callable = lambda x: x['context'],
            question_fn: Callable = lambda x: x['question'],
            answer_fn: Callable = lambda x: x['answers']['text'],
            answer_idx_fn: Callable = lambda x: x['answers']['answer_start'],
            id_fn: Callable = lambda x: x['id'],
            title_fn: Callable = lambda x: x['title']):
        '''Translate a SQuAD-like QA dataset.

        This will store the translated dataset in
        `./{dataset_id}-{split}-{target_language}.jsonl`.

        Args:
            dataset_id (str, optional):
                The dataset to translate. Must be a HuggingFace dataset ID.
                Defaults to 'squad_v2'.
            dataset_subset_id (str or None, optional):
                The subset of the dataset to translate, if there are multiple
                to choose between. If None, it is assumed that there is only
                one subset. Defaults to None.
            split (str, optional):
                The split of the dataset to translate. Defaults to 'train'.
            target_language (str, optional):
                The target language of the translation. Defaults to 'da'.
            sentence_splitter (str, optional):
                The SpaCy model to use for splitting sentences. Defaults to
                'en_core_web_sm'.
            context_fn (callable, optional):
                A function that extracts the context from a dataset example.
                Defaults to `lambda x: x['context']`, corresponding to the
                standard SQuAD setup.
            question_fn (callable, optional):
                A function that extracts the question from a dataset example.
                Defaults to `lambda x: x['question']`, corresponding to the
                standard SQuAD setup.
            answer_fn (callable, optional):
                A function that extracts the answer from a dataset example.
                Defaults to `lambda x: x['answers']['text']`, corresponding
                to the standard SQuAD setup.
            answer_idx_fn (callable, optional):
                A function that extracts the answer index from a dataset
                example. Defaults to `lambda x: x['answers']['answer_start']`,
                corresponding to the standard SQuAD setup.
            id_fn (callable, optional):
                A function that extracts the ID from a dataset example.
                Defaults to `lambda x: x['id']`, corresponding to the standard
                SQuAD setup.
            title_fn (callable, optional):
                A function that extracts the title from a dataset example.
                Defaults to `lambda x: x['title']`, corresponding to the
                standard SQuAD setup.
        '''
        # Set up the target JSONL file
        if dataset_subset_id is None:
            dataset_name = dataset_id.replace('/', '-')
        else:
            dataset_name = dataset_id.replace('/', '-')
            dataset_name = f'{dataset_name}-{dataset_subset_id}'
        path = Path(f'{dataset_name}-{split}-{target_language}.jsonl')

        # If the file already exists, raise an exception
        if path.exists():
            raise FileExistsError(f'File {path} already exists.')

        # Create the file
        path.touch()

        # Initialise the dataset streaming
        dataset = load_dataset(dataset_id, dataset_subset_id, split=split)

        # Shortened variables for the translation model
        tokenizer = self.translation_tokenizer

        # Load the sentence splitter model
        nlp = spacy.load(sentence_splitter)

        # Iterate over the dataset
        desc = f'Translating the {split} split of {dataset_name}'
        for example in tqdm(dataset, desc=desc, total=len(dataset)):

            # Get the context
            ctx = context_fn(example)

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

                    # Translate the tokenized sentence
                    translation, tokens, translated_tokens = self._translate(
                        doc=sentence,
                        target_language=target_language,
                        return_tokens=True
                    )

                    # Get a mapping between the token IDs and the character
                    # intervals over which the token ranges over
                    token_strings = (
                        tokenizer.convert_ids_to_tokens(tokens[0])
                    )
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
                        if len(local_charmap) > 0:
                            last_idx = local_charmap[-1][-1]
                        else:
                            last_idx = charstart
                        local_charmap.append((last_idx, last_idx))

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

            # Translate the question
            translated_question = self._translate(
                doc=question_fn(example),
                target_language=target_language
            )

            # Iterate over all the answers
            computed_cross_attentions = False
            computed_s_and_e = False
            answers = dict(text=list(),
                           answer_start=list(),
                           extraction_method=list())
            for answer, char_s in zip(answer_fn(example),
                                      answer_idx_fn(example)):

                # CASE 1: Check if the (non-translated) answer appears uniquely
                # in the translated context
                if translated_context.lower().count(answer.lower()) == 1:
                    answer_start = (translated_context.lower()
                                                      .index(answer.lower()))
                    answer_end = answer_start + len(answer)
                    if (answer_end >= len(translated_context) or
                            translated_context[answer_end] in '[ !?.,:;]'):
                        answers['answer_start'].append(answer_start)
                        answers['text'].append(answer)
                        answers['extraction_method'].append('unique')
                        continue


                # CASE 2: Check if the translated answer appears uniquely in
                # the translated context
                translated_answer = self._translate(
                    doc=answer,
                    target_language=target_language
                )
                if (translated_context.lower()
                                      .count(translated_answer.lower()) == 1):
                    answer_start = (translated_context
                                    .lower()
                                    .index(translated_answer.lower()))
                    answer_end = answer_start + len(translated_answer)
                    if (answer_end >= len(translated_context) or
                            translated_context[answer_end] in '[ !?.,:;]'):
                        answer = translated_context[answer_start:answer_end]
                        method = 'translated_unique'
                        answers['answer_start'].append(answer_start)
                        answers['text'].append(answer)
                        answers['extraction_method'].append(method)
                        continue


                # CASE 3: Check if the (non-translated) answer appears at all
                # in the translated context
                if answer.lower() in translated_context.lower():

                    # Get the cross attentions
                    if not computed_cross_attentions:
                        cross_attentions = self._compute_cross_attentions(
                            tokens=tokens,
                            translated_tokens=translated_tokens
                        )
                        computed_cross_attentions = True

                    # Use the cross attentions to find the rough location of
                    # the answer
                    if not computed_s_and_e:
                        s, e, ts, te = self._extract_answer(
                            char_start_idx=max(0, char_s - 20),
                            char_end_idx=min(len(ctx),
                                             char_s + len(answer) + 20),
                            charmap=charmap,
                            translated_charmap=translated_charmap,
                            translated_tokens=translated_tokens,
                            translated_context=translated_context,
                            cross_attentions=cross_attentions,
                            return_token_indices=True
                        )
                        computed_s_and_e = True

                    translated_ctx_segment = translated_context[s:e]
                    if (translated_ctx_segment
                            .lower()
                            .count(answer.lower()) == 1):
                        answer_start = (translated_ctx_segment
                                        .lower()
                                        .index(answer.lower())) + s
                        answer_end = answer_start + len(answer)
                        if (answer_end >= len(translated_context) or
                                translated_context[answer_end] in '[ !?.,:;]'):
                            answers['answer_start'].append(answer_start)
                            answers['text'].append(answer)
                            answers['extraction_method'].append('att+unique')
                            continue


                # CASE 4: Check if the translated answer appears at all in the
                # translated context
                if translated_answer.lower() in translated_context.lower():

                    # Get the cross attentions
                    if not computed_cross_attentions:
                        cross_attentions = self._compute_cross_attentions(
                            tokens=tokens,
                            translated_tokens=translated_tokens
                        )
                        computed_cross_attentions = True

                    # Use the cross attentions to find the rough location of
                    # the answer
                    if not computed_s_and_e:
                        s, e, ts, te = self._extract_answer(
                            char_start_idx=max(0, char_s - 20),
                            char_end_idx=min(len(ctx),
                                             char_s + len(answer) + 20),
                            charmap=charmap,
                            translated_charmap=translated_charmap,
                            translated_tokens=translated_tokens,
                            translated_context=translated_context,
                            cross_attentions=cross_attentions,
                            return_token_indices=True
                        )
                        computed_s_and_e = True

                    translated_ctx_segment = translated_context[s:e]
                    if (translated_ctx_segment
                            .lower()
                            .count(translated_answer.lower()) == 1):
                        answer_start = (translated_ctx_segment
                                        .lower()
                                        .index(translated_answer.lower())) + s
                        answer_end = answer_start + len(translated_answer)
                        if (answer_end >= len(translated_context) or
                                translated_context[answer_end] in '[ !?.,:;]'):
                            ans = translated_context[answer_start:answer_end]
                            method = 'att+translated_unique'
                            answers['answer_start'].append(answer_start)
                            answers['text'].append(ans)
                            answers['extraction_method'].append(method)
                            continue


                # CASE 5: Use cross-attentions to find the answer in the
                # translated context

                # Get the cross attentions
                if not computed_cross_attentions:
                    cross_attentions = self._compute_cross_attentions(
                        tokens=tokens,
                        translated_tokens=translated_tokens
                    )
                    computed_cross_attentions = True

                # Use the cross attentions to find the location of the
                # translated answer
                precise_s, precise_e = self._extract_answer(
                    char_start_idx=max(0, char_s),
                    char_end_idx=min(len(ctx), char_s + len(answer)),
                    charmap=charmap,
                    translated_charmap=translated_charmap,
                    translated_tokens=translated_tokens,
                    translated_context=translated_context,
                    cross_attentions=cross_attentions
                )

                # Store the translated answer
                answer = translated_context[precise_s:precise_e]
                answers['text'].append(answer)
                answers['answer_start'].append(precise_s)
                answers['extraction_method'].append('cross-attention')

            # Store the translated example
            new_example = dict(
                id=id_fn(example),
                title=title_fn(example),
                context=translated_context,
                question=translated_question,
                answers=answers,
                original_dataset=dataset_id
            )

            # Append the translated example to the JSONL file
            with path.open('a') as f:
                f.write(json.dumps(new_example) + '\n')


if __name__ == '__main__':
    texas = Texas()

    # SQuAD 2.0
    params = dict(dataset_id='squad_v2',
                  target_language='da')
    for split in ['train', 'validation']:
        texas.translate_dataset(split=split, **params)

    # Adversarial QA
    params = dict(dataset_id='adversarial_qa',
                  dataset_subset_id='adversarialQA',
                  target_language='da')
    for split in ['train', 'validation', 'test']:
        texas.translate_dataset(split=split, **params)

    # SberQuAD
    params = dict(dataset_id='sberquad',
                  target_language='da',
                  sentence_splitter='ru_core_news_sm')
    for split in ['train', 'validation', 'test']:
        texas.translate_dataset(split=split, **params)

    # GermanQuAD
    def answer_idx_fn(example: dict):
        answer_start = example['answers']['answer_start']
        context_prefixes = example['context'].split('===')[:-1]
        context_prefix_len = len('==='.join(context_prefixes))
        context = example['context'].split('===')[-1]
        num_context_newlines = len(context) - len(context.strip('\n'))
        return [start_idx - context_prefix_len - num_context_newlines
                for start_idx in answer_start]
    context_fn = lambda x: x['context'].split('===')[-1].strip('\n')
    params = dict(dataset_id='deepset/germanquad',
                  target_language='da',
                  sentence_splitter='de_core_news_sm',
                  title_fn=lambda x: x['context'].split('===')[0].strip('\n'),
                  context_fn=context_fn,
                  answer_idx_fn=answer_idx_fn)
    for split in ['train', 'test']:
        texas.translate_dataset(split=split, **params)
