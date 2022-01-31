'''Class used to prepare data for training and testing of QA models.

This is based on the example at:
    https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/question_answering.ipynb
'''
from datasets.dataset_dict import DatasetDict
from datasets.arrow_dataset import Dataset
from transformers import AutoTokenizer
import os

from config import Config


class QAPreparer:
    '''Class used to prepare data for training and testing of QA models.

    Args:
        config (Config):
            Configuration object.
    '''
    def __init__(self, config: Config):
        self.config = config

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)

        # Disable tokenizers parallelism
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    def prepare_train_val_datasets(self,
                                   dataset_dict: DatasetDict) -> DatasetDict:
        '''Prepare training dataset.

        Args:
            dataset_dict (DatasetDict):
                The dictionary containing the training and validation splits.

        Returns:
            DatasetDict:
                Prepared dataset.
        '''
        feature_names = dataset_dict['train'].column_names
        prepared = dataset_dict.map(self._prepare_train_examples,
                                    batched=True,
                                    remove_columns=feature_names)
        return prepared

    def prepare_test_dataset(self, test_dataset: Dataset) -> Dataset:
        '''Prepare test dataset.

        Args:
            test_dataset (Dataset):
                The dataset used for testing.

        Returns:
            Dataset:
                Prepared test dataset.
        '''
        feature_names = test_dataset.column_names
        prepared = test_dataset.map(self._prepare_test_examples,
                                    batched=True,
                                    remove_columns=feature_names)
        prepared.set_format(type=prepared.format["type"],
                            columns=list(prepared.features.keys()))
        return prepared

    def postprocess_predictions(self, predictions):
        pass

    def _prepare_train_examples(self, examples: dict) -> dict:
        '''Prepare training examples.

        Args:
            examples (dict):
                Dictionary of training examples.

        Returns:
            dict:
                Dictionary of training examples.
        '''
        # Convenience abbrevations
        tokenizer = self.tokenizer
        config = self.config

        # Some of the questions have lots of whitespace on the left, which is not
        # useful and will make the truncation of the context fail (the tokenized
        # question will take a lots of space). So we remove that left whitespace
        examples["question"] = [q.lstrip() for q in examples["question"]]

        # Tokenize our examples with truncation and padding, but keep the overflows
        # using a stride. This results in one example possible giving several
        # features when a context is long, each of those features having a context
        # that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples["question" if config.pad_on_right else "context"],
            examples["context" if config.pad_on_right else "question"],
            truncation="only_second" if config.pad_on_right else "only_first",
            max_length=config.max_length,
            stride=config.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        # Since one example might give us several features if it has a long
        # context, we need a map from a feature to its corresponding example. This
        # key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # The offset mappings will give us a map from token to character position
        # in the original context. This will help us compute the start_positions
        # and end_positions.
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):

            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the
            # context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # One example can give several spans, this is the index of the example
            # containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples["answers"][sample_index]

            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)

            # Otherwise, we extract the start and end token indices of the answer
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # Start token index of the current span in the text.
                token_start_index = 0
                context_id = 1 if config.pad_on_right else 0
                while sequence_ids[token_start_index] != context_id:
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != context_id:
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this
                # feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and
                        offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to
                    # the two ends of the answer. Note: we could go after the last
                    # offset if the answer is the last word (edge case).
                    while (token_start_index < len(offsets) and
                            offsets[token_start_index][0] <= start_char):
                        token_start_index += 1
                    token_start_index -= 1
                    tokenized_examples["start_positions"].append(token_start_index)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    token_end_index += 1
                    tokenized_examples["end_positions"].append(token_end_index)

        return tokenized_examples

    def _prepare_test_examples(self, examples: dict) -> dict:
        '''Prepare test examples.

        Args:
            examples (dict):
                Dictionary of test examples.

        Returns:
            dict:
                Dictionary of prepared test examples.
        '''
        # Convenience abbrevations
        tokenizer = self.tokenizer
        config = self.config

        # Some of the questions have lots of whitespace on the left, which is
        # not useful and will make the truncation of the context fail (the
        # tokenized question will take a lots of space). So we remove that left
        # whitespace
        examples["question"] = [q.lstrip() for q in examples["question"]]

        # Tokenize our examples with truncation and maybe padding, but keep the
        # overflows using a stride. This results in one example possible giving
        # several features when a context is long, each of those features
        # having a context that overlaps a bit the context of the previous
        # feature.
        tokenized_examples = tokenizer(
            examples["question" if config.pad_on_right else "context"],
            examples["context" if config.pad_on_right else "question"],
            truncation="only_second" if config.pad_on_right else "only_first",
            max_length=config.max_length,
            stride=config.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        # Since one example might give us several features if it has a long
        # context, we need a map from a feature to its corresponding example.
        # This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # We keep the example_id that gave us this feature and we will store
        # the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):

            # Grab the sequence corresponding to that example (to know what is
            # the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if config.pad_on_right else 0

            # One example can give several spans, this is the index of the
            # example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context
            # so it's easy to determine if a token position is part of the
            # context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples
