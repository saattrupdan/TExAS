'''Training script for finetuning a QA model'''

from datasets.dataset_dict import DatasetDict
from transformers import (AutoTokenizer,
                          AutoModelForQuestionAnswering,
                          TrainingArguments,
                          default_data_collator,
                          Trainer)
from pydantic import BaseModel
from typing import Tuple
from functools import partial
import os


class Config(BaseModel):
    '''Configuration for finetuning'''
    model_id: str = 'xlm-roberta-base'
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    epochs: int = 5
    learning_rate: float = 2e-5
    betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.01
    max_length: int = 384
    doc_stride: int = 128
    pad_on_right: bool = True


def train(dataset_dict: DatasetDict, output_model_id: str, config: Config):
    '''Finetune a pretrained model on a question-answering dataset.

    Args:
        dataset_dict (DatasetDict):
            The train/val splits of the question-answering dataset to finetune
            the model on.
        output_model_id (str):
            The name of the output model.
        config (Config):
            The configuration for the finetuning.
    '''
    # Disable tokenizers parallelism
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    model = AutoModelForQuestionAnswering.from_pretrained(config.model_id)

    # Prepare the dataset
    feature_names = dataset_dict['train'].column_names
    prepare_fn = partial(prepare_train_features,
                         tokenizer=tokenizer,
                         config=config)
    tokenized_datasets = dataset_dict.map(prepare_fn,
                                          batched=True,
                                          remove_columns=feature_names)

    # Prepare the training arguments
    args = TrainingArguments(
        output_dir=output_model_id.split('/')[-1],
        evaluation_strategy = 'epoch',
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        num_train_epochs=config.epochs,
        weight_decay=config.weight_decay,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        adam_beta1=config.betas[0],
        adam_beta2=config.betas[1],
        push_to_hub=True,
        hub_model_id=output_model_id
    )

    # Initialise the trainer
    trainer = Trainer(
        model,
        args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        data_collator=default_data_collator,
        tokenizer=tokenizer,
    )

    # Finetune the model
    trainer.train()

    # Save the model
    trainer.save_model(output_model_id.split('/')[-1])

    # TODO: Evaluation


def prepare_train_features(examples: dict, tokenizer, config: Config) -> dict:
    '''Prepare the features for training.

    Args:
        examples (dict):
            The examples to prepare the features for.
        tokenizer (Tokenizer):
            The tokenizer to use for the features.
        config (Config):
            The configuration for the finetuning.

    Returns:
        dict:
            The prepared features.
    '''
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


if __name__ == "__main__":

    # Load dataset dict
    dataset_dict = DatasetDict.from_json(dict(
        train='datasets/squad_v2-train-da.jsonl',
        validation='datasets/squad_v2-validation-da.jsonl'
    ))

    # Load config
    config = Config()

    # Train the model
    train(dataset_dict,
          output_model_id='saattrupdan/xlmr-base-texas-squad-da',
          config=config)
