'''Training script for finetuning a question-answering model.

This is based on the example at:
    https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/question_answering.ipynb
'''
from datasets import DatasetDict, load_metric
from transformers import (AutoModelForQuestionAnswering,
                          TrainingArguments,
                          default_data_collator,
                          Trainer)

from data_preparation import QAPreparer
from config import Config


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
    # Initialise the QA preparer
    preparer = QAPreparer(config)

    # Prepare the dataset for training
    dataset_dict = preparer.prepare_train_val_datasets(dataset_dict)

    # Load the model
    model = AutoModelForQuestionAnswering.from_pretrained(config.model_id)

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
        train_dataset=dataset_dict['train'],
        eval_dataset=dataset_dict['validation'],
        data_collator=default_data_collator,
        tokenizer=preparer.tokenizer,
    )

    # Finetune the model
    trainer.train()

    # Save the model
    trainer.save_model()

    # Prepare the test dataset
    test_dataset = preparer.prepare_test_dataset(dataset_dict['validation'])

    # Get test predictions
    raw_predictions = trainer.predict(test_dataset)

    # Postprocess the predictions
    predictions = preparer.postprocess_predictions(raw_predictions)

    # Load metric
    metric = load_metric('squad_v2')

    # Compute metric score
    references = [dict(id=example['id'], answers=example['answers'])
                  for example in dataset_dict["validation"]]
    scores = metric.compute(predictions=predictions, references=references)
    em_score = scores['exact_match']
    f1_score = scores['f1']

    # Print the results
    print(f'EM: {em_score:.3f}')
    print(f'F1: {f1_score:.3f}')

    # Push to hub
    trainer.push_to_hub()


if __name__ == "__main__":
    # Set language code
    LANGUAGE_CODE = 'da'

    # Load dataset dict
    dataset_dict = DatasetDict.from_json(dict(
        train=f'datasets/squad_v2-train-{LANGUAGE_CODE}.jsonl',
        validation=f'datasets/squad_v2-validation-{LANGUAGE_CODE}.jsonl'
    ))

    # Load config
    config = Config()

    # Train the model
    train(dataset_dict,
          output_model_id=f'saattrupdan/xlmr-base-texas-squad-{LANGUAGE_CODE}',
          config=config)