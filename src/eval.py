'''Evaluation script of question-answering models'''

from datasets import load_dataset, load_metric, Dataset
from transformers import AutoModelForQuestionAnswering, Trainer
from config import Config
from data_preparation import QAPreparer


def evaluate(model_id: str, language: str) -> dict:

    # Load XQuAD
    xquad = load_dataset('xquad', f'xquad.{language}', split='validation')
    #xquad = Dataset.from_json('datasets/squad_v2-validation-da.jsonl')

    print('Sample of dataset:', xquad[0])

    # Prepare XQuAD for evaluation
    config = Config(model_id=model_id)
    preparer = QAPreparer(config=config)
    prepared_xquad = preparer.prepare_test_dataset(xquad)

    # Load model
    model = AutoModelForQuestionAnswering.from_pretrained(model_id).eval()
    trainer = Trainer(model=model, tokenizer=preparer.tokenizer)

    # Get predictions and postprocess them
    predictions = trainer.predict(prepared_xquad)
    predictions = preparer.postprocess_predictions(
        test_dataset=xquad,
        prepared_test_dataset=prepared_xquad,
        predictions=predictions
    )

    # Load the metric
    metric = load_metric('squad_v2')

    # Compute metric scores
    predictions = [
        dict(id=k, prediction_text=v, no_answer_probability=0.)
        for k, v in predictions.items()
    ]
    references = [
        dict(id=example['id'],
             answers=dict(text=example['answers']['text'],
                          answer_start=example['answers']['answer_start']))
        for example in xquad
    ]
    scores = metric.compute(predictions=predictions, references=references)

    # Return EM and F1 scores
    return dict(em=scores['exact'], f1=scores['f1'])


if __name__ == '__main__':
    evaluation_models = [
        ('saattrupdan/xlmr-base-texas-squad-es', 'es'),
        ('PlanTL-GOB-ES/roberta-base-bne-sqac', 'es'),
    ]
    for model_id, language in evaluation_models:
        print(f'Evaluating {model_id} on {language}')
        scores = evaluate(model_id=model_id, language=language)
        print(f'EM: {scores["em"]}')
        print(f'F1: {scores["f1"]}')
