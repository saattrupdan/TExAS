'''Config class for finetuning QA models'''

from pydantic import BaseModel
from typing import Tuple


class Config(BaseModel):
    '''Configuration for finetuning.

    Args:
        model_id (str, optional):
            The HuggingFace model id of the pretrained language model. Defaults
            to 'xlm-roberta-base'.
        batch_size (int, optional):
            The batch size for training. Defaults to 8.
        gradient_accumulation_steps (int, optional):
            The number of steps to accumulate gradients. Defaults to 4.
        epochs (int, optional):
            The number of epochs to train for. Defaults to 3.
        learning_rate (float, optional):
            The learning rate for the optimizer. Defaults to 2e-5.
        betas (Tuple[float, float], optional):
            The betas for the AdamW optimizer. Defaults to (0.9, 0.999).
        weight_decay (float, optional):
            The weight decay for the AdamW optimizer. Defaults to 0.01.
        max_length (int, optional):
            The maximum length of a sequence. Defaults to 384.
        doc_stride (int, optional):
            The stride of the sliding window. Defaults to 128.
        pad_on_right (bool, optional):
            Whether to pad on the right. Defaults to True.
        max_answer_length (int, optional):
            The maximum length of an answer. Defaults to 30.
    '''
    model_id: str = 'xlm-roberta-base'
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    epochs: int = 3
    learning_rate: float = 2e-5
    betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.01
    max_length: int = 384
    doc_stride: int = 128
    pad_on_right: bool = True
    max_answer_length: int = 30
