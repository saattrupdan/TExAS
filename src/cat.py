'''Cross-attention Aligned Translation for Question Answering.

This is an implementation of the model described in the paper:
    TODO
'''

from pydantic import BaseModel
from .answer_extraction import extract_translated_answer


class CAT(BaseModel):
    '''Cross-attention Aligned Translation for Question Answering.

    This is an implementation of the model described in the paper:
        TODO

    Args:
        target_language (str):
            The target language of the translation.
        dataset (str, optional):
            The dataset to use. Defaults to 'squad_v2'.
        translation_model (str, optional):
            The translation model to use. Defaults to 'facebook/m2m100_1.2B'.
    '''
    target_language: str
    dataset: str = 'squad_v2'
    translation_model: str = 'facebook/m2m100_1.2B'
