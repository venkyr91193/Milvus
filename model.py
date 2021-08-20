import logging
from typing import List

import numpy as np
from transformers import AutoModel, AutoTokenizer


class Model:
    __slots__ = [
        "tokenizer",
        "model",
    ]

    def __init__(
        self, model_name: str
    ):
        self.tokenizer = None
        self.model = None
        self.initialize(model_name)

    def initialize(self, model_name: str):
        """
        Function: To initialize the model
        """
        # similarity model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            logging.info(f"Initialized model")
        except Exception as e:
            logging.error(
                f"Error in initializing model. {e}"
            )

    def make_sentence_vectors(self, sentences_to_vectorize: List[str]):
        """
        Function: Function to compare the sentences using vectors
        """
        return_vectors = list()
        for sentence in sentences_to_vectorize:
            temp_out = list()
            token_ids = self.tokenizer.encode(
                sentence,
                add_special_tokens=True,
                max_length=512,
                truncation=True,
                return_tensors="pt",
            )
            # last layer is the one which has the information for the entire sentence
            last_layer = self.model(token_ids)[0]
            # average all the vectors across each token for getting one single vector 1D output
            # converting from 2D -> 1D by avering across
            repr_ = last_layer[0][0]
            for idx in range(last_layer.shape[1] - 1):
                repr_ += last_layer[0][idx + 1]
            repr_ = repr_ / last_layer.shape[1]
            temp_out = repr_.tolist()
            # normalize the vectors to be store in milvus for IP: InnerProduct
            # milvus only accepts lists, NO numpy or tensors
            temp_out = temp_out / np.linalg.norm(temp_out)
            return_vectors.append(temp_out.tolist())
        return return_vectors
