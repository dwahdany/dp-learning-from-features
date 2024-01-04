from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class ClassificationScheme(ABC):
    @abstractmethod
    def classify(self, v_pred, m_protos):
        assert (
            len(v_pred.shape) == 1
        ), f"Expected 1-D sample vector, got shape {v_pred.shape}"
        assert (
            len(m_protos.shape) == 2
        ), f"Expected 2-D matrix of prototypes, got shape {m_protos.shape}"
        assert (
            v_pred.shape[0] == m_protos.shape[1]
        ), f"Expected same dimensionality of sample and each class prototype, but got shapes {v_pred.shape} and {m_protos.shape}"


class CosineClassification(ClassificationScheme):
    name: str = "cosine"

    def classify(self, v_pred, m_protos):
        super().classify(v_pred, m_protos)
        return np.argmax(cosine_similarity(v_pred.reshape(1, -1), m_protos))


class EuclideanClassification(ClassificationScheme):
    name: str = "euclidean"

    def classify(self, v_pred, m_protos):
        super().classify(v_pred, m_protos)
        return np.argmin(np.linalg.norm(v_pred - m_protos, ord=2, axis=1))
