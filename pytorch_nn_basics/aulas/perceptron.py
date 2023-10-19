from utils import Aula
import torch
from typing import Union, List
import numpy as np

class AulaPerceptron(Aula):
    def __init__(self) -> None:
        super().__init__()
        
    def mostra_conteudo_sobre(self, conteudo: Union[str, List[str]]) -> None:
        return super().mostra_conteudo_sobre(conteudo) 
    
    def equacao_da_reta(self) -> str:
        """_summary_
            É aprendida na forma ax+by+c, mas para nos adequarmos às nomenclaturas de redes neurais,
            podemos reescrever essa equação como w1x1 + w2x2 + b. Ou seja, w1,w2 e b são parâmetros que definem um modelo linear,
            a reta
        Returns:
            str: _description_
        """
        pass