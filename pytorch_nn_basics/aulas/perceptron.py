from utils import Aula
import torch
from typing import Union, List
import numpy as np
from matplotlib import pyplot as plt


class AulaPerceptron(Aula):
    def __init__(self) -> None:
        super().__init__()
        self.res = ""

        self.mostra_conteudo_sobre([self.equacao_da_reta(-1, 4, 0.4)])

    def mostra_conteudo_sobre(self, conteudo: Union[str, List[str]]) -> None:
        return super().mostra_conteudo_sobre(conteudo, cor="green")

    def equacao_da_reta(self, a, b, c):
        """_summary_
            É aprendida na forma ax+by+c, mas para nos adequarmos às nomenclaturas de redes neurais,
            podemos reescrever essa equação como w1x1 + w2x2 + b. Ou seja, w1,w2 e b são parâmetros que definem um modelo linear,
            a reta
        Returns:
            str: _description_
        """

        # ax + by + c = 0
        # y = (-a*x)/b
        x = np.linspace(-2, 4, 50)
        y = (-a * x) / b
        plt.savefig("./assets/reta1.png")
        self.res = str(x) + "\n" + str(y) + "\n" + "gráfico salvo em assets/reta1.png"
        return f"Equação da reta|" + self.res
