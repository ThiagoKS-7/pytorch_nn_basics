from utils import Aula
import torch
from typing import Union, List
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_classification
             

class AulaPerceptron(Aula):
    def __init__(self) -> None:
        super().__init__()
        self.res = ""
        np.random.seed(46)

        self.mostra_conteudo_sobre([
            self.equacao_da_reta(-2, 3, 0.4),
            self.distrbuicao_aleatoria(),
            self.plotmodel(5,1,0.4)
            ])

    def mostra_conteudo_sobre(self, conteudo: Union[str, List[str]]) -> None:
        return super().mostra_conteudo_sobre(conteudo, cor="green")

    def equacao_da_reta(self, a, b, c):
        """
        É aprendida na forma ax+by+c, mas para nos adequarmos às nomenclaturas de redes neurais,
        podemos reescrever essa equação como w1x1 + w2x2 + b. Ou seja, w1,w2 e b são parâmetros que definem um modelo linear,
        a reta
        """

        # ax + by + c = 0
        # y = (-ax-c)/b
        x = np.linspace(-2, 4, 50)
        y = (-a*x -c) / b
        plt.plot(x, y)
        plt.grid(True)
        plt.savefig("./assets/reta1.png")
        self.res = str(x) + "\n" + str(y) + "\n" + "gráfico salvo em assets/reta1.png"
        return f"Equacao da reta|" + self.res

    def distrbuicao_aleatoria(self):        
        X,y = make_classification(n_features=2, n_redundant=0, n_informative=1, n_clusters_per_class=1)
        
        plt.scatter(X[:,0], X[:,1], marker='o', c=y, edgecolor='k')
        plt.savefig("./assets/dist_aleatoria.png")
        plt.close()
        
        self.res = str(X) + "\n" + str(y) + "\n" + "gráfico salvo em assets/dist_aleatoria.png"
        return f"Distribuicao aleatoria|" + self.res
    
    def plotmodel(self, w1, w2, b):
       
        x = np.linspace(-2, 4, 50)
        y = (-w1*x -b) / w2
        X,Y = make_classification(n_features=2, n_redundant=0, n_informative=1, n_clusters_per_class=1)
        
        plt.scatter(X[:,0], X[:,1], marker='o', c=Y, edgecolor='k')
        plt.axvline(0,-1,1,color='k', linewidth=1)
        plt.axhline(0,-2,4,color='k', linewidth=1)
        plt.plot(x,y)
        plt.grid(True)
        plt.savefig("./assets/modelo_reta.png")
        self.res = str(x) + "\n" + str(y) + "\n" + "gráfico salvo em assets/modelo_reta.png"
        return f"Modelo da reta|" + self.res