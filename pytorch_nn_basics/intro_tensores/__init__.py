from utils import Aula
import torch
from typing import List
import numpy as np

class AulaTensores(Aula):
    def __init__(self,lista:List):
        self.lista = lista
        self.res = ""
    def mostra_conteudo_sobre(self, conteudo: str) -> None:
        return super().mostra_conteudo_sobre(conteudo) 
    
    def tipos_tensores(self) -> str:
        """_summary_
        É possivel criar tensores do PyTorch de inúmeras formas!
        Vamos ver o primeiro os tipos de tensores que estão ao nosso dispor.
        Para isso, vamos converter comns do Python em tensors do PyTorch

        Note que a impressão de tensores dos tipos float32 e int64 n vem acompanhadas de tipo
        dtype, visto que se tratam de tipos padrão do Pytorch

        Args:
            lista (List): lista a ser convertida em tensor
        """
        
        tipos_tensores = {
            "padrao": torch.Tensor(self.lista),
            "float": torch.FloatTensor(self.lista),
            "double": torch.DoubleTensor(self.lista),
            "long": torch.LongTensor(self.lista)
        }
        
        for key in tipos_tensores:
            self.res += F"{key.capitalize()} ({tipos_tensores[key].dtype})\n{tipos_tensores[key]}\n"
        return f"Padroes de tensores|" + self.res
    
    def instancias_a_partir_do_np(self) -> str:
        
        instancias_numpy = {
            "random":   np.random.rand(3, 4),
            "torch_from_numpy": torch.from_numpy(np.random.rand(3, 4))
        }

        for key in instancias_numpy:
            self.res = f"{key.capitalize()} - ({instancias_numpy[key].dtype})\n {instancias_numpy[key]}\n"
            
        return "Instâncias vindas do Numpy|" + self.res

    def tensores_ja_inicializados(self) -> str:
        """_summary_
        Recebem como parâmetro o tamanho de cada dimensão do tensor
        """
        tensores_iniciados = {
            "Tensores .ones": torch.ones(2, 3), # tensor preenchido de .1 do tamanho 2 de [3]
            "Tensores .zeros": torch.zeros(4, 3),  # tensor preenchido de 0 do tamanho 4 de [3]
            "Tensores .randn": torch.randn(3, 2) # tensor prenechido de valores aleatórios, tamanho 3 de [2]
        }
        for key in tensores_iniciados:
            self.res += f"{key} - {tensores_iniciados[key]}"
            
        return "Tensores inicializados com valores padrão|" + self.res

    def tensor_p_numpy(self) -> str:
        self.res = f"Tensores pra numpy|"
        tensores_numpy = {
            "tipo_torch": torch.randn(3, 2),
            "tipo_np": torch.randn(3, 2).data.numpy()
        }
        for key in tensores_numpy:
            self.res += f"{key} - {type(tensores_numpy[key])}"
            
        return self.res

    def indexao(self) -> str:
        """_summary_
        Consegue reorganizar os tensores, de forma semelahnte a arrays Numpy
        Faz isso através de colchetes
        
        Args:
            lista (List): lista a ser convertida em tensor
        """
        tensor = torch.Tensor(self.lista)
        tensor[0, 2] = -10  # tensor 0D
        
        self.res = (
            f"Exemplo indexao|" + 
            f"tensor não fatiado {tensor}" +
            f"tensor com indice editado {tensor}" +
            f"tensor fatiado {tensor[:, 2]}" +
            f"tensor 0D {tensor[0,2]} - size {tensor[0,2].size()}"
        )
        return self.res


    def operacoes(self) -> str:
        """_summary_
        A função .item() extrai o número de um tensor que possui um único valor,
        permitindo realizar as operações numéricas do Python.
        Caso o item não seja extraído,
        operaçãoes que envoplvam tensores vão retornar novos tensores.
        """
        tnsr = torch.randn(3, 2)
        tns1 = torch.ones(2, 2)
        tns = tnsr[:1, :]
        soma = tns + tns1
        divisao = tns / tns1
        mult = torch.mm(tns, tns1)
        
        self.res = (
            "Exemplo operações|" +
            f"Shape tensores - {tnsr.T.shape}\n{tns1.shape}" +
            f"soma tensores {soma}" +
            f"multiplicação {mult}" +
            f"divisao tensores {divisao}"
        )
        return self.res


    def size_e_view(self) -> str:
        """_summary_
        Uma operação importantíssima na manipulação	 de tensores para Deep Learning é a reorganização das suas dimensões.
        Dessa forma é possível linearizar um tensor n-dimensional
        """
        tnsr = torch.randn(3, 2)
        tns = torch.randn(2, 2, 3)
        self.res = (
            f"Exemplo .size()|" +
            f"Size - {tnsr.size()}" +
            f"Size tensor 3D {tnsr.size()}" + 
            f"\nExemplo .view()" +
            f"{tns.view(-1)}" +
            f"{tns.view(tns.size(0), -1)}"
        )
        return self.res

    def cast_gpu(self) -> str:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        tns = torch.randn(10)
        tns = tns.to(device)
        self.res = (
            "Exemplo de cast GPU|" +
            f"{device}" + 
            f"{tns}"
        )
        return self.res


    def exercicio(self) -> str:
        """_summary_
        No exemplo isso funciona pq eles partiram de um tensor de mesma dimensionalidade
        """
        tns = torch.randn(9, 12)
        tns1 = tns[0:5, 0:4]
        tns2 = tns[5:, 4:]
        resultado = torch.mm(tns1, tns2)
        self.res = (
            "Resultado do exercício|" +
            resultado.size()
        )