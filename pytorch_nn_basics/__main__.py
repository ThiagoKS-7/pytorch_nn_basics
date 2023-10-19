import torch
from typing import List


def mostra_tipos(lista: List) -> None:
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
        "padrao": torch.Tensor(lista),
        "float": torch.FloatTensor(lista),
        "double": torch.DoubleTensor(lista),
        "long": torch.LongTensor(lista)
    }

    print("\n Padrões de tensores")
    for key in tipos_tensores:
        print(F"{key.capitalize()} ({tipos_tensores[key].dtype})\n{tipos_tensores[key]}\n")


def instancias_a_partir_do_np() -> None:
    import numpy as np
    
    instancias_numpy = {
        "random":   np.random.rand(3, 4),
        "torch_from_numpy": torch.from_numpy(np.random.rand(3, 4))
    }
    print("\nInstâncias vindas do Numpy")
    for key in instancias_numpy:
        print(f"{key.capitalize()} - ({instancias_numpy[key].dtype})\n {instancias_numpy[key]}\n")


def tensores_ja_inicializados() -> None:
    """_summary_
    Recebem como parâmetro o tamanho de cada dimensão do tensor
    """
    print("\n Tensores inicializados com valores padrão")
    tensores_iniciados = {
        "Tensores .ones": torch.ones(2, 3), # tensor preenchido de .1 do tamanho 2 de [3]
        "Tensores .zeros": torch.zeros(4, 3),  # tensor preenchido de 0 do tamanho 4 de [3]
        "Tensores .randn": torch.randn(3, 2) # tensor prenechido de valores aleatórios, tamanho 3 de [2]
    }
    for key in tensores_iniciados:
        print(f"{key} - {tensores_iniciados[key]}")

def tensor_p_numpy() -> None:
    print(f"\nTensores pra numpy")
    tensores_numpy = {
        "tipo_torch": torch.randn(3, 2),
        "tipo_np": torch.randn(3, 2).data.numpy()
    }
    for key in tensores_numpy:
        print(f"{key} - {type(tensores_numpy[key])}")


def indexao(lista:List) -> None:
    """_summary_
    Consegue reorganizar os tensores, de forma semelahnte a arrays Numpy
    Faz isso através de colchetes
    
    Args:
        lista (List): lista a ser convertida em tensor
    """
    print(f"\nExemplo indexao")
    tensor = torch.Tensor(lista)
    print(f"tensor não fatiado {tensor}")
    tensor[0, 2] = -10  # tensor 0D
    print(f"tensor com indice editado {tensor}")
    print(f"tensor fatiado {tensor[:, 2]}")
    print(f"tensor 0D {tensor[0,2]} - size {tensor[0,2].size()}")


def operacoes() -> None:
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
    print(f"\nExemplo operações")
    print(f"Shape tensores - {tnsr.T.shape}\n{tns1.shape}")
    print(f"soma tensores {soma}")
    print(f"multiplicação {mult}")
    print(f"divisao tensores {divisao}")


def size_e_view() -> None:
    """_summary_
    Uma operação importantíssima na manipulação	 de tensores para Deep Learning é a reorganização das suas dimensões.
    Dessa forma é possível linearizar um tensor n-dimensional
    """
    tnsr = torch.randn(3, 2)
    tns = torch.randn(2, 2, 3)
    print(f"\nExemplo .size()")
    print(f"Size - {tnsr.size()}")
    print(f"Size tensor 3D {tnsr.size()}")
    print(f"\nExemplo .view()")
    print(f"{tns.view(-1)}")
    print(f"{tns.view(tns.size(0), -1)}")


def cast_gpu() -> None:
    print(f"\nExemplo de cast GPU")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)
    tns = torch.randn(10)
    tns = tns.to(device)
    print(tns)


def exercicio() -> None:
    """_summary_
    No exemplo isso funciona pq eles partiram de um tensor de mesma dimensionalidade
    """
    print(f"\nResultado do exercício")
    tns = torch.randn(9, 12)
    tns1 = tns[0:5, 0:4]
    tns2 = tns[5:, 4:]

    resultado = torch.mm(tns1, tns2)
    print(resultado.size())


if __name__ == "__main__":
    lista = [[1, 2, 3], [4, 5, 6]]  # tensor 2 de [3]
    print(f"Is CUDA available? - {torch.cuda.is_available()}\n")
    mostra_tipos(lista)
    instancias_a_partir_do_np()
    tensores_ja_inicializados()
    tensor_p_numpy()
    indexao(lista)
    operacoes()
    size_e_view()
    cast_gpu()
    exercicio()
