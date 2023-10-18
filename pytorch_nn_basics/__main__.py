import torch
from typing import List


def mostra_tipos(lista: List) -> None:
    """
    É possivel criar tensores do PyTorch de inúmeras formas!
    Vamos ver o primeiro os tipos de tensores que estão ao nosso dispor.
    Para isso, vamos converter comns do Python em tensors do PyTorch

    Note que a impressão de tensores dos tipos float32 e int64 n vem acompanhadas de tipo
    dtype, visto que se tratam de tipos padrão do Pytorch
    """
    tns = torch.Tensor(lista)
    tns2 = torch.FloatTensor(lista)
    tns3 = torch.DoubleTensor(lista)
    tns4 = torch.LongTensor(lista)

    print("\n Padrões de tensores")
    print(f"Padrão (Float 32) - {tns.dtype}")
    print(tns)
    print(f"Float 32- {tns2.dtype}")
    print(tns2)
    print(f"Double - {tns3.dtype}")
    print(tns)
    print(f"Int (Long) - {tns4.dtype}")
    print(tns4)


def instancias_a_partir_do_np() -> None:
    import numpy as np

    arr = np.random.rand(3, 4)
    tns = torch.from_numpy(arr)
    print("\n Instâncias vindas do Numpy")
    print(f"{arr} - {arr.dtype}")
    print(f"{tns} - {tns.dtype}")


def tensores_ja_inicializados() -> None:
    print("\n Tensores já inicializados")
    tns1 = torch.ones(2, 3)  # tensor preenchido de .1 do tamanho 2 de [3]
    print(f" Tensores .ones - {tns1}")
    tns2 = torch.zeros(4, 3)  # tensor preenchido de 0 do tamanho 4 de [3]
    print(f" Tensores .zeros - {tns2}")
    tns3 = torch.randn(3, 2)
    print(
        f" Tensores .randn - {tns3}"
    )  # tensor prenechido de valores aleatórios, tamanho 3 de [2]


def tensor_p_numpy() -> None:
    tnsr = torch.randn(3, 2)
    print(type(tnsr))
    arr = tnsr.data.numpy()
    print(type(arr))


def indexao(tensor) -> None:
    print(f"\nExemplo indexao")
    tensor = torch.Tensor(tensor)
    print(f"tensor não fatiado {tensor}")
    tensor[0, 2] = -10  # tensor 0D
    print(f"tensor com indice editado {tensor}")
    print(f"tensor fatiado {tensor[:, 2]}")
    print(f"tensor 0D {tensor[0,2]} - size {tensor[0,2].size()}")


def operacoes() -> None:
    """
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
    """
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
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(device)
    tns = torch.randn(10)
    tns = tns.to(device)
    print(tns)


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
