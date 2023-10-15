import torch
from typing import List


def mostra_tipos(lista: List) -> None:
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


def instancias_a_partir_do_np():
    import numpy as np

    arr = np.random.rand(3, 4)
    tns = torch.from_numpy(arr)
    print("\n Instâncias vindas do Numpy")
    print(f"{arr} - {arr.dtype}")
    print(f"{tns} - {tns.dtype}")


def tensores_ja_inicializados():
    print("\n Tensores já inicializados")
    tns1 = torch.ones(2, 3)  # tensor preenchido de .1 do tamanho 2 de [3]
    print(f" Tensores .ones - {tns1}")
    tns2 = torch.zeros(4, 3)  # tensor preenchido de 0 do tamanho 4 de [3]
    print(f" Tensores .zeros - {tns2}")
    tns3 = torch.randn(3, 2)
    print(
        f" Tensores .randn - {tns3}"
    )  # tensor prenechido de valores aleatórios, tamanho 3 de [2]


if __name__ == "__main__":
    lista = [[1, 2, 3], [4, 5, 6]]  # tensor 2 de [3]
    print(f"Is CUDA available? - {torch.cuda.is_available()}\n")
    mostra_tipos(lista)
    instancias_a_partir_do_np()
    tensores_ja_inicializados()
