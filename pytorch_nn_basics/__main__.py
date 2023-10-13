import torch
from typing import List


def mostra_tipos(lista: List) -> None:
    tns = torch.Tensor(lista)
    tns2 = torch.FloatTensor(lista)
    tns3 = torch.DoubleTensor(lista)
    tns4 = torch.LongTensor(lista)

    print("\n Instâncias vindas do Numpy")
    print(f"\nPadrão (Float 32) - {tns.dtype}")
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


if __name__ == "__main__":
    lista = [[1, 2, 3], [4, 5, 6]]
    print(f"Is CUDA available? - {torch.cuda.is_available()}\n")
    mostra_tipos(lista)
    instancias_a_partir_do_np()
