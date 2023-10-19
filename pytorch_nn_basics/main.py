from intro_tensores import AulaTensores


if __name__ == '__main__':
    at = AulaTensores( [[1, 2, 3], [4, 5, 6]]) # tensor 2 de [3]
    at.mostra_conteudo_sobre([
        at.tipos_tensores(),
        at.instancias_a_partir_do_np(),
        at.tensores_ja_inicializados(),
        at.tensor_p_numpy(),
        at.indexao(),
        at.operacoes(),
        at.size_e_view(),
        at.cast_gpu(),
    ])