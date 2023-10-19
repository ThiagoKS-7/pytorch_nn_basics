from abc import ABCMeta, abstractmethod
from termcolor import colored
import pyfiglet
from typing import Union, List


class Aula(object):
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def mostra_conteudo_sobre(self, conteudo:Union[str, List[str]], fonte:str="doom", cor:str="cyan") -> None:
        if isinstance(conteudo,str):
            titulo = pyfiglet.figlet_format(conteudo.split("|")[0], font = fonte ) 
            print(colored(f"\n{titulo}", cor))
            print(conteudo.split("|")[1])
        else:
            for item in conteudo:
                titulo = pyfiglet.figlet_format(item.split("|")[0], font = fonte ) 
                print(colored(f"\n{titulo}", cor))
                print(item.split("|")[1])