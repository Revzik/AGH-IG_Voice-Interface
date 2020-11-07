from src.param.mfcc import Mfcc
"""
This is a console version of menu to the interface
"""


def print_message():
    print("Interfejs głosowy 2020/2021")
    print("")
    print("Wybierz opcję:")
    print("1 - Trening algorytmów")
    print("2 - Rozpoznanie mowy")
    print("3 - Synteza mowy")
    print("q - Wyjście")


def check_option(option):
    if option != "1" and option != "2" and option != "3" and option != "q":
        print("Zła wartość!")
        return False
    return True


def start():
    running = True
    while running:
        print_message()
        option = input(":")
        if not check_option(option):
            continue
        if option == "1":
            print("Tu będzie opcja treningu")
        elif option == "2":
            print("Tu będzie opcja rozpoznania")
        elif option == "3":
            print("Tu będzie opcja synteza")
        elif option == "q":
            print("Zamykanie programu")
            running = False
