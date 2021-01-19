from src.utils import sound_loader
import random
import math
import numpy as np
from src.conf import config

def configuration_k_fold():

    # numbers of splits/groups
    n = config.analysis['number_of_group']

    # load words
    ciszej = sound_loader.load_sound_file("E:\STUDIA\SEMESTR 9\Interfejs głosowy\Baza\ZNANI\ciszej")
    do_przodu = sound_loader.load_sound_file("E:\STUDIA\SEMESTR 9\Interfejs głosowy\Baza\ZNANI\do_przodu")
    do_tylu = sound_loader.load_sound_file("E:\STUDIA\SEMESTR 9\Interfejs głosowy\Baza\ZNANI\do_tyłu")
    glosniej = sound_loader.load_sound_file("E:\STUDIA\SEMESTR 9\Interfejs głosowy\Baza\ZNANI\głośniej")
    igla = sound_loader.load_sound_file("E:\STUDIA\SEMESTR 9\Interfejs głosowy\Baza\ZNANI\igła")
    losuj = sound_loader.load_sound_file("E:\STUDIA\SEMESTR 9\Interfejs głosowy\Baza\ZNANI\losuj")
    odstaw = sound_loader.load_sound_file("E:\STUDIA\SEMESTR 9\Interfejs głosowy\Baza\ZNANI\odstaw")
    postaw = sound_loader.load_sound_file("E:\STUDIA\SEMESTR 9\Interfejs głosowy\Baza\ZNANI\postaw")
    przewin = sound_loader.load_sound_file("E:\STUDIA\SEMESTR 9\Interfejs głosowy\Baza\ZNANI\przewiń")
    start = sound_loader.load_sound_file("E:\STUDIA\SEMESTR 9\Interfejs głosowy\Baza\ZNANI\start")
    stop = sound_loader.load_sound_file("E:\STUDIA\SEMESTR 9\Interfejs głosowy\Baza\ZNANI\stop")
    wybierz = sound_loader.load_sound_file("E:\STUDIA\SEMESTR 9\Interfejs głosowy\Baza\ZNANI\wybierz")

    # change order in words list
    random.shuffle(ciszej)
    random.shuffle(do_przodu)
    random.shuffle(do_tylu)
    random.shuffle(glosniej)
    random.shuffle(igla)
    random.shuffle(losuj)
    random.shuffle(odstaw)
    random.shuffle(postaw)
    random.shuffle(przewin)
    random.shuffle(start)
    random.shuffle(stop)
    random.shuffle(wybierz)

    # numbers of fold, the same len of each words
    x = len(ciszej)
    l = n
    # group size initialization
    size = np.zeros(l, dtype=np.int16 )

    # each gropu size
    for i in range(l):
        size[i] = math.ceil(x/l)
        x = x - size[i]
        l = l - 1

    k_fold_group = []
    a = 0

    for i in range(n):
        d_1 = {}
        d_1['ciszej'] = ciszej[a:a+size[i]]
        d_1['do_przodu'] = do_przodu[a:a+size[i]]
        d_1['do_tylu'] = do_tylu[a:a+size[i]]
        d_1['glosniej'] = glosniej[a:a+size[i]]
        d_1['igla'] = igla[a:a+size[i]]
        d_1['losuj'] = losuj[a:a+size[i]]
        d_1['odstaw'] = odstaw[a:a+size[i]]
        d_1['postaw'] = postaw[a:a+size[i]]
        d_1['przewin'] = przewin[a:a+size[i]]
        d_1['start'] = start[a:a+size[i]]
        d_1['stop'] = stop[a:a+size[i]]
        d_1['wybierz'] = wybierz[a:a+size[i]]
        a = a + size[i]
        k_fold_group.append(d_1)

    return k_fold_group
