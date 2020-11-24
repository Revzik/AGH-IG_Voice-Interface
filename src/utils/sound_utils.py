def normalize(sound_list):
    for i in range(len(sound_list)):
        sound_list[i].normalize()

    return sound_list
