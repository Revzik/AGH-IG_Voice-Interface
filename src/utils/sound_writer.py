import soundfile as sf

def save_sound_file(sound_signal, fs, file_path):
    sf.write(file_path, sound_signal, fs)
