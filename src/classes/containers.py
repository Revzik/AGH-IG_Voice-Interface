class SoundWave:
    def __init__(self, samples, fs=44100, phrase=''):
        self.samples = samples
        self.fs = fs
        self.phrase = phrase

    def __getitem__(self, i):
        return self.samples[i]

    def __setitem__(self, i, value):
        self.samples[i] = value

    def length(self):
        return self.samples.size
