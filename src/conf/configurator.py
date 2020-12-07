import configparser as cp
import os


class Configurator:
    def __init__(self):
        directory = os.path.dirname(__file__)

        analysis_path = os.path.abspath(os.path.join(directory, '..', '..', 'conf', 'analysis.ini'))
        self.analysis_config = cp.ConfigParser()
        self.analysis_config.read(analysis_path)
        self.analysis = self.parse_analysis()

        synthesis_path = os.path.abspath(os.path.join(directory, '..', '..', 'conf', 'synthesis.ini'))
        self.synthesis_config = cp.ConfigParser()
        self.synthesis_config.read(synthesis_path)
        self.synthesis = self.parse_synthesis()

    def parse_analysis(self):
        analysis = {
            'sampling_frequency': self.analysis_config.getint('preprocessing', 'sampling_frequency', fallback=8000),
            'preemphasis': self.analysis_config.getfloat('preprocessing', 'preemphasis', fallback=0.95),
            'vad_threshold': self.analysis_config.getfloat('vad', 'vad_threshold', fallback=40),
            'window_length': self.analysis_config.getint('parametrization', 'window_length', fallback=20),
            'window_overlap': self.analysis_config.getint('parametrization', 'window_overlap', fallback=10),
            'bottom_filterbank_frequency': self.analysis_config.getfloat('parametrization', 'bottom_filterbank_frequency', fallback=0),
            'top_filterbank_frequency': self.analysis_config.getfloat('parametrization', 'top_filterbank_frequency', fallback=4000),
            'filterbank_size': self.analysis_config.getint('parametrization', 'filterbank_size', fallback=14)
        }
        return analysis

    def parse_synthesis(self):
        return {}

    def reload(self):
        directory = os.path.dirname(__file__)

        analysis_path = os.path.abspath(os.path.join(directory, '..', '..', 'conf', 'analysis.ini'))
        self.analysis_config = cp.ConfigParser()
        self.analysis_config.read(analysis_path)
        self.analysis = self.parse_analysis()

        synthesis_path = os.path.abspath(os.path.join(directory, '..', '..', 'conf', 'synthesis.ini'))
        self.synthesis_config = cp.ConfigParser()
        self.synthesis_config.read(synthesis_path)
        self.synthesis = self.parse_synthesis()
