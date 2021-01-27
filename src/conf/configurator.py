import configparser as cp
import os


class Configurator:
    def __init__(self):
        directory = os.path.dirname(__file__)

        analysis_path = os.path.abspath(os.path.join(directory, '..', '..', 'conf', 'analysis.ini'))
        self.analysis_config = cp.ConfigParser()
        self.analysis_config.read(analysis_path)
        self.analysis = self.parse_analysis()

        folders_path = os.path.abspath(os.path.join(directory, '..', '..', 'conf', 'folders.ini'))
        self.folders_config = cp.ConfigParser()
        self.folders_config.read(folders_path)
        self.folders = self.parse_folders()

        synthesis_path = os.path.abspath(os.path.join(directory, '..', '..', 'conf', 'synthesis.ini'))
        self.synthesis_config = cp.ConfigParser()
        self.synthesis_config.read(synthesis_path)
        self.synthesis = self.parse_synthesis()

    def parse_analysis(self):
        analysis = {
            'sampling_frequency': self.analysis_config.getint('preprocessing', 'sampling_frequency', fallback=8000),
            'preemphasis': self.analysis_config.getfloat('preprocessing', 'preemphasis', fallback=0.95),
            'vad_threshold': self.analysis_config.getfloat('vad', 'vad_threshold', fallback=0.3),
            'window_length': self.analysis_config.getint('mfcc', 'window_length', fallback=20),
            'window_overlap': self.analysis_config.getint('mfcc', 'window_overlap', fallback=10),
            'bottom_filterbank_frequency': self.analysis_config.getfloat('mfcc', 'bottom_filterbank_frequency', fallback=300),
            'top_filterbank_frequency': self.analysis_config.getfloat('mfcc', 'top_filterbank_frequency', fallback=4000),
            'filterbank_size': self.analysis_config.getint('mfcc', 'filterbank_size', fallback=14),
            'n_clusters': self.analysis_config.getint('gmm', 'n_clusters', fallback=14),
            'iterations': self.analysis_config.getint('gmm', 'iterations', fallback=20),
            'number_of_groups': self.analysis_config.getint('k_folds', 'number_of_groups', fallback=5),
            'fmin': self.analysis_config.getint('synthesis', 'fmin', fallback=40),
            'fmax': self.analysis_config.getint('synthesis', 'fmax', fallback=400)
        }
        return analysis

    def parse_folders(self):
        folders = {
            'known': {},
            'unknown': {}
        }
        for key in self.folders_config['known']:
            folders['known'][key] = self.folders_config.get('known', key)
        for key in self.folders_config['unknown']:
            folders['unknown'][key] = self.folders_config.get('unknown', key)
        return folders

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
