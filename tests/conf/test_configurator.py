import unittest
from src.conf.configurator import Configurator


class ConfiguratorTest(unittest.TestCase):
    def setUp(self):
        self.config = Configurator()

    def test_init(self):
        self.assertEqual(self.config.analysis_config.sections(), ['preprocessing', 'parametrization'])
        self.assertEqual(self.config.synthesis_config.sections(), [])

    def test_parameters_types(self):
        # analysis
        self.assertEqual(type(self.config.analysis["sampling_frequency"]), int)
        self.assertEqual(type(self.config.analysis["preemphasis"]), float)
        self.assertEqual(type(self.config.analysis["speech_threshold"]), float)
        self.assertEqual(type(self.config.analysis["window_length"]), int)
        self.assertEqual(type(self.config.analysis["window_overlap"]), int)
        self.assertEqual(type(self.config.analysis["filterbank_size"]), int)

    def test_default_parameters(self):
        self.assertEqual(self.config.analysis["sampling_frequency"], 8000)
        self.assertEqual(self.config.analysis["preemphasis"], 0.95)
        self.assertEqual(self.config.analysis["speech_threshold"], 0.4)
        self.assertEqual(self.config.analysis["window_length"], 20)
        self.assertEqual(self.config.analysis["window_overlap"], 10)
        self.assertEqual(self.config.analysis["filterbank_size"], 14)

    def test_parameters_fallbacks(self):
        # analysis
        self.config.analysis_config.clear()
        self.config.analysis = self.config.parse_analysis()

        self.assertEqual(self.config.analysis["sampling_frequency"], 8000)
        self.assertEqual(self.config.analysis["preemphasis"], 0.95)
        self.assertEqual(self.config.analysis["speech_threshold"], 0.4)
        self.assertEqual(self.config.analysis["window_length"], 20)
        self.assertEqual(self.config.analysis["window_overlap"], 10)
        self.assertEqual(self.config.analysis["filterbank_size"], 14)


if __name__ == '__main__':
    unittest.main()
