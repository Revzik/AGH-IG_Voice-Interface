import unittest
from src.conf import config


class ConfiguratorTest(unittest.TestCase):
    def setUp(self):
        config.reload()

    def test_init(self):
        self.assertEqual(['preprocessing', 'parametrization'], config.analysis_config.sections())
        self.assertEqual([], config.synthesis_config.sections())

    def test_parameters_types(self):
        # analysis
        self.assertEqual(int, type(config.analysis["sampling_frequency"]))
        self.assertEqual(float, type(config.analysis["preemphasis"]))
        self.assertEqual(float, type(config.analysis["speech_threshold"]))
        self.assertEqual(int, type(config.analysis["window_length"]))
        self.assertEqual(int, type(config.analysis["window_overlap"]))
        self.assertEqual(int, type(config.analysis["filterbank_size"]))

    def test_default_parameters(self):
        self.assertEqual(8000, config.analysis["sampling_frequency"])
        self.assertEqual(0.95, config.analysis["preemphasis"])
        self.assertEqual(0.4, config.analysis["speech_threshold"])
        self.assertEqual(20, config.analysis["window_length"])
        self.assertEqual(10, config.analysis["window_overlap"])
        self.assertEqual(14, config.analysis["filterbank_size"])

    def test_parameters_fallbacks(self):
        # analysis
        config.analysis_config.clear()
        config.analysis = config.parse_analysis()

        self.assertEqual(8000, config.analysis["sampling_frequency"])
        self.assertEqual(0.95, config.analysis["preemphasis"])
        self.assertEqual(0.4, config.analysis["speech_threshold"])
        self.assertEqual(20, config.analysis["window_length"])
        self.assertEqual(10, config.analysis["window_overlap"])
        self.assertEqual(14, config.analysis["filterbank_size"])


if __name__ == '__main__':
    unittest.main()
