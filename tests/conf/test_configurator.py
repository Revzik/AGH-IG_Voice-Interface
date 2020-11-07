import unittest
from src.conf import config


class ConfiguratorTest(unittest.TestCase):
    def setUp(self):
        config.reload()

    def test_init(self):
        self.assertEqual(config.analysis_config.sections(), ['preprocessing', 'parametrization'])
        self.assertEqual(config.synthesis_config.sections(), [])

    def test_parameters_types(self):
        # analysis
        self.assertEqual(type(config.analysis["sampling_frequency"]), int)
        self.assertEqual(type(config.analysis["preemphasis"]), float)
        self.assertEqual(type(config.analysis["speech_threshold"]), float)
        self.assertEqual(type(config.analysis["window_length"]), int)
        self.assertEqual(type(config.analysis["window_overlap"]), int)
        self.assertEqual(type(config.analysis["filterbank_size"]), int)

    def test_default_parameters(self):
        self.assertEqual(config.analysis["sampling_frequency"], 8000)
        self.assertEqual(config.analysis["preemphasis"], 0.95)
        self.assertEqual(config.analysis["speech_threshold"], 0.4)
        self.assertEqual(config.analysis["window_length"], 20)
        self.assertEqual(config.analysis["window_overlap"], 10)
        self.assertEqual(config.analysis["filterbank_size"], 14)

    def test_parameters_fallbacks(self):
        # analysis
        config.analysis_config.clear()
        config.analysis = config.parse_analysis()

        self.assertEqual(config.analysis["sampling_frequency"], 8000)
        self.assertEqual(config.analysis["preemphasis"], 0.95)
        self.assertEqual(config.analysis["speech_threshold"], 0.4)
        self.assertEqual(config.analysis["window_length"], 20)
        self.assertEqual(config.analysis["window_overlap"], 10)
        self.assertEqual(config.analysis["filterbank_size"], 14)


if __name__ == '__main__':
    unittest.main()
