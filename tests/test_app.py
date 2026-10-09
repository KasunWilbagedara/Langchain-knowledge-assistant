import importlib
import os
import sys
import unittest
from unittest.mock import MagicMock, patch


class CreateKnowledgeChainTests(unittest.TestCase):
    def reload_chains_module(self, env):
        for key in ('GEMINI_API_KEY', 'GOOGLE_API_KEY'):
            os.environ.pop(key, None)
        for key, value in env.items():
            os.environ[key] = value
        sys.modules.pop('app.chains', None)
        return importlib.import_module('app.chains')

    def test_create_knowledge_chain_accepts_google_api_key(self):
        module = self.reload_chains_module({'GOOGLE_API_KEY': 'demo-google-key'})

        with patch.object(module, 'init_chat_model', return_value=MagicMock()) as mock_init:
            chain = module.create_knowledge_chain()

        self.assertIsNotNone(chain)
        self.assertEqual(mock_init.call_count, 1)
        self.assertEqual(mock_init.call_args.kwargs['api_key'], 'demo-google-key')
        self.assertEqual(mock_init.call_args.kwargs['temperature'], 0)

    def test_create_knowledge_chain_accepts_gemini_api_key(self):
        module = self.reload_chains_module({'GEMINI_API_KEY': 'demo-gemini-key'})

        with patch.object(module, 'init_chat_model', return_value=MagicMock()) as mock_init:
            chain = module.create_knowledge_chain()

        self.assertIsNotNone(chain)
        self.assertEqual(mock_init.call_count, 1)
        self.assertEqual(mock_init.call_args.kwargs['api_key'], 'demo-gemini-key')

    def test_create_knowledge_chain_rejects_placeholder_keys(self):
        module = self.reload_chains_module({'GOOGLE_API_KEY': 'your_google_api_key_here'})

        with self.assertRaisesRegex(ValueError, 'Set GOOGLE_API_KEY or GEMINI_API_KEY'):
            module.create_knowledge_chain()


if __name__ == '__main__':
    unittest.main()
