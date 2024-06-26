# tests/test_trittention.py

import unittest
import torch
from models.trittention import Trittention
from config import TrittentionConfig

class TestTrittention(unittest.TestCase):
    def setUp(self):
        self.config = TrittentionConfig(
            hidden_size=768,
            num_attention_heads=12,
            attention_probs_dropout_prob=0.1
        )
        self.trittention = Trittention(self.config)
    
    def test_forward(self):
        batch_size = 2
        seq_length = 10
        hidden_size = self.config.hidden_size
        
        hidden_states = torch.randn(batch_size, seq_length, hidden_size)
        attention_mask = torch.ones(batch_size, seq_length)
        
        output = self.trittention(hidden_states, attention_mask)
        
        self.assertEqual(output.shape, (batch_size, seq_length, hidden_size))
    
    def test_attention_mask(self):
        batch_size = 2
        seq_length = 10
        hidden_size = self.config.hidden_size
        
        hidden_states = torch.randn(batch_size, seq_length, hidden_size)
        attention_mask = torch.tensor([[1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]], dtype=torch.float)
        
        attention_mask = attention_mask[:, None, None, :]
        output = self.trittention(hidden_states, attention_mask)
        
        self.assertEqual(output.shape, (batch_size, seq_length * 2, hidden_size))
    
    def test_backward(self):
        batch_size = 2
        seq_length = 10
        hidden_size = self.config.hidden_size

        hidden_states = torch.randn(batch_size, seq_length, hidden_size, requires_grad=True)
        attention_mask = torch.ones(batch_size, seq_length)
        
        attention_mask = attention_mask[:, None, None, :]
        output = self.trittention(hidden_states, attention_mask)
        output.sum().backward()

        self.assertIsNotNone(hidden_states.grad)
        self.assertEqual(hidden_states.grad.shape, hidden_states.shape)

if __name__ == '__main__':
    unittest.main()
