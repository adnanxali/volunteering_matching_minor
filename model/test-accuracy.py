import unittest
from unittest.mock import patch, MagicMock, mock_open
import json
import io
import os
import sys
from flask import Flask
import numpy as np

# Mock Recommender to simulate graph generation and evaluation results
class MockRecommender:
    def __init__(self):
        self.evaluation_results = {}
    
    def generate_accuracy_graphs(self):
        # Simulate an image buffer
        buf = io.BytesIO()
        buf.write(b'mock_image_data')
        buf.seek(0)
        return buf

class TestAccuracyGraphEndpoint(unittest.TestCase):
    def setUp(self):
        self.app = Flask(__name__)
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()
        self.mock_recommender = MockRecommender()

        self.sample_eval_results = {
            "regression_metrics": {
                "train": {"rmse": 0.25, "mae": 0.2, "r2": 0.75},
                "test": {"rmse": 0.3, "mae": 0.25, "r2": 0.7}
            },
            "classification_metrics": {
                "train": {"accuracy": 0.85, "precision": 0.8, "recall": 0.9, "f1": 0.85, "auc": 0.9},
                "test": {"accuracy": 0.8, "precision": 0.75, "recall": 0.85, "f1": 0.8, "auc": 0.85}
            },
            "roc_data": {
                "fpr": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                "tpr": [0.0, 0.3, 0.5, 0.65, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98, 1.0]
            },
            "feature_importance": [
                {"feature": "skill_similarity", "importance": 0.4},
                {"feature": "location_proximity", "importance": 0.3},
                {"feature": "skill_overlap_count", "importance": 0.3}
            ],
            "threshold": 0.5,
            "timestamp": "2025-04-28T12:00:00"
        }

    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open, read_data=json.dumps({}))
    @patch('recommender.send_file')  # Changed from flask.send_file to recommender.send_file
    def test_get_accuracy_graph_no_results(self, mock_send_file, mock_file, mock_exists):
        mock_exists.return_value = False

        from ml_recommender import app
        self.client = app.test_client()
        
        # Make sure the recommender object is available
        import ml_recommender
        ml_recommender.recommender.evaluation_results = {}
        
        response = self.client.get('/api/accuracy/graph')
        self.assertEqual(response.status_code, 404)
        response_json = json.loads(response.data)
        self.assertFalse(response_json['success'])
        self.assertIn('No evaluation results available', response_json['error'])

    @patch('os.path.exists')
    @patch('flask.send_file')
    def test_get_accuracy_graph_existing_file(self, mock_send_file, mock_exists):
        mock_exists.side_effect = lambda path: path == './model/accuracy_graphs.png'
        mock_send_file.return_value = "Mocked file response"

        from ml_recommender import app
        self.client = app.test_client()
        
        response = self.client.get('/api/accuracy/graph')
        self.assertEqual(response.status_code, 200)
        
        # Verify send_file was called with correct args
        mock_send_file.assert_called_once_with('./model/accuracy_graphs.png', mimetype='image/png')

    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('ml_recommender.VolunteerProjectRecommender.generate_accuracy_graphs')
    @patch('flask.send_file')
    def test_get_accuracy_graph_generate_new(self, mock_send_file, mock_generate_graphs, mock_file, mock_exists):
        # Set up mocks
        mock_exists.side_effect = lambda path: path == './model/evaluation_results.json'
        mock_file.return_value.__enter__.return_value.read.return_value = json.dumps(self.sample_eval_results)
        
        # Create a mock for the generated image buffer
        mock_buffer = io.BytesIO()
        mock_buffer.write(b'test_image_data')
        mock_buffer.seek(0)
        mock_generate_graphs.return_value = mock_buffer
        
        # Set expected return from send_file
        mock_send_file.return_value = "Mocked send_file response"

        from ml_recommender import app, recommender
        self.client = app.test_client()
        
        # Set the evaluation results directly
        recommender.evaluation_results = self.sample_eval_results
        
        response = self.client.get('/api/accuracy/graph')
        self.assertEqual(response.status_code, 200)
        
        # Verify graph generation function was called
        mock_generate_graphs.assert_called_once()
        
        # Verify send_file was called with correct buffer
        mock_send_file.assert_called_once()
        file_arg = mock_send_file.call_args[0][0]
        self.assertTrue(isinstance(file_arg, io.BytesIO))
        self.assertEqual(file_arg.getvalue(), b'test_image_data')

    @patch('os.path.exists')
    def test_get_accuracy_graph_error_handling(self, mock_exists):
        mock_exists.side_effect = Exception("Test exception")

        from ml_recommender import app
        self.client = app.test_client()
        
        response = self.client.get('/api/accuracy/graph')
        self.assertEqual(response.status_code, 500)
        response_json = json.loads(response.data)
        self.assertFalse(response_json['success'])
        self.assertIn('Test exception', response_json['error'])

    # New test to verify the metrics endpoint
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open)
    def test_get_accuracy_metrics(self, mock_file, mock_exists):
        mock_exists.return_value = True
        mock_file.return_value.__enter__.return_value.read.return_value = json.dumps(self.sample_eval_results)
        
        from ml_recommender import app, recommender
        self.client = app.test_client()
        
        # Set evaluation results directly
        recommender.evaluation_results = self.sample_eval_results
        
        # Test classification metrics
        response = self.client.get('/api/accuracy/metrics?type=classification')
        self.assertEqual(response.status_code, 200)
        response_json = json.loads(response.data)
        self.assertTrue(response_json['success'])
        self.assertIn('metrics', response_json)
        self.assertIn('train', response_json['metrics'])
        self.assertIn('test', response_json['metrics'])
        self.assertEqual(response_json['metrics']['test']['accuracy'], 0.8)
        
        # Test regression metrics
        response = self.client.get('/api/accuracy/metrics?type=regression')
        self.assertEqual(response.status_code, 200)
        response_json = json.loads(response.data)
        self.assertTrue(response_json['success'])
        self.assertEqual(response_json['metrics']['test']['rmse'], 0.3)
        
        # Test feature importance
        response = self.client.get('/api/accuracy/metrics?type=feature_importance')
        self.assertEqual(response.status_code, 200)
        response_json = json.loads(response.data)
        self.assertTrue(response_json['success'])
        self.assertEqual(len(response_json['feature_importance']), 3)
        self.assertEqual(response_json['feature_importance'][0]['feature'], 'skill_similarity')

    # New test to verify custom graph generation
    @patch('ml_recommender.plt')
    @patch('ml_recommender.io.BytesIO')
    @patch('flask.send_file')
    def test_generate_custom_graph(self, mock_send_file, mock_bytesio, mock_plt):
        # Set up mock for bytesio
        mock_buffer = MagicMock()
        mock_bytesio.return_value = mock_buffer
        
        # Set expected return from send_file
        mock_send_file.return_value = "Mocked custom graph response"
        
        from ml_recommender import app, recommender
        self.client = app.test_client()
        
        # Set evaluation results directly
        recommender.evaluation_results = self.sample_eval_results
        
        # Test bar chart generation
        request_data = {
            "metrics": ["accuracy", "precision", "recall"],
            "type": "bar",
            "title": "Test Performance Metrics"
        }
        
        response = self.client.post('/api/custom_graph', 
                             data=json.dumps(request_data),
                             content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        
        # Verify matplotlib was used
        mock_plt.figure.assert_called_once()
        mock_plt.savefig.assert_called_once()
        
        # Verify send_file was called with the buffer
        mock_send_file.assert_called_once_with(mock_buffer, mimetype='image/png')

if __name__ == '__main__':
    unittest.main()