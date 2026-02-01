
import pytest
from unittest.mock import patch, MagicMock
from PIL import Image
import sys
import os

# Add scripts directory to path to import visualize_data
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from visualize_data import visualize_data

def test_visualize_data_success():
    """Test visualize_data with valid inputs."""
    
    # Create 6 dummy PIL images
    images = [Image.new('RGB', (100, 100), color='red') for _ in range(6)]
    text = "Test visualization description"

    with patch('matplotlib.pyplot.show') as mock_show:
        visualize_data(text, images)
        mock_show.assert_called_once()
