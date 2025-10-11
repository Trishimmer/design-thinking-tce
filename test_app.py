import pytest
from app import app
import time
import os
import random


@pytest.fixture
def client():
    """Fixture for Flask test client with application context."""
    app.config['TESTING'] = True
    with app.app_context():
        with app.test_client() as client:
            yield client


def test_valid_ph(client):
    """Test with a valid pH value."""
    response = client.post('/predict_crop', data={
        'nitrogen': 50,
        'phosphorus': 30,
        'potassium': 40,
        'temperature': 25.0,
        'humidity': 70.0,
        'ph': 7.0,
        'rainfall': 199.0
    }, content_type='application/x-www-form-urlencoded')
    assert response.status_code == 200
    assert b'crop' in response.data


def test_ph_out_of_upper_bound(client):
    """Test with pH greater than 10."""
    response = client.post('/predict_crop', data={
        'nitrogen': 50,
        'phosphorus': 20,
        'potassium': 30,
        'temperature': 25.0,
        'humidity': 70.0,
        'ph': 11.0,
        'rainfall': 199.0
    }, content_type='application/x-www-form-urlencoded')
    assert response.status_code == 400
    assert b'Invalid pH value' in response.data


def test_ph_out_of_lower_bound(client):
    """Test with pH less than 1."""
    response = client.post('/predict_crop', data={
        'nitrogen': 50,
        'phosphorus': 20,
        'potassium': 30,
        'temperature': 25.0,
        'humidity': 70.0,
        'ph': -2.0,
        'rainfall': 199.0
    }, content_type='application/x-www-form-urlencoded')
    assert response.status_code == 400
    assert b'Invalid pH value' in response.data


def test_invalid_ph_type(client):
    """Test with non-numeric pH value."""
    response = client.post('/predict_crop', data={
        'nitrogen': 50,
        'phosphorus': 30,
        'potassium': 40,
        'temperature': 25.0,
        'humidity': 70.0,
        'ph': 'acidic',
        'rainfall': 199.0
    }, content_type='application/x-www-form-urlencoded')
    assert response.status_code == 400
    assert b'Invalid input type' in response.data


def test_multiple_validation_errors(client):
    """Test with multiple invalid values."""
    response = client.post('/predict_crop', data={
        'nitrogen': -10,  # Invalid
        'phosphorus': 30,
        'potassium': 40,
        'temperature': 40.0,  # Invalid
        'humidity': 70.0,
        'ph': 15.0,  # Invalid
        'rainfall': -5.0  # Invalid
    }, content_type='application/x-www-form-urlencoded')
    assert response.status_code == 400
    assert b'Invalid nitrogen value' in response.data
    assert b'Invalid temperature value' in response.data
    assert b'Invalid pH value' in response.data
    assert b'Invalid rainfall value' in response.data


# Add a custom pytest hook to display "All test cases passed"
def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Hook to display custom message after all tests pass."""
    if exitstatus == 0:  # Exit status 0 means all tests passed
        terminalreporter.write_sep("=", "All test cases passed")


def poppers_effect():
    """Simulate poppers flying with more dynamic animation."""
    colors = ['\033[91m', '\033[92m', '\033[93m', '\033[94m', '\033[95m', '\033[96m']  # Red, Green, Yellow, Blue, Magenta, Cyan
    reset = '\033[0m'

    # Define the terminal height and width
    width = 40
    height = 15

    # Generate random frames
    for _ in range(20):  # Number of frames
        os.system('cls' if os.name == 'nt' else 'clear')  # Clear the screen

        # Generate confetti
        for _ in range(random.randint(5, 15)):  # Random number of confetti pieces
            x = random.randint(0, width)
            y = random.randint(0, height)
            confetti = random.choice(['*', '.', 'o', '+', 'x', '🎉'])
            color = random.choice(colors)
            print(f"{color}{confetti}{reset}".rjust(x + len(confetti)))

        # Print "Poppers!" at the bottom
        print("\n" * height)  # Move the message down
        print("🎉✨ Poppers! ✨🎉".center(width + 20))  # Center message
        time.sleep(0.2)  # Delay between frames

    # Final celebratory message
    os.system('cls' if os.name == 'nt' else 'clear')
    print("\n🎉 All test cases passed! 🎉 Poppers flying everywhere! 🎉\n")
    print("✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨\n")


if __name__ == '__main__':
    # Run pytest programmatically
    exit_code = pytest.main()

    # Check if all tests passed
    if exit_code == 0:
        poppers_effect()
