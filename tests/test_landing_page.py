import pytest
from app import app
from bs4 import BeautifulSoup

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_landing_page_educational_content(client):
    """
    Test that the landing page contains educational content about MPC.
    This test is expected to fail until the content is added.
    """
    rv = client.get('/')
    assert rv.status_code == 200
    soup = BeautifulSoup(rv.data, 'html.parser')
    page_text = soup.get_text()

    # Keywords we expect to find after implementation
    expected_keywords = [
        "Model Predictive Control",
        "MPC",
        "optimal control",
        "prediction horizon",
        "control horizon"
    ]

    for keyword in expected_keywords:
        assert keyword in page_text, f"Expected keyword '{keyword}' not found in landing page"

def test_landing_page_external_links(client):
    """
    Test that the landing page contains external links to learn more about MPC.
    This test is expected to fail until the links are added.
    """
    rv = client.get('/')
    assert rv.status_code == 200
    soup = BeautifulSoup(rv.data, 'html.parser')

    # Expected external links (example URLs that don't exist yet)
    expected_links = [
        "https://en.wikipedia.org/wiki/Model_predictive_control",
        "https://www.youtube.com/watch?v=some_mpc_tutorial",
        "https://www.example.com/mpc-resource"
    ]
    
    found_links = [a['href'] for a in soup.find_all('a', href=True)]

    for link in expected_links:
        assert link in found_links, f"Expected external link '{link}' not found in landing page"
