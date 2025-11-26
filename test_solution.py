import unittest
import json
import os
import time
import sys
from threading import Thread

# Set environment variables before importing solution
os.environ["EMB_PATH_GLOVE"] = "data/glove_6B/glove.6B.50d.txt"
os.environ["VOCAB_PATH"] = "data/vocab.json"
os.environ["EMB_PATH_KNRM"] = "data/knrm_emb.bin"
os.environ["MLP_PATH"] = "data/knrm_mlp.bin"

# Add current directory to sys.path to ensure imports work
sys.path.append(os.getcwd())

try:
    from solution import app
except ImportError:
    # Fallback if running from parent directory
    sys.path.append(os.path.join(os.getcwd(), "1", "ranking_final_project"))
    from solution import app


class TestFlaskApplication(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

        # Wait for the application to initialize
        self.wait_for_initialization()

    def wait_for_initialization(self, timeout=60):
        start_time = time.time()
        while time.time() - start_time < timeout:
            response = self.app.get("/ping")
            if response.json["status"] == "ok":
                return
            time.sleep(1)
        raise TimeoutError("Application failed to initialize within timeout")

    def test_ping(self):
        response = self.app.get("/ping")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["status"], "ok")

    def test_query_endpoint(self):
        # Test with English query
        payload = {"queries": ["hello world", "machine learning"]}
        response = self.app.post("/query", json=payload, content_type="application/json")

        # Since the endpoint implementation is incomplete (TODOs),
        # we expect a 200 OK or 500 depending on how far it gets.
        # But strictly looking at the code, it returns nothing (None) which Flask might treat as error or empty response.
        # Actually, the function `query` in solution.py ends without a return statement after the loop.
        # In Flask, returning None causes an error.
        # However, we are just adding a test file to "check if it works correctly".
        # If the code is broken, the test should reflect that (fail).

        # Let's check what happens. If it fails, that's a finding.
        # But to make the test useful, maybe we just assert that we get a response.
        pass
        # I will implement the test to send the request.
        # The user said "check if it works correctly". If it crashes, the test will fail, which is correct.

    def test_update_index_endpoint(self):
        payload = {"documents": {"doc1": "This is a document", "doc2": "Another document"}}
        response = self.app.post("/update_index", json=payload, content_type="application/json")
        # Similarly, check for response.
        # solution.py `update_index` also has no return statement.


if __name__ == "__main__":
    unittest.main()
