
import unittest
import json
from app import app, mpc_config, system_config

class AppTestCase(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_index(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)

    def test_get_config(self):
        response = self.app.get('/api/config')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('mpc', data)
        self.assertIn('system', data)

    def test_update_config(self):
        new_config = {
            'mpc': {
                'dt': 0.2,
                'N': 15,
                'Q': [1.0, 1.0],
                'R': [0.1],
                'x_min': [-10.0, -10.0],
                'x_max': [10.0, 10.0],
                'u_min': [-1.0, -1.0],
                'u_max': [1.0, 1.0],
                'terminal_region_radius': 0.1,
                'simulation_time': 10.0,
                'initial_state': [0.0, 0.0],
                'reference_state': [1.0, 1.0]
            },
            'system': {
                'system_type': 'linear',
                'api_config': {},
                'sim_config': {
                    'A': [[1, 0.1], [0, 1]],
                    'B': [[0.05], [0.1]],
                    'noise_level': 0.0,
                    'disturbance_level': 0.0
                }
            }
        }
        response = self.app.post('/api/config',
                                 data=json.dumps(new_config),
                                 content_type='application/json')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(mpc_config.dt, 0.2)
        self.assertEqual(system_config.system_type, 'linear')

    def test_run_simulation(self):
        response = self.app.post('/api/run')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('t', data)
        self.assertIn('x', data)
        self.assertIn('u', data)
        self.assertIn('mode', data)

    def test_get_results(self):
        self.app.post('/api/run')
        response = self.app.get('/api/results')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('t', data)
        self.assertIn('x', data)
        self.assertIn('u', data)
        self.assertIn('mode', data)

if __name__ == '__main__':
    unittest.main()
