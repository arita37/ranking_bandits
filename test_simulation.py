
import unittest
from simulation import generate_click_data


class TestGenerateClickData(unittest.TestCase):

    def test_generate_click_data(self):
        # Generate a sample cfg dictionary and T value
        cfg = {
            'loc_probas': {
                'loc1': 0.5,
                'loc2': 0.3,
                'loc3': 0.2,
            },
            'item_probas': {
                'item1': {'loc1': 0.6, 'loc2': 0.4, 'loc3': 0.2},
                'item2': {'loc1': 0.4, 'loc2': 0.5, 'loc3': 0.3},
            }
        }
        T = 1000

        # Call the generate_click_data function
        df = generate_click_data(cfg, T)
        # Check if the DataFrame has the correct columns
        expected_columns = ['ts', 'loc_id', 'item_id', 'is_clk']
        self.assertListEqual(list(df.columns), expected_columns)

        # Check if the DataFrame has the expected number of rows (equal to T)
        self.assertEqual(len(df), T)

        # Check if is_clk values are within the expected range (0 or 1)
        self.assertTrue(all(df['is_clk'].isin([0, 1])))


if __name__ == '__main__':
    unittest.main()
