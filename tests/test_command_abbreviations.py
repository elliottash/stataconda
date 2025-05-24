import unittest
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication
from PyQt5.QtTest import QTest
from PyQt5.QtCore import Qt
from main import StatacondaGUI, StoredEstimates, set_test_environment

class TestCommandAbbreviations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication(sys.argv)
        set_test_environment(True)

    def setUp(self):
        # Create test data
        n = 100
        np.random.seed(42)
        self.df = pd.DataFrame({
            'y': np.random.normal(0, 1, n),
            'x1': np.random.normal(0, 1, n),
            'x2': np.random.normal(0, 1, n),
            'z': np.random.normal(0, 1, n),  # Instrument
            'firm': np.random.choice(['A', 'B', 'C'], n),
            'year': np.random.choice([2000, 2001, 2002], n)
        })
        self.gui = StatacondaGUI()
        self.gui._df = self.df.copy()
        self.gui._stored_estimates = StoredEstimates()

    def tearDown(self):
        self.gui.close()
        plt.close('all')

    @classmethod
    def tearDownClass(cls):
        cls.app.quit()
        plt.close('all')
        set_test_environment(False)

    def test_reg_abbreviations(self):
        """Test reg/regress/reghdfe abbreviations"""
        # Test basic regression with 'reg'
        result = self.gui.translate_reghdfe('reg y x1 x2')
        self.assertIn('Linear regression', result)
        self.assertIn('Number of obs', result)
        self.assertIn('R-squared', result)

        # Test with 'regress'
        result = self.gui.translate_reghdfe('regress y x1 x2')
        self.assertIn('Linear regression', result)
        self.assertIn('Number of obs', result)
        self.assertIn('R-squared', result)

        # Test with 'reghdfe' and absorb
        result = self.gui.translate_reghdfe('reghdfe y x1 x2, absorb(firm)')
        self.assertIn('Linear regression', result)
        self.assertIn('Absorbed fixed effects', result)

        # Test with cluster option
        result = self.gui.translate_reghdfe('reg y x1 x2, cluster(firm)')
        self.assertIn('Linear regression', result)
        self.assertIn('Clustered on', result)

    def test_ivreg_abbreviations(self):
        """Test ivreg/ivregress/ivreghdfe abbreviations"""
        # Test basic IV regression with 'ivreg'
        result = self.gui.translate_ivreghdfe('ivreg y (x1 = z) x2')
        self.assertIn('Instrumental variables', result)
        self.assertIn('First-stage regression', result)

        # Test with 'ivregress'
        result = self.gui.translate_ivreghdfe('ivregress y (x1 = z) x2')
        self.assertIn('Instrumental variables', result)
        self.assertIn('First-stage regression', result)

        # Test with 'ivreghdfe' and absorb
        result = self.gui.translate_ivreghdfe('ivreghdfe y (x1 = z) x2, absorb(firm)')
        self.assertIn('Instrumental variables', result)
        self.assertIn('First-stage regression', result)

        # Test with cluster option
        result = self.gui.translate_ivreghdfe('ivreg y (x1 = z) x2, cluster(firm)')
        self.assertIn('Instrumental variables', result)
        self.assertIn('Clustered on', result)

    def test_post_estimation_commands(self):
        """Test post-estimation commands with abbreviated regression commands"""
        # Run regression with 'reg'
        self.gui.translate_reghdfe('reg y x1 x2')
        
        # Test coefplot
        plot_result = self.gui.cmd_coefplot('coefplot')
        self.assertIn('Coefficient plot', plot_result)
        
        # Test estout
        estout_result = self.gui.cmd_estout('estout')
        self.assertIn('Variable', estout_result)

        # Run IV regression with 'ivreg'
        self.gui.translate_ivreghdfe('ivreg y (x1 = z) x2')
        
        # Test coefplot with IV results
        plot_result = self.gui.cmd_coefplot('coefplot')
        self.assertIn('Coefficient plot', plot_result)
        
        # Test estout with IV results
        estout_result = self.gui.cmd_estout('estout')
        self.assertIn('Variable', estout_result)

    def test_error_handling(self):
        """Test error handling for invalid commands"""
        # Test invalid regression syntax
        result = self.gui.translate_reghdfe('reg')
        self.assertIn('Usage:', result)

        # Test invalid IV regression syntax
        result = self.gui.translate_ivreghdfe('ivreg')
        self.assertIn('Invalid ivreghdfe command format', result)

        # Test invalid options
        result = self.gui.translate_reghdfe('reg y x1 x2, invalid_option')
        self.assertIn('Usage:', result)

        # Test invalid IV options
        result = self.gui.translate_ivreghdfe('ivreg y (x1 = z) x2, invalid_option')
        self.assertIn('Invalid ivreghdfe command format', result)

    def test_estout_after_regression(self):
        """Test that estout works after running a regression with an abbreviation"""
        # Run regression with 'reg' abbreviation
        self.gui.command_registry['reg']('reg y x1 x2')
        # Call estout for the lastreg estimate
        estout_result = self.gui.cmd_estout('estout lastreg')
        self.assertIn('Estimate lastreg:', estout_result)
        self.assertIn('coef', estout_result.lower())

if __name__ == '__main__':
    unittest.main() 