import unittest
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import QApplication
import sys
import matplotlib.pyplot as plt
from main import StatacondaGUI, set_test_environment

def show_and_close_figures():
    plt.show(block=False)
    plt.pause(0.5)
    plt.close('all')

class TestRegressionCommands(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication(sys.argv)
        set_test_environment(True)

    def setUp(self):
        n = 50
        np.random.seed(0)
        self.df = pd.DataFrame({
            'y': np.random.normal(0, 1, n),
            'x1': np.random.normal(0, 1, n),
            'x2': np.random.normal(0, 1, n),
            'firm': np.random.choice(['A', 'B', 'C'], n),
            'year': np.random.choice([2000, 2001, 2002], n)
        })
        self.gui = StatacondaGUI()
        self.gui._df = self.df.copy()

    def tearDown(self):
        self.gui.close()
        plt.close('all')

    @classmethod
    def tearDownClass(cls):
        cls.app.quit()
        plt.close('all')
        set_test_environment(False)

    def test_reg(self):
        result = self.gui.translate_reghdfe('reg y x1 x2')
        self.assertIn('Linear regression', result)
        # coefplot should work
        plot_result = self.gui.cmd_coefplot('coefplot')
        self.assertIn('Coefficient plot', plot_result)
        show_and_close_figures()
        # estout should work
        estout_result = self.gui.cmd_estout('estout')
        self.assertIn('Variable', estout_result)

    def test_reghdfe_absorb(self):
        result = self.gui.translate_reghdfe('reghdfe y x1 x2, absorb(firm)')
        self.assertIn('Linear regression', result)
        plot_result = self.gui.cmd_coefplot('coefplot')
        self.assertIn('Coefficient plot', plot_result)
        show_and_close_figures()
        estout_result = self.gui.cmd_estout('estout')
        self.assertIn('Variable', estout_result)

    def test_regress(self):
        result = self.gui.cmd_regress('regress y x1 x2')
        self.assertIn('Linear regression', result)
        plot_result = self.gui.cmd_coefplot('coefplot')
        self.assertIn('Coefficient plot', plot_result)
        show_and_close_figures()
        estout_result = self.gui.cmd_estout('estout')
        self.assertIn('Variable', estout_result)

    def test_ivregress(self):
        # Generate an instrument
        self.gui._df['z'] = self.gui._df['x1'] + np.random.normal(0, 0.1, len(self.gui._df))
        result = self.gui.cmd_ivregress('ivregress 2sls y (x1 = z) x2')
        self.assertIn('Instrumental variables', result)
        plot_result = self.gui.cmd_coefplot('coefplot')
        self.assertIn('Coefficient plot', plot_result)
        show_and_close_figures()
        estout_result = self.gui.cmd_estout('estout')
        self.assertIn('Variable', estout_result)

    def test_ivreghdfe(self):
        # Generate an instrument
        self.gui._df['z'] = self.gui._df['x1'] + np.random.normal(0, 0.1, len(self.gui._df))
        result = self.gui.translate_ivreghdfe('ivreghdfe y (x1 = z) x2, absorb(firm)')
        self.assertIn('Instrumental variables', result)
        plot_result = self.gui.cmd_coefplot('coefplot')
        self.assertIn('Coefficient plot', plot_result)
        show_and_close_figures()
        estout_result = self.gui.cmd_estout('estout')
        self.assertIn('Variable', estout_result)

if __name__ == '__main__':
    unittest.main() 