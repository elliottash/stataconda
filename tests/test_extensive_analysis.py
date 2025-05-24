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

class TestExtensiveAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication(sys.argv)
        set_test_environment(True)

    def setUp(self):
        # Create test data
        n = 100
        np.random.seed(42)
        self.df = pd.DataFrame({
            'invest': np.random.normal(100, 50, n),
            'value': np.random.normal(500, 200, n),
            'capital': np.random.normal(300, 100, n),
            'year': np.tile(np.arange(2000, 2000 + n // 5), 5),
            'firm': np.repeat([f'Firm{i+1}' for i in range(5)], n // 5)
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

    def test_extensive_analysis(self):
        # 2. Inspect data structure and overview
        result = self.gui.cmd_describe('describe')
        self.assertIn('Variable', result)
        result = self.gui.cmd_summarize('summarize')
        self.assertIn('Summary statistics', result)
        result = self.gui.cmd_browse('browse')
        self.assertIn('Data browser opened', result)

        # 3. Detailed summary for key numeric variables
        result = self.gui.cmd_summarize('summarize invest value capital, detail')
        self.assertIn('Summary statistics', result)

        # 4. Tabulations
        result = self.gui.cmd_tabulate('tabulate firm, sort')
        self.assertIn('Tabulation of firm', result)
        result = self.gui.cmd_tabulate('tabulate year, sort')
        self.assertIn('Tabulation of year', result)

        # 5. Histograms
        result = self.gui.cmd_histogram('histogram invest, title("Distribution of Investment")')
        self.assertIn('Generated histogram of invest', result)
        show_and_close_figures()
        result = self.gui.cmd_histogram('histogram value, percent title("Distribution of Firm Value")')
        self.assertIn('Generated histogram of value', result)
        show_and_close_figures()

        # 6. Line graph: Investment over time for each firm
        result = self.gui.cmd_lgraph('lgraph invest year firm, title("Investment Trajectories by Firm") xtitle("Year") ytitle("Investment")')
        self.assertIn('Line graph of invest vs year', result)
        show_and_close_figures()

        # 7. Scatter plot and binned scatter
        result = self.gui.cmd_scatter('scatter invest capital || lfitci invest capital, title("Investment vs. Capital with 95% CI")')
        self.assertIn('Scatter plot of invest vs capital', result)
        show_and_close_figures()
        result = self.gui.cmd_binscatter('binscatter invest capital, by(firm) residuals(value) title("Binned Scatter: Invest vs Capital, controlling for Value") xlabel(0(500)2000)')
        self.assertIn('Generated binscatter of invest vs capital', result)
        show_and_close_figures()

        # 9. Fixed-effects regression with reghdfe
        result = self.gui.cmd_reghdfe('reghdfe invest value capital, absorb(firm year) cluster(firm)')
        self.assertIn('Regression results', result)

        # 10. Instrumental variables with ivreghdfe
        # tsset firm year
        self.gui.cmd_tsset('tsset firm year')
        # gen L1_capital = L.capital
        self.gui.cmd_gen('gen L1_capital = L.capital')
        result = self.gui.cmd_ivreghdfe('ivreghdfe invest (capital = L1_capital) value, absorb(firm year) cluster(firm)')
        self.assertIn('IV regression results', result)

        # 11. Coefficient plot comparing OLS and IV estimates
        self.gui.cmd_estimates('estimates store OLS_model')
        self.gui.cmd_estimates('estimates store IV_model')
        result = self.gui.cmd_coefplot('coefplot OLS_model IV_model, drop(_cons) xline(0) title("Coefficient Comparison: OLS vs IV") legend(on)')
        self.assertIn('Coefficient plot', result)
        show_and_close_figures()

        # 12. Save regression tables to RTF
        result = self.gui.cmd_eesttab('eesttab OLS_model IV_model using analysis_results.rtf, replace se label')
        self.assertIn('Saved regression table', result)

    def test_scatter_plot(self):
        # ... existing code ...
        show_and_close_figures()

    def test_histogram(self):
        # ... existing code ...
        show_and_close_figures()

    def test_graph_bar(self):
        # ... existing code ...
        show_and_close_figures()

    def test_binscatter(self):
        # ... existing code ...
        show_and_close_figures()

    def test_coefplot(self):
        # ... existing code ...
        show_and_close_figures()

    def test_lgraph(self):
        # ... existing code ...
        show_and_close_figures()

    def test_hist(self):
        # ... existing code ...
        show_and_close_figures()

if __name__ == '__main__':
    unittest.main() 