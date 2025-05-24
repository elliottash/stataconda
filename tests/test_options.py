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

class TestOptionParsing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initialize Qt application once for all tests."""
        cls.app = QApplication(sys.argv)
        set_test_environment(True)

    def setUp(self):
        """Set up test data and GUI instance."""
        # Create test data
        self.df = pd.DataFrame({
            'invest': np.random.normal(100, 50, 100),
            'value': np.random.normal(500, 200, 100),
            'year': np.arange(1935, 2035),
            'firm': ['Firm' + str(i % 5) for i in range(100)]
        })
        
        # Create GUI instance
        self.gui = StatacondaGUI()
        self.gui._df = self.df

    def tearDown(self):
        """Clean up after each test."""
        self.gui.close()
        plt.close('all')  # Close all matplotlib figures

    @classmethod
    def tearDownClass(cls):
        """Clean up Qt application after all tests."""
        cls.app.quit()
        plt.close('all')  # Ensure all figures are closed
        set_test_environment(False)

    def test_split_command_options(self):
        """Test splitting commands into main part and options part."""
        # Test basic splitting
        main, opts = self.gui._split_command_options('command var1 var2, opt1 opt2')
        self.assertEqual(main, 'command var1 var2')
        self.assertEqual(opts, 'opt1 opt2')

        # Test with no options
        main, opts = self.gui._split_command_options('command var1 var2')
        self.assertEqual(main, 'command var1 var2')
        self.assertEqual(opts, '')

        # Test with quoted strings
        main, opts = self.gui._split_command_options('command var1 var2, title("My Title")')
        self.assertEqual(main, 'command var1 var2')
        self.assertEqual(opts, 'title("My Title")')

        # Test with nested parentheses
        main, opts = self.gui._split_command_options('command var1 var2, title("My (Title)")')
        self.assertEqual(main, 'command var1 var2')
        self.assertEqual(opts, 'title("My (Title)")')

    def test_parse_options(self):
        """Test parsing options into a dictionary."""
        # Test flag options
        opts = self.gui._parse_options('percent')
        self.assertEqual(opts, {'percent': True})

        # Test value options
        opts = self.gui._parse_options('title("My Title")')
        self.assertEqual(opts, {'title': '"My Title"'})

        # Test multiple options
        opts = self.gui._parse_options('percent title("My Title") xtitle("X")')
        self.assertEqual(opts, {
            'percent': True,
            'title': '"My Title"',
            'xtitle': '"X"'
        })

        # Test options with numbers
        opts = self.gui._parse_options('bins(20)')
        self.assertEqual(opts, {'bins': '20'})

        # Test case insensitivity
        opts = self.gui._parse_options('TITLE("My Title")')
        self.assertEqual(opts, {'title': '"My Title"'})

    def test_parse_plot_titles(self):
        """Test parsing plot titles from options."""
        # Test with all titles
        options = {
            'title': 'Main Title',
            'xtitle': 'X Axis',
            'ytitle': 'Y Axis'
        }
        xtitle, ytitle, title = self.gui._parse_plot_titles(
            options, 'default_x', 'default_y', 'default_title'
        )
        self.assertEqual(xtitle, 'X Axis')
        self.assertEqual(ytitle, 'Y Axis')
        self.assertEqual(title, 'Main Title')

        # Test with missing titles
        options = {'title': 'Main Title'}
        xtitle, ytitle, title = self.gui._parse_plot_titles(
            options, 'default_x', 'default_y', 'default_title'
        )
        self.assertEqual(xtitle, 'default_x')
        self.assertEqual(ytitle, 'default_y')
        self.assertEqual(title, 'Main Title')

    def test_histogram_command(self):
        """Test histogram command with various options."""
        # Test basic histogram
        result = self.gui.cmd_histogram('histogram invest')
        self.assertIn('Generated histogram of invest', result)
        show_and_close_figures()

        # Test with percent option
        result = self.gui.cmd_histogram('histogram invest, percent')
        self.assertIn('Generated histogram of invest', result)
        show_and_close_figures()

        # Test with titles
        result = self.gui.cmd_histogram('histogram invest, title("My Title") xtitle("X") ytitle("Y")')
        self.assertIn('Generated histogram of invest', result)
        show_and_close_figures()

        # Test with comma after variable
        result = self.gui.cmd_histogram('histogram invest, percent')
        self.assertIn('Generated histogram of invest', result)
        show_and_close_figures()

    def test_scatter_command(self):
        """Test scatter command with various options."""
        # Test basic scatter
        result = self.gui.cmd_scatter('scatter invest value')
        self.assertIn('Scatter plot of invest vs value', result)
        show_and_close_figures()

        # Test with titles
        result = self.gui.cmd_scatter('scatter invest value, title("My Title") xtitle("X") ytitle("Y")')
        self.assertIn('Scatter plot of invest vs value', result)
        show_and_close_figures()

        # Test with regression line
        result = self.gui.cmd_scatter('scatter invest value || lfitci invest value')
        self.assertIn('Scatter plot of invest vs value', result)
        show_and_close_figures()

    def test_binscatter_command(self):
        """Test binscatter command with various options."""
        # Test basic binscatter
        result = self.gui.cmd_binscatter('binscatter invest value')
        self.assertIn('Generated binscatter of invest vs value', result)
        show_and_close_figures()

        # Test with bins option
        result = self.gui.cmd_binscatter('binscatter invest value, bins(10)')
        self.assertIn('Generated binscatter of invest vs value', result)
        show_and_close_figures()

        # Test with titles
        result = self.gui.cmd_binscatter('binscatter invest value, title("My Title") xtitle("X") ytitle("Y")')
        self.assertIn('Generated binscatter of invest vs value', result)
        show_and_close_figures()

    def test_lgraph_command(self):
        """Test lgraph command with various options."""
        # Test basic lgraph
        result = self.gui.cmd_lgraph('lgraph invest year')
        self.assertIn('Line graph of invest vs year', result)
        show_and_close_figures()

        # Test with group variable
        result = self.gui.cmd_lgraph('lgraph invest year firm')
        self.assertIn('Line graph of invest vs year', result)
        show_and_close_figures()

        # Test with titles
        result = self.gui.cmd_lgraph('lgraph invest year, title("My Title") xtitle("X") ytitle("Y")')
        self.assertIn('Line graph of invest vs year', result)
        show_and_close_figures()

        # Test with group variable and titles
        result = self.gui.cmd_lgraph('lgraph invest year firm, title("My Title") xtitle("X") ytitle("Y")')
        self.assertIn('Line graph of invest vs year', result)
        show_and_close_figures()

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