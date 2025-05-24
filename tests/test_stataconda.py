import sys
import unittest
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt, QEvent
from PyQt5.QtTest import QTest
from main import StatacondaGUI, set_test_environment
import pandas as pd
import numpy as np
import statsmodels.api as sm
import os

class TestStataconda(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create QApplication instance
        cls.app = QApplication(sys.argv)
        set_test_environment(True)
        
    def setUp(self):
        # Create a new instance of StatacondaGUI for each test
        self.window = StatacondaGUI()
        
        # Initialize test data
        self.window._df = sm.datasets.grunfeld.load_pandas().data
        self.window.update_variables_pane()
        
    def test_page_up_down_history_navigation(self):
        """Test Page Up/Down keys for command history navigation in the command prompt"""
        # Add some commands to history
        test_commands = ["describe", "summarize", "regress invest value"]
        for cmd in test_commands:
            self.window.command_history.append(cmd)
            self.window.history_index = -1
        
        # Test Page Up navigation (should go to most recent command)
        QTest.keyClick(self.window.command_prompt, Qt.Key_PageUp)
        self.assertEqual(self.window.command_prompt.toPlainText(), "regress invest value")
        
        QTest.keyClick(self.window.command_prompt, Qt.Key_PageUp)
        self.assertEqual(self.window.command_prompt.toPlainText(), "summarize")
        
        QTest.keyClick(self.window.command_prompt, Qt.Key_PageUp)
        self.assertEqual(self.window.command_prompt.toPlainText(), "describe")
        
        # Test Page Down navigation (should go forward in history)
        QTest.keyClick(self.window.command_prompt, Qt.Key_PageDown)
        self.assertEqual(self.window.command_prompt.toPlainText(), "summarize")
        
        QTest.keyClick(self.window.command_prompt, Qt.Key_PageDown)
        self.assertEqual(self.window.command_prompt.toPlainText(), "regress invest value")
        
        QTest.keyClick(self.window.command_prompt, Qt.Key_PageDown)
        self.assertEqual(self.window.command_prompt.toPlainText(), "")

    def test_command_history_navigation(self):
        """Test command history navigation using Up/Down arrows"""
        # Add some commands to history
        test_commands = ["describe", "summarize", "regress invest value"]
        for cmd in test_commands:
            self.window.command_history.append(cmd)
            self.window.history_index = -1
        
        # Test Up arrow navigation
        QTest.keyClick(self.window.command_prompt, Qt.Key_Up)
        self.assertEqual(self.window.command_prompt.toPlainText(), "regress invest value")
        
        QTest.keyClick(self.window.command_prompt, Qt.Key_Up)
        self.assertEqual(self.window.command_prompt.toPlainText(), "summarize")
        
        # Test Down arrow navigation
        QTest.keyClick(self.window.command_prompt, Qt.Key_Down)
        self.assertEqual(self.window.command_prompt.toPlainText(), "regress invest value")
        
        QTest.keyClick(self.window.command_prompt, Qt.Key_Down)
        self.assertEqual(self.window.command_prompt.toPlainText(), "")
        
    # def test_line_graph_functionality(self):
    #     """Test line graph command functionality"""
    #     # Test basic line graph
    #     self.window.command_prompt.setPlainText("lgraph invest year")
    #     self.window.execute_command()
    #     self.assertIn("Line graph of invest vs year", self.window.results_window.toPlainText())
    #     
    #     # Test line graph with group variable
    #     self.window.command_prompt.setPlainText("lgraph invest year firm")
    #     self.window.execute_command()
    #     self.assertIn("Line graph of invest vs year", self.window.results_window.toPlainText())
    #     
    #     # Test line graph with title
    #     self.window.command_prompt.setPlainText('lgraph invest year, title("Investment Over Time")')
    #     self.window.execute_command()
    #     self.assertIn("Line graph of invest vs year", self.window.results_window.toPlainText())
        
    def test_results_pane_formatting(self):
        """Test results pane formatting and highlighting"""
        # Test command formatting
        self.window.command_prompt.setPlainText("describe")
        self.window.execute_command()
        text = self.window.results_window.toPlainText()
        self.assertTrue(text.startswith("describe"))
        
        # Test error formatting
        self.window.command_prompt.setPlainText("invalid_command")
        self.window.execute_command()
        text = self.window.results_window.toPlainText()
        self.assertIn("Error:", text)
        
    def test_command_execution(self):
        """Test various command executions"""
        # Test Stata commands
        self.window.command_prompt.setPlainText("describe")
        self.window.execute_command()
        self.assertIn("Columns:", self.window.results_window.toPlainText())
        
        self.window.command_prompt.setPlainText("summarize")
        self.window.execute_command()
        self.assertIn("count", self.window.results_window.toPlainText())
        
        # Test Python commands
        self.window.command_prompt.setPlainText(">>> print('Hello, World!')")
        self.window.execute_command()
        self.assertIn("Hello, World!", self.window.results_window.toPlainText())
        
        # Test bash commands
        self.window.command_prompt.setPlainText("!pwd")
        self.window.execute_command()
        self.assertIn("/", self.window.results_window.toPlainText())
        
    def test_variable_operations(self):
        """Test variable operations and updates"""
        # Test variable generation
        self.window.command_prompt.setPlainText("gen newvar = invest * 2")
        self.window.execute_command()
        self.assertIn("newvar", self.window._df.columns)
        
        # Test variable dropping
        self.window.command_prompt.setPlainText("drop newvar")
        self.window.execute_command()
        self.assertNotIn("newvar", self.window._df.columns)
        
    def test_dataframe_operations(self):
        """Test DataFrame operations and updates"""
        # Test DataFrame loading
        self.window.command_prompt.setPlainText("use grunfeld.dta")
        self.window.execute_command()
        self.assertIsNotNone(self.window._df)
        
        # Test DataFrame saving
        self.window.command_prompt.setPlainText("save test_output.dta")
        self.window.execute_command()
        self.assertTrue(os.path.exists("test_output.dta"))
        
    def test_egen_functions(self):
        """Test egen functions"""
        # Test mean calculation
        self.window.command_prompt.setPlainText("egen mean_invest = mean(invest)")
        self.window.execute_command()
        self.assertIn("mean_invest", self.window._df.columns)
        self.assertAlmostEqual(self.window._df['mean_invest'].iloc[0], self.window._df['invest'].mean())
        
        # Test grouped mean
        self.window.command_prompt.setPlainText("egen mean_invest_by_firm = mean(invest), by(firm)")
        self.window.execute_command()
        self.assertIn("mean_invest_by_firm", self.window._df.columns)
        # Verify grouped means
        for firm in self.window._df['firm'].unique():
            firm_mean = self.window._df[self.window._df['firm'] == firm]['invest'].mean()
            test_mean = self.window._df[self.window._df['firm'] == firm]['mean_invest_by_firm'].iloc[0]
            self.assertAlmostEqual(firm_mean, test_mean)
        
        # Test sum calculation
        self.window.command_prompt.setPlainText("egen total_invest = sum(invest)")
        self.window.execute_command()
        self.assertIn("total_invest", self.window._df.columns)
        self.assertAlmostEqual(self.window._df['total_invest'].iloc[0], self.window._df['invest'].sum())
        
        # Test grouped sum
        self.window.command_prompt.setPlainText("egen total_invest_by_firm = sum(invest), by(firm)")
        self.window.execute_command()
        self.assertIn("total_invest_by_firm", self.window._df.columns)
        # Verify grouped sums
        for firm in self.window._df['firm'].unique():
            firm_sum = self.window._df[self.window._df['firm'] == firm]['invest'].sum()
            test_sum = self.window._df[self.window._df['firm'] == firm]['total_invest_by_firm'].iloc[0]
            self.assertAlmostEqual(firm_sum, test_sum)
        
        # Test min calculation
        self.window.command_prompt.setPlainText("egen min_invest = min(invest)")
        self.window.execute_command()
        self.assertIn("min_invest", self.window._df.columns)
        self.assertAlmostEqual(self.window._df['min_invest'].iloc[0], self.window._df['invest'].min())
        
        # Test max calculation
        self.window.command_prompt.setPlainText("egen max_invest = max(invest)")
        self.window.execute_command()
        self.assertIn("max_invest", self.window._df.columns)
        self.assertAlmostEqual(self.window._df['max_invest'].iloc[0], self.window._df['invest'].max())
        
        # Test sd (standard deviation) calculation
        self.window.command_prompt.setPlainText("egen sd_invest = sd(invest)")
        self.window.execute_command()
        self.assertIn("sd_invest", self.window._df.columns)
        self.assertAlmostEqual(self.window._df['sd_invest'].iloc[0], self.window._df['invest'].std())
        
        # Test grouped sd
        self.window.command_prompt.setPlainText("egen sd_invest_by_firm = sd(invest), by(firm)")
        self.window.execute_command()
        self.assertIn("sd_invest_by_firm", self.window._df.columns)
        # Verify grouped standard deviations
        for firm in self.window._df['firm'].unique():
            firm_std = self.window._df[self.window._df['firm'] == firm]['invest'].std()
            test_std = self.window._df[self.window._df['firm'] == firm]['sd_invest_by_firm'].iloc[0]
            self.assertAlmostEqual(firm_std, test_std)
        
    def test_by_prefix(self):
        """Test by prefix functionality"""
        # Test by with egen
        self.window.command_prompt.setPlainText("by firm: egen mean_invest = mean(invest)")
        self.window.execute_command()
        self.assertIn("mean_invest", self.window._df.columns)
        
    def tearDown(self):
        # Clean up after each test
        self.window.close()
        
    @classmethod
    def tearDownClass(cls):
        # Clean up QApplication
        cls.app.quit()
        set_test_environment(False)

if __name__ == '__main__':
    unittest.main() 