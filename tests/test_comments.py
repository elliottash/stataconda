import unittest
from main import StatacondaGUI
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import QApplication
import sys

def run_command(gui, command):
    """Helper function to run a command and return its output"""
    gui.command_prompt.setPlainText(command)
    gui.execute_command()
    return gui.results_window.toPlainText()

class TestComments(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up QApplication for all tests"""
        cls.app = QApplication(sys.argv)

    def setUp(self):
        """Set up test environment before each test"""
        self.gui = StatacondaGUI()
        # Create a simple test dataset
        self.test_df = pd.DataFrame({
            'x': [1, 2, 3, 4, 5],
            'y': [2, 4, 6, 8, 10]
        })
        self.gui._df = self.test_df
        self.gui._python_context['_df'] = self.test_df

    def tearDown(self):
        """Clean up after each test"""
        self.gui.close()

    @classmethod
    def tearDownClass(cls):
        """Clean up QApplication"""
        cls.app.quit()

    def test_full_line_comments(self):
        """Test that full-line comments are ignored"""
        # Test * comments
        output = run_command(self.gui, '* This is a comment')
        self.assertEqual(output.strip(), '')
        
        # Test // comments
        output = run_command(self.gui, '// This is also a comment')
        self.assertEqual(output.strip(), '')

    def test_end_of_line_comments(self):
        """Test that end-of-line comments are properly handled"""
        # Test with summarize command
        output = run_command(self.gui, 'summarize x // Get summary stats')
        self.assertIn('x', output)
        self.assertNotIn('Get summary stats', output)
        
        # Test with generate command
        output = run_command(self.gui, 'generate z = x + y // Create sum variable')
        self.assertIn('z', self.gui._df.columns)
        self.assertEqual(list(self.gui._df['z']), [3, 6, 9, 12, 15])

    def test_comment_preservation(self):
        """Test that comments are preserved in command history"""
        # Run commands with comments
        run_command(self.gui, '* First comment')
        run_command(self.gui, 'summarize x // Second comment')
        run_command(self.gui, '// Third comment')
        
        # Check history
        history_items = [self.gui.history_list.item(i).text() 
                        for i in range(self.gui.history_list.count())]
        self.assertIn('* First comment', history_items)
        self.assertIn('summarize x // Second comment', history_items)
        self.assertIn('// Third comment', history_items)

    def test_do_file_comments(self):
        """Test comment handling in do-files"""
        # Create a test do-file with comments
        with open('test_comments.do', 'w') as f:
            f.write('* This is a do-file comment\n')
            f.write('use test.dta  // Load data\n')
            f.write('// Another comment\n')
            f.write('summarize x  * Get stats\n')
        
        # Run the do-file
        output = run_command(self.gui, 'do test_comments.do')
        
        # Clean up
        import os
        os.remove('test_comments.do')
        
        # Check that comments were handled properly
        self.assertNotIn('This is a do-file comment', output)
        self.assertNotIn('Load data', output)
        self.assertNotIn('Another comment', output)
        self.assertNotIn('Get stats', output)

    def test_nested_comments(self):
        """Test handling of nested or malformed comments"""
        # Test multiple // in one line
        output = run_command(self.gui, 'summarize x // comment1 // comment2')
        self.assertIn('x', output)
        self.assertNotIn('comment1', output)
        self.assertNotIn('comment2', output)
        
        # Test * in the middle of a line
        output = run_command(self.gui, 'summarize x * not a comment')
        self.assertIn('x', output)
        self.assertIn('not a comment', output)  # This should be treated as part of the command

    def test_comment_with_quotes(self):
        """Test comment handling with quoted strings"""
        # Test with quoted strings
        output = run_command(self.gui, 'label variable x "My variable" // Add label')
        self.assertIn('My variable', output)
        self.assertNotIn('Add label', output)
        
        # Test with quotes in comments
        output = run_command(self.gui, 'summarize x // This is a "quoted" comment')
        self.assertIn('x', output)
        self.assertNotIn('quoted', output)

    def test_comment_with_special_chars(self):
        """Test comment handling with special characters"""
        # Test with various special characters
        special_chars = ['#', '@', '$', '%', '&', '(', ')', '[', ']', '{', '}']
        for char in special_chars:
            output = run_command(self.gui, f'summarize x // {char}comment')
            self.assertIn('x', output)
            self.assertNotIn(f'{char}comment', output)

if __name__ == '__main__':
    unittest.main() 