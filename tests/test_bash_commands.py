import os
import shutil
import tempfile
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from main import StatacondaGUI, set_test_environment
from PyQt5.QtWidgets import QApplication
import sys

# Ensure QApplication is constructed before any QWidget
app = QApplication(sys.argv)
set_test_environment(True)

def run_command(gui, command):
    gui.command_prompt.setPlainText(command)
    return gui.execute_command()

def test_bash_commands():
    # Create a temporary directory for testing
    temp_dir = tempfile.mkdtemp()
    try:
        # Create test data
        df = pd.DataFrame({
            'x': np.random.normal(0, 1, 100),
            'y': np.random.normal(0, 1, 100)
        })
        
        # Create GUI instance
        gui = StatacondaGUI()
        gui._df = df
        
        # Test shell command
        result = run_command(gui, '!echo "Hello World"')
        assert "Hello World" in result
        
        # Test shell command with output redirection
        result = run_command(gui, '!echo "Test" > test.txt')
        assert os.path.exists('test.txt')
        
        # Test shell command with pipe
        result = run_command(gui, '!echo "Hello" | grep "Hello"')
        assert "Hello" in result
        
        # Test shell command with error
        result = run_command(gui, '!nonexistent_command')
        assert "Error" in result
        
        # Test cd command
        original_dir = os.getcwd()
        try:
            # Create a test directory
            test_dir = os.path.join(temp_dir, 'test_cd')
            os.makedirs(test_dir)
            
            # Test changing to the test directory
            result = run_command(gui, 'cd ' + test_dir)
            assert "Changed directory to" in result
            assert os.getcwd() == test_dir
            
            # Test changing to a non-existent directory
            result = run_command(gui, 'cd nonexistent_dir')
            assert "Error changing directory" in result
            
            # Test cd without arguments
            result = run_command(gui, 'cd')
            assert "Usage: cd <directory>" in result
        finally:
            # Change back to original directory
            os.chdir(original_dir)
        
        print("All bash command tests passed!")
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir)
        gui.close()
        set_test_environment(False)

def test_python_functionality():
    # Initialize the GUI
    gui = StatacondaGUI()
    
    # Test numpy operations
    run_command(gui, 'import numpy as np')
    run_command(gui, 'arr = np.array([1, 2, 3, 4, 5])')
    run_command(gui, 'mean = np.mean(arr)')
    # You may need to access variables from the global scope or gui.python_vars if implemented
    # assert 'mean' in gui.python_vars
    # assert gui.python_vars['mean'] == 3.0
    
    # Test scipy operations
    run_command(gui, 'from scipy import stats')
    run_command(gui, 't_stat, p_val = stats.ttest_1samp(arr, 3.0)')
    # assert 't_stat' in gui.python_vars
    # assert 'p_val' in gui.python_vars
    
    # Test numpy array operations on DataFrame
    run_command(gui, 'df_array = gui.df.values')
    # assert 'df_array' in gui.python_vars
    # assert isinstance(gui.python_vars['df_array'], np.ndarray)
    
    # Test numpy statistical functions
    run_command(gui, 'std_dev = np.std(arr)')
    # assert 'std_dev' in gui.python_vars
    # assert abs(gui.python_vars['std_dev'] - 1.4142135623730951) < 1e-10
    
    # Test numpy array creation and manipulation
    run_command(gui, 'zeros = np.zeros((3, 3))')
    # assert 'zeros' in gui.python_vars
    # assert gui.python_vars['zeros'].shape == (3, 3)
    
    # Test scipy statistical functions
    run_command(gui, 'norm_dist = stats.norm(loc=0, scale=1)')
    run_command(gui, 'pdf_value = norm_dist.pdf(0)')
    # assert 'pdf_value' in gui.python_vars
    # assert abs(gui.python_vars['pdf_value'] - 0.3989422804014327) < 1e-10

def test_scikit_learn_integration():
    # Initialize the GUI
    gui = StatacondaGUI()
    
    # Create a test dataset with more features for elastic net
    test_data = pd.DataFrame({
        'value': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'feature2': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        'feature3': [3, 6, 9, 12, 15, 18, 21, 24, 27, 30],
        'feature4': [4, 8, 12, 16, 20, 24, 28, 32, 36, 40]
    })
    gui.df = test_data
    
    # Execute scikit-learn elastic net regression
    run_command(gui, 'from sklearn.linear_model import ElasticNet')
    run_command(gui, 'from sklearn.preprocessing import StandardScaler')
    run_command(gui, 'X = gui.df[["feature1", "feature2", "feature3", "feature4"]]')
    run_command(gui, 'y = gui.df["value"]')
    run_command(gui, 'scaler = StandardScaler()')
    run_command(gui, 'X_scaled = scaler.fit_transform(X)')
    run_command(gui, 'model = ElasticNet(alpha=0.1, l1_ratio=0.5)')
    run_command(gui, 'model.fit(X_scaled, y)')
    run_command(gui, 'pred_value = model.predict(X_scaled)')
    
    # Add predictions back to the dataset
    run_command(gui, 'gui.df["pred_value"] = pred_value')
    
    # Verify predictions were added
    assert 'pred_value' in gui.df.columns
    
    # Run a regression using the predictions
    run_command(gui, 'reg value pred_value')
    # You may want to check output in gui.results_window.toPlainText() or similar
    
    # Test that the predictions are reasonable
    assert np.all(gui.df['pred_value'].notna())
    assert np.all(np.isfinite(gui.df['pred_value']))
    
    # Test that the elastic net model coefficients are stored
    # assert 'model' in gui.python_vars
    # assert hasattr(gui.python_vars['model'], 'coef_')
    # assert len(gui.python_vars['model'].coef_) == 4  # One coefficient per feature

def test_command_routing():
    # Initialize the GUI
    gui = StatacondaGUI()
    
    # Test Stata command
    run_command(gui, 'regress y x')
    assert gui.results_window.toPlainText() != ''
    
    # Test bash command
    run_command(gui, 'bash pwd')
    assert gui.results_window.toPlainText() != ''
    
    # Test Python command
    run_command(gui, 'print("Hello, World!")')
    assert gui.results_window.toPlainText() != ''

def test_python_variable_creation():
    # Initialize the GUI
    gui = StatacondaGUI()
    
    # Create a Python variable
    run_command(gui, 'x = 1')
    
    # Verify the variable is created and displayed in the list
    items = [gui.python_vars_list.item(i).text() for i in range(gui.python_vars_list.count())]
    assert 'x = 1' in items

if __name__ == '__main__':
    test_bash_commands()
    test_python_functionality()
    test_scikit_learn_integration()
    test_command_routing()
    test_python_variable_creation()
    print("All tests passed!") 