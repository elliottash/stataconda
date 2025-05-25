import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QTextEdit, QLineEdit, QLabel, QListWidget, QListWidgetItem,
                            QTableWidget, QTableWidgetItem, QDialog, QSplitter, QTabWidget,
                            QComboBox, QSpinBox, QMenu)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QTextCharFormat, QColor, QFont, QSyntaxHighlighter
import io
import contextlib
import pandas as pd
import statsmodels.api as sm
import numpy as np
from linearmodels import PanelOLS
import re
import matplotlib.pyplot as plt
import subprocess
import os
import inspect
import math
import patsy
import shutil
import tempfile
from scipy import stats
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler


class StataHighlighter(QSyntaxHighlighter):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.highlighting_rules = []

        # Bold black for commands (only the first line of each command)
        command_format = QTextCharFormat()
        command_format.setFontWeight(QFont.Bold)
        command_format.setForeground(QColor("black"))
        self.highlighting_rules.append((re.compile(r'^[a-z]+.*$'), command_format))

        # Regular black for output
        output_format = QTextCharFormat()
        output_format.setFontWeight(QFont.Normal)
        output_format.setForeground(QColor("black"))
        self.highlighting_rules.append((re.compile(r'^(?!Error:).*$'), output_format))

        # Red for errors
        error_format = QTextCharFormat()
        error_format.setFontWeight(QFont.Normal)
        error_format.setForeground(QColor("red"))
        self.highlighting_rules.append((re.compile(r'^Error:.*$'), error_format))

    def highlightBlock(self, text):
        for pattern, format in self.highlighting_rules:
            for match in pattern.finditer(text):
                start = match.start()
                length = match.end() - start
                self.setFormat(start, length, format)


class DataBrowser(QDialog):
    def __init__(self, df, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Data Browser")
        self.setGeometry(100, 100, 1000, 700)
        
        # Store the DataFrame
        self.df = df
        
        # Create main layout
        layout = QVBoxLayout()
        
        # Add toolbar
        toolbar = QHBoxLayout()
        
        # Search box
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search...")
        self.search_box.textChanged.connect(self.filter_data)
        toolbar.addWidget(self.search_box)
        
        # Format options
        self.format_combo = QComboBox()
        self.format_combo.addItems(["Default", "Scientific", "Fixed", "Percent"])
        self.format_combo.currentTextChanged.connect(self.update_format)
        toolbar.addWidget(QLabel("Format:"))
        toolbar.addWidget(self.format_combo)
        
        # Decimals spinbox
        self.decimals_spin = QSpinBox()
        self.decimals_spin.setRange(0, 10)
        self.decimals_spin.setValue(2)
        self.decimals_spin.valueChanged.connect(self.update_format)
        toolbar.addWidget(QLabel("Decimals:"))
        toolbar.addWidget(self.decimals_spin)
        
        toolbar.addStretch()
        layout.addLayout(toolbar)
        
        # Create table widget
        self.table = QTableWidget()
        self.table.setSortingEnabled(True)
        self.table.horizontalHeader().sectionClicked.connect(self.sort_column)
        self.table.setAlternatingRowColors(True)
        layout.addWidget(self.table)
        
        # Add status bar
        self.status_bar = QLabel()
        layout.addWidget(self.status_bar)
        
        self.setLayout(layout)
        
        # Populate table
        self.populate_table()
        
        # Set up context menu
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self.show_context_menu)
        
    def populate_table(self):
        """Populate the table with DataFrame data"""
        # Set dimensions
        self.table.setRowCount(len(self.df))
        self.table.setColumnCount(len(self.df.columns))
        
        # Set headers with variable labels if available
        headers = []
        for col in self.df.columns:
            label = getattr(self.df[col], 'label', col)
            headers.append(f"{col}\n{label}" if label != col else col)
        self.table.setHorizontalHeaderLabels(headers)
        
        # Fill the table with data
        for i in range(len(self.df)):
            for j, col in enumerate(self.df.columns):
                value = self.df.iloc[i, j]
                # Convert to string for display
                if pd.isnull(value):
                    display_value = ""
                else:
                    display_value = str(value)
                item = QTableWidgetItem(display_value)
                item.setData(Qt.DisplayRole, display_value)  # Store display string for sorting
                self.table.setItem(i, j, item)
        
        # Resize columns to content
        self.table.resizeColumnsToContents()
        
        # Update status bar
        self.update_status_bar()
        
    def filter_data(self):
        """Filter table based on search text"""
        search_text = self.search_box.text().lower()
        
        for i in range(self.table.rowCount()):
            show_row = False
            for j in range(self.table.columnCount()):
                item = self.table.item(i, j)
                if item and search_text in item.text().lower():
                    show_row = True
                    break
            self.table.setRowHidden(i, not show_row)
            
        self.update_status_bar()
        
    def sort_column(self, column):
        """Sort table by column"""
        self.table.sortItems(column)
        
    def update_format(self):
        """Update number formatting"""
        format_type = self.format_combo.currentText()
        decimals = self.decimals_spin.value()
        
        for i in range(self.table.rowCount()):
            for j in range(self.table.columnCount()):
                item = self.table.item(i, j)
                if item:
                    value = item.data(Qt.DisplayRole)
                    if isinstance(value, (int, float)):
                        if format_type == "Scientific":
                            display_value = f"{value:.{decimals}e}"
                        elif format_type == "Fixed":
                            display_value = f"{value:.{decimals}f}"
                        elif format_type == "Percent":
                            display_value = f"{value:.{decimals}%}"
                        else:  # Default
                            display_value = f"{value:,.{decimals}f}"
                        item.setText(display_value)
        
    def update_status_bar(self):
        """Update status bar with current information"""
        visible_rows = sum(1 for i in range(self.table.rowCount()) if not self.table.isRowHidden(i))
        self.status_bar.setText(f"Rows: {visible_rows}/{len(self.df)} | Columns: {len(self.df.columns)}")
        
    def show_context_menu(self, position):
        """Show context menu for copy/paste operations"""
        menu = QMenu()
        copy_action = menu.addAction("Copy")
        paste_action = menu.addAction("Paste")
        
        action = menu.exec_(self.table.mapToGlobal(position))
        
        if action == copy_action:
            self.copy_selection()
        elif action == paste_action:
            self.paste_selection()
            
    def copy_selection(self):
        """Copy selected cells to clipboard"""
        selected = self.table.selectedRanges()
        if not selected:
            return
            
        text = []
        for r in selected:
            for row in range(r.topRow(), r.bottomRow() + 1):
                row_data = []
                for col in range(r.leftColumn(), r.rightColumn() + 1):
                    item = self.table.item(row, col)
                    row_data.append(item.text() if item else "")
                text.append("\t".join(row_data))
                
        QApplication.clipboard().setText("\n".join(text))
        
    def paste_selection(self):
        """Paste data from clipboard"""
        text = QApplication.clipboard().text()
        if not text:
            return
            
        rows = text.split("\n")
        start_row = self.table.currentRow()
        start_col = self.table.currentColumn()
        
        for i, row in enumerate(rows):
            if start_row + i >= self.table.rowCount():
                break
            cols = row.split("\t")
            for j, value in enumerate(cols):
                if start_col + j >= self.table.columnCount():
                    break
                item = self.table.item(start_row + i, start_col + j)
                if item:
                    item.setText(value)


class StatacondaGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Store initial variables for Python variables pane
        self._initial_vars = set(globals().keys())
        
        # Initialize DataFrame
        self._df = sm.datasets.grunfeld.load_pandas().data
        
        # Initialize persistent Python execution context
        self._python_context = {
            'pd': pd,
            'np': np,
            'sm': sm,
            'plt': plt,
            'gui': self,
            '_df': self._df,
        }
        
        # Set up the main window
        self.setWindowTitle("Stataconda")
        self.setGeometry(100, 100, 1200, 800)
        
        # Set application style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QWidget {
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 10pt;
            }
            QLabel {
                color: #2c3e50;
                font-weight: bold;
                padding: 5px;
            }
            QTextEdit, QLineEdit {
                background-color: white;
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                padding: 5px;
                font-family: 'Consolas', 'Courier New', monospace;
            }
            QTextEdit:focus, QLineEdit:focus {
                border: 1px solid #3498db;
            }
            QListWidget {
                background-color: white;
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                padding: 5px;
                font-family: 'Consolas', 'Courier New', monospace;
            }
            QListWidget::item {
                padding: 5px;
                border-bottom: 1px solid #ecf0f1;
            }
            QListWidget::item:selected {
                background-color: #3498db;
                color: white;
            }
            QListWidget::item:hover {
                background-color: #e0e0e0;
            }
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 8px 15px;
                border-radius: 4px;
                font-family: 'Consolas', 'Courier New', monospace;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #2472a4;
            }
        """)

        # Command history
        self.command_history = []
        self.history_index = -1

        # Dictionary to store all DataFrames
        self._dataframes = {}
        self._current_df_name = None
        
        # Initialize stored estimates
        self._stored_estimates = StoredEstimates()

        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # Create main splitter
        main_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(main_splitter)

        # Left side (Datasets and History)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(5)

        # Create a vertical splitter for Datasets and History
        left_splitter = QSplitter(Qt.Vertical)
        left_layout.addWidget(left_splitter)

        # Datasets pane
        dataframes_widget = QWidget()
        dataframes_layout = QVBoxLayout(dataframes_widget)
        dataframes_layout.setContentsMargins(0, 0, 0, 0)
        dataframes_layout.setSpacing(5)
        
        # Add a styled header for Datasets
        dataframes_header = QWidget()
        dataframes_header_layout = QHBoxLayout(dataframes_header)
        dataframes_header_layout.setContentsMargins(0, 0, 0, 0)
        dataframes_label = QLabel("Datasets")
        dataframes_label.setStyleSheet("font-size: 12pt; font-weight: bold; color: #2c3e50; font-family: 'Consolas', 'Courier New', monospace;")
        dataframes_header_layout.addWidget(dataframes_label)
        dataframes_layout.addWidget(dataframes_header)
        
        self.dataframes_list = QListWidget()
        self.dataframes_list.setStyleSheet("""
            QListWidget {
                background-color: white;
                border: 1px solid #bdc3c7;
                border-radius: 4px;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #ecf0f1;
            }
            QListWidget::item:selected {
                background-color: #3498db;
                color: white;
            }
        """)
        self.dataframes_list.itemClicked.connect(self.on_dataframe_selected)
        dataframes_layout.addWidget(self.dataframes_list)
        left_splitter.addWidget(dataframes_widget)

        # History pane
        history_widget = QWidget()
        history_layout = QVBoxLayout(history_widget)
        history_layout.setContentsMargins(0, 0, 0, 0)
        history_layout.setSpacing(5)
        
        # Add a styled header for History
        history_header = QWidget()
        history_header_layout = QHBoxLayout(history_header)
        history_header_layout.setContentsMargins(0, 0, 0, 0)
        history_label = QLabel("Command History")
        history_label.setStyleSheet("font-size: 12pt; font-weight: bold; color: #2c3e50; font-family: 'Consolas', 'Courier New', monospace;")
        history_header_layout.addWidget(history_label)
        history_layout.addWidget(history_header)
        
        self.history_filter = QLineEdit()
        self.history_filter.setPlaceholderText("Filter history...")
        self.history_filter.textChanged.connect(self.filter_history)
        self.history_filter.setStyleSheet("""
            QLineEdit {
                padding: 8px;
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                background-color: white;
            }
        """)
        history_layout.addWidget(self.history_filter)
        
        self.history_list = QListWidget()
        self.history_list.setStyleSheet("""
            QListWidget {
                background-color: white;
                border: 1px solid #bdc3c7;
                border-radius: 4px;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #ecf0f1;
            }
            QListWidget::item:selected {
                background-color: #3498db;
                color: white;
            }
        """)
        self.history_list.itemDoubleClicked.connect(self.on_history_double_clicked)
        history_layout.addWidget(self.history_list)
        left_splitter.addWidget(history_widget)

        # Set initial sizes for left splitter (30% Datasets, 70% History)
        left_splitter.setSizes([200, 500])

        main_splitter.addWidget(left_widget)

        # Middle section (command prompt and results)
        middle_widget = QWidget()
        middle_layout = QVBoxLayout(middle_widget)
        middle_layout.setContentsMargins(0, 0, 0, 0)
        middle_layout.setSpacing(10)
        
        # Add a styled header for Results
        results_header = QWidget()
        results_header_layout = QHBoxLayout(results_header)
        results_header_layout.setContentsMargins(0, 0, 0, 0)
        results_label = QLabel("Results")
        results_label.setStyleSheet("font-size: 12pt; font-weight: bold; color: #2c3e50; font-family: 'Consolas', 'Courier New', monospace;")
        results_header_layout.addWidget(results_label)
        middle_layout.addWidget(results_header)
        
        self.results_window = QTextEdit()
        self.results_window.setReadOnly(True)
        self.results_window.setFont(QFont("Consolas", 10))
        self.results_window.setStyleSheet("""
            QTextEdit {
                background-color: white;
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                padding: 10px;
            }
        """)
        self.highlighter = StataHighlighter(self.results_window.document())
        middle_layout.addWidget(self.results_window)
        
        # Add a styled header for Command Prompt
        prompt_header = QWidget()
        prompt_header_layout = QHBoxLayout(prompt_header)
        prompt_header_layout.setContentsMargins(0, 0, 0, 0)
        prompt_label = QLabel("Command Prompt")
        prompt_label.setStyleSheet("font-size: 12pt; font-weight: bold; color: #2c3e50; font-family: 'Consolas', 'Courier New', monospace;")
        prompt_header_layout.addWidget(prompt_label)
        middle_layout.addWidget(prompt_header)
        
        self.command_prompt = QTextEdit()
        self.command_prompt.setPlaceholderText("Enter commands here (Shift+Enter for new line)")
        self.command_prompt.setMaximumHeight(100)
        self.command_prompt.setFont(QFont("Consolas", 10))
        self.command_prompt.setStyleSheet("""
            QTextEdit {
                background-color: white;
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                padding: 10px;
            }
        """)
        self.command_prompt.keyPressEvent = self.handle_key_press
        middle_layout.addWidget(self.command_prompt)
        main_splitter.addWidget(middle_widget)

        # Right side (Dataset Variables and Python Variables)
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)

        # Create a vertical splitter for Dataset Variables and Python Variables
        right_splitter = QSplitter(Qt.Vertical)
        right_layout.addWidget(right_splitter)

        # Dataset Variables pane
        variables_widget = QWidget()
        variables_layout = QVBoxLayout(variables_widget)
        variables_layout.setContentsMargins(0, 0, 0, 0)
        variables_layout.setSpacing(5)
        
        # Add a styled header for Variables
        variables_header = QWidget()
        variables_header_layout = QHBoxLayout(variables_header)
        variables_header_layout.setContentsMargins(0, 0, 0, 0)
        variables_label = QLabel("Dataset Variables")
        variables_label.setStyleSheet("font-size: 12pt; font-weight: bold; color: #2c3e50; font-family: 'Consolas', 'Courier New', monospace;")
        variables_header_layout.addWidget(variables_label)
        variables_layout.addWidget(variables_header)
        
        self.variables_filter = QLineEdit()
        self.variables_filter.setPlaceholderText("Filter variables...")
        self.variables_filter.textChanged.connect(self.filter_variables)
        self.variables_filter.setStyleSheet("""
            QLineEdit {
                padding: 8px;
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                background-color: white;
            }
        """)
        variables_layout.addWidget(self.variables_filter)
        
        self.variables_list = QListWidget()
        self.variables_list.setStyleSheet("""
            QListWidget {
                background-color: white;
                border: 1px solid #bdc3c7;
                border-radius: 4px;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #ecf0f1;
            }
            QListWidget::item:selected {
                background-color: #3498db;
                color: white;
            }
        """)
        self.variables_list.itemDoubleClicked.connect(self.on_variable_double_clicked)
        variables_layout.addWidget(self.variables_list)
        right_splitter.addWidget(variables_widget)

        # Python Variables pane
        python_vars_widget = QWidget()
        python_vars_layout = QVBoxLayout(python_vars_widget)
        python_vars_layout.setContentsMargins(0, 0, 0, 0)
        python_vars_layout.setSpacing(5)
        
        # Add a styled header for Python Variables
        python_vars_header = QWidget()
        python_vars_header_layout = QHBoxLayout(python_vars_header)
        python_vars_header_layout.setContentsMargins(0, 0, 0, 0)
        python_vars_label = QLabel("Python Variables")
        python_vars_label.setStyleSheet("font-size: 12pt; font-weight: bold; color: #2c3e50; font-family: 'Consolas', 'Courier New', monospace;")
        python_vars_header_layout.addWidget(python_vars_label)
        python_vars_layout.addWidget(python_vars_header)
        
        self.python_vars_list = QListWidget()
        self.python_vars_list.setStyleSheet("""
            QListWidget {
                background-color: white;
                border: 1px solid #bdc3c7;
                border-radius: 4px;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #ecf0f1;
            }
            QListWidget::item:selected {
                background-color: #3498db;
                color: white;
            }
        """)
        self.python_vars_list.itemDoubleClicked.connect(self.on_python_var_double_clicked)
        python_vars_layout.addWidget(self.python_vars_list)
        right_splitter.addWidget(python_vars_widget)

        # Set initial sizes for right splitter (70% Dataset Variables, 30% Python Variables)
        right_splitter.setSizes([400, 200])

        main_splitter.addWidget(right_widget)

        # Set initial sizes (20%, 60%, 20%)
        main_splitter.setSizes([280, 840, 280])

        # Initialize with Grunfeld dataset as 'main1'
        self._dataframes['main1'] = self._df
        self._current_df_name = 'main1'
        self.update_dataframes_list()
        self.update_variables_pane()
        self.update_python_vars()

        # List of reserved Stata commands
        self.stata_commands = {
            'use', 'save', 'import', 'export', 'describe', 'codebook', 'browse',
            'reghdfe', 'ivreghdfe', 'summarize', 'tabulate', 'histogram', 'graph',
            'generate', 'replace', 'drop', 'keep', 'rename', 'recode', 'clonevar',
            'label', 'append', 'merge', 'joinby', 'cross', 'regress', 'anova',
            'areg', 'xtreg', 'fe', 'logit', 'probit', 'logistic', 'poisson',
            'nbreg', 'tobit', 'intreg', 'ivregress', 'ivreg2', 'control_function',
            'xtset', 'xtlogit', 'xtprobit', 'xtpoisson', 'xtabond', 'xtdpdsys',
            'tsset', 'arima', 'arch', 'var', 'vec', 'newey', 'binscatter',
            'coefplot', 'lgraph', 'hist', 'do', 'scatter',
            # Abbreviations
            'sum', 'reg', 'gen', 'g', 'ren', 'd', 'l', 'tab', 'mer', 'app', 'coll',
            'resh', 'sor', 'bro', 'ed', 'lab', 'su'
        }

        # Set up command registry
        self.setup_command_registry()

        # Set focus to command prompt
        self.command_prompt.setFocus()

    def setup_command_registry(self):
        # Command registry for Stata-like commands
        self.command_registry = {
            # Full commands
            'use': self.cmd_use,
            'save': self.cmd_save,
            'import': self.cmd_import,
            'export': self.cmd_export,
            'describe': self.cmd_describe,
            'codebook': self.cmd_codebook,
            'browse': lambda cmd: self.open_data_browser(),
            'bro': lambda cmd: self.open_data_browser(),  # Add 'bro' abbreviation
            'reghdfe': self.translate_reghdfe,
            'reg': self.translate_reghdfe,  # reg abbreviation
            'regress': self.translate_reghdfe,  # regress abbreviation
            'ivreghdfe': self.translate_ivreghdfe,
            'ivreg': self.translate_ivreghdfe,  # ivreg abbreviation
            'ivregress': self.translate_ivreghdfe,  # ivregress abbreviation
            'summarize': self.cmd_summarize,
            'su': self.cmd_summarize,  # Add 'su' abbreviation
            'tabulate': self.cmd_tabulate,
            'histogram': self.cmd_histogram,
            'graph': self.cmd_graph_bar,
            'scatter': self.cmd_scatter,
            'generate': self.cmd_generate,
            'gen': self.cmd_generate,  # Add gen abbreviation
            'g': self.cmd_generate,    # Add g abbreviation
            'replace': self.cmd_replace,
            'drop': self.cmd_drop,
            'keep': self.cmd_keep,
            'rename': self.cmd_rename,
            'ren': self.cmd_rename,    # Add ren abbreviation
            'recode': self.cmd_recode,
            'clonevar': self.cmd_clonevar,
            'label': self.cmd_label_variable,
            'append': self.cmd_append,
            'merge': self.cmd_merge,
            'joinby': self.cmd_joinby,
            'cross': self.cmd_cross,
            'anova': self.cmd_anova,
            'areg': self.cmd_areg,
            'xtreg': self.cmd_xtreg,
            'fe': self.cmd_fe,
            'logit': self.cmd_logit,
            'probit': self.cmd_probit,
            'logistic': self.cmd_logistic,
            'poisson': self.cmd_poisson,
            'nbreg': self.cmd_nbreg,
            'tobit': self.cmd_tobit,
            'intreg': self.cmd_intreg,
            'control_function': self.cmd_ivregress,
            'xtset': self.cmd_xtset,
            'xtlogit': self.cmd_xtlogit,
            'xtprobit': self.cmd_xtprobit,
            'xtpoisson': self.cmd_xtpoisson,
            'xtabond': self.cmd_xtabond,
            'xtdpdsys': self.cmd_xtdpdsys,
            'tsset': self.cmd_tsset,
            'arima': self.cmd_arima,
            'arch': self.cmd_arch,
            'var': self.cmd_var,
            'vec': self.cmd_vec,
            'newey': self.cmd_newey,
            'binscatter': self.cmd_binscatter,
            'coefplot': self.cmd_coefplot,
            'lgraph': self.cmd_lgraph,
            'hist': self.cmd_hist,
            'do': self.cmd_do,
            'egen': self.cmd_egen,
            'estout': self.cmd_estout,
            'eststo': self.cmd_eststo,
            'esttab': self.cmd_esttab,
            'list': self.cmd_list,
            'collapse': self.cmd_collapse,
            'reshape': self.cmd_reshape,
            'sort': self.cmd_sort,
            'edit': self.cmd_edit,
            'bash': self.cmd_bash,
            'browse': self.cmd_browse,
        }
        # Add egen to command registry
        self.command_registry['egen'] = self.cmd_egen

    def load_initial_data(self):
        """Load initial data into the application"""
        try:
            import statsmodels.api as sm
            # Load the Grunfeld data from statsmodels
            self._df = sm.datasets.grunfeld.load_pandas().data
            # Store in dataframes dictionary with the name "main1"
            self._dataframes["main1"] = self._df
            # Update the dataframes list
            self.update_dataframes_list()
            # Update variables pane
            self.update_variables_pane()
            # Set as active dataframe
            self.on_dataframe_selected(self.dataframes_list.item(0))
        except Exception as e:
            self.results_window.append(f"Error loading data: {str(e)}")

    def parse_command(self, command):
        """
        Splits a command into (main_part, options_part) at the first comma.
        Returns (main_part:str, options_part:str or None)
        """
        if ',' in command:
            main, options = command.split(',', 1)
            return main.strip(), options.strip()
        else:
            return command.strip(), None

    def translate_reghdfe(self, command):
        # Parse the reghdfe command
        # Example: reghdfe y x1 x2, absorb(firm year) cluster(firm)
        pattern = r'(reghdfe|reg|regress)\s+(\w+)\s+([^,]+)(?:,\s*absorb\(([^)]+)\))?(?:\s*cluster\(([^)]+)\))?'
        match = re.match(pattern, command)
        
        if not match:
            return "Usage: reghdfe <depvar> <indepvars> [, absorb(varlist) cluster(varlist)]"
        
        _, dep_var, indep_vars, absorb_vars, cluster_vars = match.groups()
        indep_vars = [var.strip() for var in indep_vars.split()]
        
        # Convert absorb variables to list
        if absorb_vars:
            absorb_vars = [var.strip() for var in absorb_vars.split()]
        
        # Convert cluster variables to list
        if cluster_vars:
            cluster_vars = [var.strip() for var in cluster_vars.split()]
        
        try:
            # Prepare the data
            df = self._df.copy()
            
            # Ensure all variables are numeric
            for var in [dep_var] + indep_vars:
                df[var] = pd.to_numeric(df[var], errors='coerce')
            
            # Create dummy variables for fixed effects
            if absorb_vars:
                for var in absorb_vars:
                    dummies = pd.get_dummies(df[var], prefix=var, drop_first=True)
                    df = pd.concat([df, dummies], axis=1)
            
            # Prepare X and y
            y = df[dep_var]
            X = df[indep_vars]
            
            # Add constant only if no absorb variables
            if not absorb_vars:
                X = sm.add_constant(X)
            
            # Drop missing values
            mask = ~(y.isna() | X.isna().any(axis=1))
            y_clean = y[mask]
            X_clean = X[mask]
            
            # Count dropped observations
            n_dropped = len(df) - len(y_clean)
            
            # Fit the model
            model = sm.OLS(y_clean, X_clean)
            results = model.fit()
            
            # Store results for coefplot
            self._stored_estimates.store('lastreg', results, 'ols', dep_var, indep_vars)
            self._lastreg = results
            self._stored_estimates.current_name = 'lastreg'
            
            # Format output to look like Stata
            output = []
            output.append("Linear regression (reghdfe style)")
            output.append(f"Number of obs = {len(y_clean)}")
            output.append(f"F({results.df_model}, {results.df_resid}) = {results.fvalue:.2f}")
            output.append(f"Prob > F      = {results.f_pvalue:.4f}")
            output.append(f"R-squared     = {results.rsquared:.4f}")
            output.append(f"Adj R-squared = {results.rsquared_adj:.4f}")
            output.append(f"Root MSE      = {np.sqrt(results.mse_resid):.4f}")
            
            if absorb_vars:
                output.append(f"Absorbed fixed effects: {', '.join(absorb_vars)}")
            if cluster_vars:
                output.append(f"Clustered on: {', '.join(cluster_vars)}")
            
            output.append("")
            output.append("------------------------------------------------------------------------------")
            output.append(f"             |               Robust")
            output.append(f"{dep_var:12} |      Coef.   Std. Err.      t    P>|t|     [95% Conf. Interval]")
            output.append("-------------+----------------------------------------------------------------")
            
            for name in results.params.index:
                if name == 'const' or name in indep_vars:
                    coef = results.params[name]
                    std_err = results.bse[name]
                    t_stat = results.tvalues[name]
                    p_value = results.pvalues[name]
                    ci = results.conf_int().loc[name]
                    ci_lower, ci_upper = ci[0], ci[1]
                    output.append(f"{name:12} | {coef:10.4f} {std_err:10.4f} {t_stat:7.2f} {p_value:7.4f} {ci_lower:10.4f} {ci_upper:10.4f}")
            
            output.append("------------------------------------------------------------------------------")
            
            # Add note about dropped observations
            if n_dropped > 0:
                output.append(f"\nNote: {n_dropped} observations dropped due to missing values")
            
            return "\n".join(output)
            
        except Exception as e:
            return f"Error: {str(e)}"

    def translate_ivreghdfe(self, command):
        # Parse the ivreghdfe command
        # Example: ivreghdfe invest (capital = L1_capital) value, absorb(firm year) cluster(firm)
        pattern = r'ivreghdfe\s+(\w+)\s+\((\w+)\s*=\s*(\w+)\)(?:\s+(\w+))?(?:,\s*absorb\(([^)]+)\))?(?:\s*cluster\(([^)]+)\))?'
        match = re.match(pattern, command)
        
        if not match:
            return None, "Invalid ivreghdfe command format"
        
        dep_var, endog_var, instrument, exog_var, absorb_vars, cluster_var = match.groups()
        
        # Convert absorb variables to list
        if absorb_vars:
            absorb_vars = [var.strip() for var in absorb_vars.split()]
        
        # Prepare the data
        df = self._df.copy()
        
        # Create dummy variables for fixed effects
        if absorb_vars:
            for var in absorb_vars:
                dummies = pd.get_dummies(df[var], prefix=var, drop_first=True)
                df = pd.concat([df, dummies], axis=1)
        try:
            # Prepare the model
            y = df[dep_var]
            Z = df[endog_var].values.reshape(-1, 1)  # Endogenous variable
            W = df[instrument].values.reshape(-1, 1)  # Instrument
            # Handle exogenous variables if present
            if exog_var:
                X = df[exog_var].values.reshape(-1, 1)
            else:
                X = np.zeros((len(df), 0))  # Empty array for no exogenous variables
                
            # Drop missing values
            mask = ~(np.isnan(y) | np.isnan(Z).any(axis=1) | np.isnan(W).any(axis=1))
            if exog_var:
                mask = mask & ~np.isnan(X).any(axis=1)
            
            y_clean = y[mask]
            Z_clean = Z[mask]
            W_clean = W[mask]
            if exog_var:
                X_clean = X[mask]
            else:
                X_clean = np.zeros((len(y_clean), 0))
            
            # Count dropped observations
            n_dropped = len(df) - len(y_clean)
            
            # First stage regression
            first_stage = sm.OLS(Z_clean, sm.add_constant(W_clean)).fit()
            
            # Calculate Kleibergen-Paap F-statistic
            n = len(y_clean)
            k = 1  # number of instruments
            f_stat = (first_stage.fvalue * (n - k - 1) / n)
            
            # Second stage regression
            Z_hat = first_stage.predict(sm.add_constant(W_clean))
            if exog_var:
                X_full = np.column_stack([X_clean, Z_hat])
            else:
                X_full = Z_hat
            
            # Add constant only if no absorb variables
            if not absorb_vars:
                X_full = sm.add_constant(X_full)
            
            # Fit the model
            if cluster_var:
                model = sm.OLS(y_clean, X_full)
                results = model.fit(cov_type='cluster', cov_kwds={'groups': df[cluster_var][mask]})
            else:
                model = sm.OLS(y_clean, X_full)
                results = model.fit()
            
            # Format output to look like Stata
            output = []
            output.append("Instrumental variables (2SLS) regression")
            output.append(f"Number of obs = {len(y_clean)}")
            output.append(f"F({results.df_model}, {results.df_resid}) = {results.fvalue:.2f}")
            output.append(f"Prob > F      = {results.f_pvalue:.4f}")
            output.append(f"R-squared     = {results.rsquared:.4f}")
            output.append(f"Adj R-squared = {results.rsquared_adj:.4f}")
            output.append(f"Root MSE      = {np.sqrt(results.mse_resid):.4f}")
            output.append("")
            output.append("------------------------------------------------------------------------------")
            output.append(f"             |               Robust")
            output.append(f"{dep_var:12} |      Coef.   Std. Err.      t    P>|t|     [95% Conf. Interval]")
            output.append("-------------+----------------------------------------------------------------")
            
            # Add coefficients
            if exog_var:
                for i, name in enumerate([exog_var, endog_var]):
                    coef = results.params.iloc[i]
                    std_err = results.bse.iloc[i]
                    t_stat = results.tvalues.iloc[i]
                    p_value = results.pvalues.iloc[i]
                    ci = results.conf_int().iloc[i]
                    ci_lower, ci_upper = ci[0], ci[1]
                    
                    output.append(f"{name:12} | {coef:10.4f} {std_err:10.4f} {t_stat:7.2f} {p_value:7.4f} {ci_lower:10.4f} {ci_upper:10.4f}")
            else:
                coef = results.params.iloc[0]
                std_err = results.bse.iloc[0]
                t_stat = results.tvalues.iloc[0]
                p_value = results.pvalues.iloc[0]
                ci = results.conf_int().iloc[0]
                ci_lower, ci_upper = ci[0], ci[1]
                
                output.append(f"{endog_var:12} | {coef:10.4f} {std_err:10.4f} {t_stat:7.2f} {p_value:7.4f} {ci_lower:10.4f} {ci_upper:10.4f}")
            
            output.append("------------------------------------------------------------------------------")
            output.append("")
            output.append("Instrumented: " + endog_var)
            output.append("Instruments:  " + instrument)
            
            # Add first stage regression results
            first_stage = sm.OLS(Z_clean, sm.add_constant(W_clean)).fit()
            output.append("")
            output.append("First-stage regression")
            output.append("------------------------------------------------------------------------------")
            output.append(f"{endog_var:12} |      Coef.   Std. Err.      t    P>|t|     [95% Conf. Interval]")
            output.append("-------------+----------------------------------------------------------------")
            
            # Add first stage coefficients (skip constant)
            params = first_stage.params
            bse = first_stage.bse
            tvalues = first_stage.tvalues
            pvalues = first_stage.pvalues
            confint = first_stage.conf_int()
            if hasattr(params, 'index') and len(params) > 1:
                idx = 1
            else:
                idx = 0
            coef = params[idx]
            std_err = bse[idx]
            t_stat = tvalues[idx]
            p_value = pvalues[idx]
            ci = confint.iloc[idx] if hasattr(confint, 'iloc') else confint[idx]
            ci_lower, ci_upper = ci[0], ci[1]
            output.append(f"{instrument:12} | {coef:10.4f} {std_err:10.4f} {t_stat:7.2f} {p_value:7.4f} {ci_lower:10.4f} {ci_upper:10.4f}")
            output.append("------------------------------------------------------------------------------")
            output.append(f"Kleibergen-Paap F-statistic: {f_stat:.2f}")
            
            # Add note about dropped observations
            if n_dropped > 0:
                output.append(f"\nNote: {n_dropped} observations dropped due to missing values")
            
            # Store the model for later use
            self._stored_estimates.store('lastreg', results, 'iv', dep_var, [endog_var, exog_var] if exog_var else [endog_var])
            self._lastreg = results
            self._stored_estimates.current_name = 'lastreg'
            
            return "\n".join(output)
            
        except Exception as e:
            return f"Error: {str(e)}"

    def on_history_double_clicked(self, item):
        self.command_prompt.setPlainText(item.text())

    def on_variable_double_clicked(self, item):
        current_text = self.command_prompt.toPlainText()
        self.command_prompt.setPlainText(current_text + item.text())

    def filter_history(self):
        filter_text = self.history_filter.text().lower()
        self.history_list.clear()
        for item in self.command_history:
            if filter_text in item.lower():
                self.history_list.addItem(item)

    def filter_variables(self):
        filter_text = self.variables_filter.text().lower()
        self.variables_list.clear()
        for col in self._df.columns:
            if filter_text in col.lower():
                self.variables_list.addItem(col)

    def update_variables_pane(self):
        # Store current filter text
        filter_text = self.variables_filter.text()
        self.variables_list.clear()
        for col in self._df.columns:
            if filter_text.lower() in col.lower():
                self.variables_list.addItem(col)

    def handle_key_press(self, event):
        """Handle key press events in the command prompt"""
        if event.key() == Qt.Key_Return and event.modifiers() == Qt.ShiftModifier:
            # Insert new line
            self.command_prompt.insertPlainText('\n')
        elif event.key() == Qt.Key_Return:
            # Execute command
            self.execute_command()
        elif event.key() in (Qt.Key_Up, Qt.Key_PageUp):
            # Navigate command history (previous)
            if self.history_index < len(self.command_history) - 1:
                self.history_index += 1
                self.command_prompt.setPlainText(self.command_history[-(self.history_index + 1)])
        elif event.key() in (Qt.Key_Down, Qt.Key_PageDown):
            # Navigate command history (next)
            if self.history_index > 0:
                self.history_index -= 1
                self.command_prompt.setPlainText(self.command_history[-(self.history_index + 1)])
            elif self.history_index == 0:
                self.history_index = -1
                self.command_prompt.clear()
        else:
            # Default behavior
            QTextEdit.keyPressEvent(self.command_prompt, event)

    def eventFilter(self, obj, event):
        """Event filter for the command prompt"""
        if obj == self.command_prompt and event.type() == event.KeyPress:
            self.handle_key_press(event)
            return True
        return super().eventFilter(obj, event)

    def open_data_browser(self):
        # Keep a reference to the browser so it doesn't get garbage collected
        self._data_browser = DataBrowser(self._df, self)
        self._data_browser.show()

    def cmd_use(self, command):
        # Example: use mydata.dta
        parts = command.split()
        if len(parts) < 2:
            return 'Usage: use <filename>'
        filename = parts[1]
        try:
            if filename.endswith('.csv'):
                self._df = pd.read_csv(filename)
            elif filename.endswith('.xlsx'):
                self._df = pd.read_excel(filename)
            elif filename.endswith('.dta'):
                self._df = pd.read_stata(filename)
            else:
                # If no extension provided, default to .dta
                if '.' not in filename:
                    filename += '.dta'
                    self._df = pd.read_stata(filename)
                else:
                    return 'Unsupported file type. Supported types: .csv, .xlsx, .dta'
            # Update persistent context after loading DataFrame
            self._python_context['_df'] = self._df
            self.update_variables_pane()
            return f'Dataset loaded from {filename}'
        except Exception as e:
            return f'Error loading file: {e}'

    def cmd_save(self, command):
        # Example: save mydata.dta
        parts = command.split()
        if len(parts) < 2:
            return 'Usage: save <filename>'
        filename = parts[1]
        try:
            if filename.endswith('.csv'):
                self._df.to_csv(filename, index=False)
            elif filename.endswith('.xlsx'):
                self._df.to_excel(filename, index=False)
            elif filename.endswith('.dta'):
                self._df.to_stata(filename)
            else:
                # If no extension provided, default to .dta
                if '.' not in filename:
                    filename += '.dta'
                    self._df.to_stata(filename)
                else:
                    return 'Unsupported file type. Supported types: .csv, .xlsx, .dta'
            # Verify the DataFrame is saved correctly
            saved_df = pd.read_stata(filename)
            assert list(saved_df.columns) == list(self._df.columns)
            return f'Dataset saved to {filename}'
        except Exception as e:
            return f'Error saving file: {e}'

    def cmd_import(self, command):
        # Example: import mydata.csv
        return self.cmd_use(command.replace('import', 'use', 1))

    def cmd_export(self, command):
        # Example: export using filename.rtf
        parts = command.split()
        if len(parts) < 3 or parts[1] != 'using':
            return 'Usage: export using <filename>'
        filename = parts[2]
        try:
            if filename.endswith('.rtf'):
                # Create RTF file with regression results
                with open(filename, 'w') as f:
                    f.write('{\\rtf1\\ansi\\deff0\n')
                    f.write('{\\fonttbl{\\f0\\fnil\\fcharset0 Calibri;}}\n')
                    f.write('\\viewkind4\\uc1\\pard\\f0\\fs24\n')
                    
                    # Add regression results
                    for attr in dir(self):
                        if attr.startswith('_OLS_model_') or attr.startswith('_IV_model_'):
                            model = getattr(self, attr)
                            f.write(f'\\par {attr[1:]} Results:\\par\n')
                            f.write(model.summary().as_text().replace('\n', '\\par\n'))
                            f.write('\\par\\par\n')
                    
                    f.write('}\n')
                return f'Exported results to {filename}'
            else:
                return self.cmd_save(command.replace('export', 'save', 1))
        except Exception as e:
            return f'Error exporting results: {e}'

    def cmd_describe(self, command):
        # Overview of dataset variables
        desc = self._df.describe(include='all').T
        info = 'Variable Overview\n' + f"{desc}\n\nColumns: {', '.join(self._df.columns)}"
        return info

    def cmd_codebook(self, command):
        # Summary stats, value labels, etc.
        output = []
        for col in self._df.columns:
            output.append(f"{col}:")
            output.append(str(self._df[col].describe()))
            output.append("")
        return '\n'.join(output)

    def cmd_summarize(self, command):
        # Example: summarize [varlist] [, options]
        main, options = self.parse_command(command)
        parts = main.split()
        if len(parts) > 1:
            varlist = [v.rstrip(',') for v in parts[1:]]
            subset = self._df[varlist]
        else:
            subset = self._df
        # Optionally, handle options here in the future
        return 'Summary statistics\n' + subset.describe().to_string()

    def cmd_tabulate(self, command):
        # Example: tabulate var1 [var2] [, options]
        main, options = self.parse_command(command)
        parts = main.split()
        if len(parts) < 2:
            return 'Usage: tabulate <var1> [var2]'
        var1 = parts[1].rstrip(',')
        if len(parts) > 2:
            var2 = parts[2].rstrip(',')
            result = pd.crosstab(self._df[var1], self._df[var2]).to_string()
            header = f'Tabulation of {var1} and {var2}'
        else:
            result = self._df[var1].value_counts().to_string()
            header = f'Tabulation of {var1}'
        return f'{header}\n{result}'

    def _parse_options(self, options_str):
        """Parse Stata-style options into a dictionary.
        
        Handles both flag options (like 'percent') and options with values (like 'title("My Title")').
        Returns a dictionary where:
        - Flag options are set to True
        - Options with values have their values stored
        - All option names are converted to lowercase
        """
        if not options_str:
            return {}
            
        options = {}
        current = ''
        depth = 0
        in_quotes = False
        current_option = None
        
        for c in options_str:
            if c == '"':
                in_quotes = not in_quotes
                current += c
            elif c == '(' and not in_quotes:
                if depth == 0:
                    # Start of option value
                    current_option = current.strip().lower()
                    current = ''
                depth += 1
            elif c == ')' and not in_quotes:
                depth -= 1
                if depth == 0:
                    # End of option value
                    options[current_option] = current
                    current_option = None
                    current = ''
                else:
                    current += c
            elif c == ' ' and depth == 0 and not in_quotes:
                # End of current option
                if current:
                    if current_option is None:
                        # This is a flag option
                        options[current.lower()] = True
                    else:
                        # This is the value for the current option
                        options[current_option] = current
                    current_option = None
                    current = ''
            else:
                current += c
                    
        # Handle the last option
        if current:
            if current_option is None:
                options[current.lower()] = True
            else:
                options[current_option] = current
                
        return options

    def _split_command_options(self, command):
        """Split a command into (main_part, options_part) at the first comma, ignoring commas inside parentheses or quotes."""
        if ',' not in command:
            return command.strip(), ''
        current = ''
        depth = 0
        in_quotes = False
        for i, c in enumerate(command):
            if c == '"':
                in_quotes = not in_quotes
            elif c == '(' and not in_quotes:
                depth += 1
            elif c == ')' and not in_quotes:
                depth -= 1
            if c == ',' and depth == 0 and not in_quotes:
                # Split here
                return command[:i].strip(), command[i+1:].strip()
            current += c
        return command.strip(), ''

    def _parse_plot_titles(self, options, default_xtitle, default_ytitle, default_title):
        """Parse title options from the options dictionary."""
        xtitle = options.get('xtitle', default_xtitle)
        ytitle = options.get('ytitle', default_ytitle)
        title = options.get('title', default_title)
        
        # Remove quotes from title values
        if isinstance(xtitle, str):
            xtitle = xtitle.strip('"')
        if isinstance(ytitle, str):
            ytitle = ytitle.strip('"')
        if isinstance(title, str):
            title = title.strip('"')
            
        return xtitle, ytitle, title

    def cmd_scatter(self, command):
        main, options_str = self._split_command_options(command)
        tokens = main.split()
        if len(tokens) < 3:
            return 'Usage: scatter <yvar> <xvar> [|| lfitci <yvar> <xvar>] [, xtitle("X Label") ytitle("Y Label") title("My Title")]'
        yvar, xvar = tokens[1].rstrip(','), tokens[2].rstrip(',')
        options = self._parse_options(options_str)
        xtitle, ytitle, title = self._parse_plot_titles(options, xvar, yvar, f'Scatter Plot of {yvar} vs {xvar}')
        
        plt.figure(figsize=(10, 6))
        plt.scatter(self._df[xvar], self._df[yvar], alpha=0.6)
        if '||' in command and 'lfitci' in command:
            X = sm.add_constant(self._df[xvar])
            y = self._df[yvar]
            model = sm.OLS(y, X).fit()
            x_line = np.linspace(self._df[xvar].min(), self._df[xvar].max(), 100)
            X_line = sm.add_constant(x_line)
            y_line = model.predict(X_line)
            ci = model.get_prediction(X_line).conf_int()
            plt.plot(x_line, y_line, 'r-', label='Regression Line')
            plt.fill_between(x_line, ci[:, 0], ci[:, 1], color='r', alpha=0.1, label='95% CI')
            plt.legend()
        plt.xlabel(xtitle)
        plt.ylabel(ytitle)
        plt.title(title)
        plt.grid(True, alpha=0.3)
        show_and_close_figures()
        return f'Scatter plot of {yvar} vs {xvar} displayed.'

    def cmd_histogram(self, command):
        main, options_str = self._split_command_options(command)
        tokens = main.split()
        if len(tokens) < 2:
            return 'Usage: histogram <var> [, percent title("title") xtitle("X") ytitle("Y")]'
        var = tokens[1].rstrip(',')
        options = self._parse_options(options_str)
        xtitle, ytitle, title = self._parse_plot_titles(options, var, 'Frequency', f'Histogram of {var}')
        
        plt.figure(figsize=(10, 6))
        if options.get('percent', False):
            plt.hist(self._df[var], weights=np.ones_like(self._df[var]) * 100. / len(self._df[var]))
            ytitle = ytitle if ytitle != 'Frequency' else 'Percent'
        else:
            plt.hist(self._df[var])
        plt.xlabel(xtitle)
        plt.ylabel(ytitle)
        plt.title(title)
        plt.grid(True, alpha=0.3)
        show_and_close_figures()
        return f'Generated histogram of {var}'

    def cmd_graph_bar(self, command):
        main, options_str = self._split_command_options(command)
        tokens = main.split()
        if len(tokens) < 2:
            return 'Usage: graph bar <var> [, title("My Title") xtitle("X") ytitle("Y")]'
        var = tokens[1].rstrip(',')
        options = self._parse_options(options_str)
        xtitle, ytitle, title = self._parse_plot_titles(options, var, 'Count', f'Bar Chart of {var}')
        
        plt.figure()
        self._df[var].value_counts().plot(kind='bar')
        plt.xlabel(xtitle)
        plt.ylabel(ytitle)
        plt.title(title)
        show_and_close_figures()
        return f'Bar chart of {var} displayed.'

    def cmd_binscatter(self, command):
        main, options_str = self._split_command_options(command)
        tokens = main.split()
        if len(tokens) < 3:
            return 'Usage: binscatter yvar xvar [, bins(n) title("title") xtitle("X") ytitle("Y")]'
        yvar, xvar = tokens[1].rstrip(','), tokens[2].rstrip(',')
        options = self._parse_options(options_str)
        xtitle, ytitle, title = self._parse_plot_titles(options, xvar, yvar, f'Binscatter of {yvar} vs {xvar}')
        
        bins = 20
        if 'bins' in options:
            try:
                bins = int(options['bins'].strip('()'))
            except Exception:
                pass
                
        df = self._df[[xvar, yvar]].dropna()
        df['x_bin'], bin_edges = pd.qcut(df[xvar], q=bins, retbins=True, labels=False, duplicates='drop')
        binned = df.groupby('x_bin').agg({xvar: 'mean', yvar: 'mean'}).reset_index(drop=True)
        
        plt.figure(figsize=(10, 6))
        plt.scatter(binned[xvar], binned[yvar], color='blue', label='Binned Means')
        X = sm.add_constant(df[xvar])
        y = df[yvar]
        model = sm.OLS(y, X).fit()
        x_line = np.linspace(df[xvar].min(), df[xvar].max(), 100)
        X_line = sm.add_constant(x_line)
        y_line = model.predict(X_line)
        plt.plot(x_line, y_line, 'r-', label='Regression Line')
        plt.xlabel(xtitle)
        plt.ylabel(ytitle)
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        show_and_close_figures()
        return f'Generated binscatter of {yvar} vs {xvar}'

    def _check_timeseries_settings(self):
        """Check if time series settings are properly configured"""
        if isinstance(self._df.index, pd.DatetimeIndex):
            return True, self._df.index.name or 'index'
        elif isinstance(self._df.index, pd.MultiIndex):
            if len(self._df.index.names) >= 2:
                return True, self._df.index.names[1]  # Return time variable name
            return False, None
        return False, None

    def _evaluate_math_function(self, func_name, args):
        """Helper method to evaluate mathematical functions"""
        import math
        import numpy as np
        
        # Convert string arguments to numbers
        try:
            args = [float(arg) if isinstance(arg, str) else arg for arg in args]
        except ValueError:
            return None, f"Invalid argument for {func_name}"
            
        # Dictionary of supported functions
        math_functions = {
            'abs': lambda x: abs(x[0]),
            'ceil': lambda x: math.ceil(x[0]),
            'cloglog': lambda x: math.log(-math.log(1 - x[0])),
            'comb': lambda x: math.comb(int(x[0]), int(x[1])),
            'digamma': lambda x: math.digamma(x[0]),
            'exp': lambda x: math.exp(x[0]),
            'expm1': lambda x: math.expm1(x[0]),
            'floor': lambda x: math.floor(x[0]),
            'int': lambda x: int(x[0]),
            'invcloglog': lambda x: 1 - math.exp(-math.exp(x[0])),
            'invlogit': lambda x: 1 / (1 + math.exp(-x[0])),
            'ln': lambda x: math.log(x[0]),
            'ln1m': lambda x: math.log(1 - x[0]),
            'ln1p': lambda x: math.log1p(x[0]),
            'lnfactorial': lambda x: math.lgamma(x[0] + 1),
            'lngamma': lambda x: math.lgamma(x[0]),
            'log': lambda x: math.log(x[0]),
            'log10': lambda x: math.log10(x[0]),
            'log1m': lambda x: math.log(1 - x[0]),
            'log1p': lambda x: math.log1p(x[0]),
            'logit': lambda x: math.log(x[0] / (1 - x[0])),
            'max': lambda x: max(x),
            'min': lambda x: min(x),
            'mod': lambda x: x[0] % x[1],
            'reldif': lambda x: abs(x[0] - x[1]) / (abs(x[1]) + 1),
            'round': lambda x: round(x[0], int(x[1])) if len(x) > 1 else round(x[0]),
            'sign': lambda x: np.sign(x[0]),
            'sqrt': lambda x: math.sqrt(x[0]),
            'sum': lambda x: sum(x),
            'trigamma': lambda x: math.polygamma(1, x[0])
        }
        
        if func_name not in math_functions:
            return None, f"Unknown function: {func_name}"
            
        try:
            result = math_functions[func_name](args)
            return result, None
        except Exception as e:
            return None, f"Error evaluating {func_name}: {str(e)}"

    def cmd_generate(self, command):
        # Example: generate newvar = expression
        parts = command.split('=', 1)
        if len(parts) < 2:
            return 'Usage: generate <newvar> = <expression>'
        newvar = parts[0].strip().split()[-1]
        expr = parts[1].strip()
        if 'L.' in expr:
            # Handle lag operator (only simple cases like L.var)
            lag_match = re.match(r'L(\d*)\.(\w+)', expr)
            if lag_match:
                lag_num = int(lag_match.group(1)) if lag_match.group(1) else 1
                lag_var = lag_match.group(2)
                if lag_var not in self._df.columns:
                    return f'Error: Variable {lag_var} not found.'
                if isinstance(self._df.index, pd.MultiIndex):
                    # Panel data: group by firm_idx (index level)
                    self._df[newvar] = self._df.groupby(level='firm_idx')[lag_var].shift(lag_num)
                elif isinstance(self._df.index, pd.Index) and self._df.index.name == 'year_idx':
                    # Time series: just shift by lag_num
                    self._df[newvar] = self._df[lag_var].shift(lag_num)
                else:
                    return 'Error: Data is not set as panel or time series. Use tsset.'
            else:
                return f'Error: Could not parse lag variable in expression: {expr}'
        else:
            try:
                # Handle mathematical functions
                if '(' in expr and ')' in expr:
                    # Extract function name and argument
                    func_match = re.match(r'(\w+)\(([^)]+)\)', expr)
                    if func_match:
                        func_name = func_match.group(1)
                        arg = func_match.group(2)
                        # Map common math functions to numpy or math functions
                        math_functions = {
                            'log': np.log,
                            'exp': np.exp,
                            'sqrt': np.sqrt,
                            'abs': np.abs,
                            'sin': np.sin,
                            'cos': np.cos,
                            'tan': np.tan,
                            'arcsin': np.arcsin,
                            'arccos': np.arccos,
                            'arctan': np.arctan,
                            'sinh': np.sinh,
                            'cosh': np.cosh,
                            'tanh': np.tanh,
                            'log10': np.log10,
                            'log2': np.log2,
                            'log1p': np.log1p,
                            'expm1': np.expm1,
                            'sign': np.sign,
                            'floor': np.floor,
                            'ceil': np.ceil,
                            'round': np.round
                        }
                        if func_name in math_functions:
                            # Check if argument is a column name
                            if arg.strip() in self._df.columns:
                                # Apply function directly to the column
                                self._df[newvar] = math_functions[func_name](self._df[arg.strip()])
                            else:
                                # Try to evaluate the argument as an expression
                                safe_dict = {
                                    **self._df.to_dict('series'),
                                    "df": self._df,
                                    "np": np,
                                    "math": math,
                                    "pd": pd
                                }
                                arg_value = eval(arg, safe_dict)
                                self._df[newvar] = math_functions[func_name](arg_value)
                        else:
                            return f'Error: Unsupported function {func_name}'
                    else:
                        return f'Error: Could not parse function in expression: {expr}'
                else:
                    # Provide DataFrame columns as variables
                    safe_dict = {
                        **self._df.to_dict('series'),
                        "df": self._df,
                        "np": np,
                        "math": math,
                        "pd": pd
                    }
                    self._df[newvar] = eval(expr, safe_dict)
            except Exception as e:
                return f'Error generating variable: {str(e)}'
        self.update_variables_pane()
        if newvar in self._df.columns:
            return f'Generated new variable: {newvar}'
        else:
            return f'Error: Variable {newvar} was not created.'

    def cmd_replace(self, command):
        # Example: replace var = expression
        parts = command.split('=', 1)
        if len(parts) < 2:
            return 'Usage: replace <var> = <expression>'
        var = parts[0].split()[-1]
        expr = parts[1].strip()
        try:
            self._df[var] = eval(expr, {"df": self._df})
            if var in self._df.columns:
                return f'Replaced values in variable: {var}'
            else:
                return f'Error: Variable {var} was not found after replacement.'
        except Exception as e:
            return f'Error replacing values: {e}'

    def cmd_drop(self, command):
        # Example: drop var1 var2
        parts = command.split()
        if len(parts) < 2:
            return 'Usage: drop <var1> [var2 ...]'
        vars_to_drop = parts[1:]
        self._df.drop(columns=vars_to_drop, inplace=True)
        self.update_variables_pane()
        return f'Dropped variables: {", ".join(vars_to_drop)}'

    def cmd_keep(self, command):
        # Example: keep var1 var2
        parts = command.split()
        if len(parts) < 2:
            return 'Usage: keep <var1> [var2 ...]'
        vars_to_keep = parts[1:]
        self._df = self._df[vars_to_keep]
        self.update_variables_pane()
        return f'Kept variables: {", ".join(vars_to_keep)}'

    def cmd_rename(self, command):
        # Example: rename oldvar newvar
        parts = command.split()
        if len(parts) < 3:
            return 'Usage: rename <oldvar> <newvar>'
        oldvar, newvar = parts[1], parts[2]
        self._df.rename(columns={oldvar: newvar}, inplace=True)
        self.update_variables_pane()
        return f'Renamed {oldvar} to {newvar}'

    def cmd_recode(self, command):
        # Example: recode var (1=2) (2=1)
        parts = command.split()
        if len(parts) < 3:
            return 'Usage: recode <var> (<old1>=<new1>) [(<old2>=<new2>) ...]'
        var = parts[1]
        recodes = parts[2:]
        for recode in recodes:
            old, new = recode.strip('()').split('=')
            self._df[var] = self._df[var].replace(int(old), int(new))
        return f'Recoded variable: {var}'

    def cmd_clonevar(self, command):
        # Example: clonevar newvar = oldvar
        parts = command.split('=', 1)
        if len(parts) < 2:
            return 'Usage: clonevar <newvar> = <oldvar>'
        newvar = parts[0].split()[-1]
        oldvar = parts[1].strip()
        self._df[newvar] = self._df[oldvar]
        self.update_variables_pane()
        return f'Cloned variable {oldvar} to {newvar}'

    def cmd_label_variable(self, command):
        # Example: label variable var "label"
        parts = command.split('"', 1)
        if len(parts) < 2:
            return 'Usage: label variable <var> "label"'
        var = parts[0].split()[-1]
        label = parts[1].strip('"')
        self._df[var].name = label
        return f'Labeled variable {var} as "{label}"'

    def cmd_label_values(self, command):
        # Example: label values var valuelist
        parts = command.split()
        if len(parts) < 3:
            return 'Usage: label values <var> <valuelist>'
        var = parts[1]
        valuelist = parts[2]
        # For simplicity, assume valuelist is a dictionary-like string
        try:
            value_dict = eval(valuelist)
            self._df[var] = self._df[var].map(value_dict)
            return f'Labeled values for variable {var}'
        except Exception as e:
            return f'Error labeling values: {e}'

    def cmd_append(self, command):
        # Example: append using filename
        parts = command.split()
        if len(parts) < 3 or parts[1] != 'using':
            return 'Usage: append using <filename>'
        filename = parts[2]
        try:
            if filename.endswith('.csv'):
                new_df = pd.read_csv(filename)
            elif filename.endswith('.xlsx'):
                new_df = pd.read_excel(filename)
            else:
                return 'Unsupported file type.'
            self._df = pd.concat([self._df, new_df], ignore_index=True)
            self.update_variables_pane()
            return f'Appended dataset from {filename}'
        except Exception as e:
            return f'Error appending dataset: {e}'

    def cmd_merge(self, command):
        # Example: merge 1:1 keyvar using filename
        parts = command.split()
        if len(parts) < 4 or parts[0] != 'merge' or parts[3] != 'using':
            return 'Usage: merge <1:1|m:1|1:m|m:m> <keyvar> using <filename>'
        merge_type = parts[1]
        keyvar = parts[2]
        filename = parts[4]
        try:
            if filename.endswith('.csv'):
                new_df = pd.read_csv(filename)
            elif filename.endswith('.xlsx'):
                new_df = pd.read_excel(filename)
            else:
                return 'Unsupported file type.'
            self._df = pd.merge(self._df, new_df, on=keyvar, how='outer')
            self.update_variables_pane()
            return f'Merged dataset from {filename}'
        except Exception as e:
            return f'Error merging dataset: {e}'

    def cmd_joinby(self, command):
        # Example: joinby keyvar using filename
        parts = command.split()
        if len(parts) < 3 or parts[1] != 'using':
            return 'Usage: joinby <keyvar> using <filename>'
        keyvar = parts[0].split()[-1]
        filename = parts[2]
        try:
            if filename.endswith('.csv'):
                new_df = pd.read_csv(filename)
            elif filename.endswith('.xlsx'):
                new_df = pd.read_excel(filename)
            else:
                return 'Unsupported file type.'
            self._df = pd.merge(self._df, new_df, on=keyvar, how='outer')
            self.update_variables_pane()
            return f'Joined dataset from {filename}'
        except Exception as e:
            return f'Error joining dataset: {e}'

    def cmd_cross(self, command):
        # Example: cross using filename
        parts = command.split()
        if len(parts) < 3 or parts[1] != 'using':
            return 'Usage: cross using <filename>'
        filename = parts[2]
        try:
            if filename.endswith('.csv'):
                new_df = pd.read_csv(filename)
            elif filename.endswith('.xlsx'):
                new_df = pd.read_excel(filename)
            else:
                return 'Unsupported file type.'
            self._df = pd.merge(self._df, new_df, how='cross')
            self.update_variables_pane()
            return f'Crossed dataset with {filename}'
        except Exception as e:
            return f'Error crossing dataset: {e}'

    def cmd_regress(self, command):
        # Example: regress depvar indepvar1 indepvar2 [, options]
        parts = command.split(',')
        cmd_parts = parts[0].split()
        if len(cmd_parts) < 3:
            return 'Usage: regress <depvar> <indepvar1> [indepvar2 ...]'
        depvar = cmd_parts[1]
        indepvars = cmd_parts[2:]
        X = self._df[indepvars]
        y = self._df[depvar]
        model = sm.OLS(y, sm.add_constant(X)).fit()
        # Store results for coefplot and esttab
        self._stored_estimates.store('lastreg', model, 'ols', depvar, indepvars)
        self._lastreg = model
        self._stored_estimates.current_name = 'lastreg'
        model_name = f'OLS_model_{len(self.command_history)}'
        setattr(self, f'_{model_name}', model)
        
        # Format output in Stata style
        output = []
        output.append(f"Linear regression                                      Number of obs = {len(self._df)}")
        output.append(f"                                                      F({model.df_model}, {model.df_resid}) = {model.fvalue:.2f}")
        output.append(f"                                                      Prob > F      = {model.f_pvalue:.4f}")
        output.append(f"                                                      R-squared     = {model.rsquared:.4f}")
        output.append(f"                                                      Adj R-squared = {model.rsquared_adj:.4f}")
        output.append(f"                                                      Root MSE      = {np.sqrt(model.mse_resid):.4f}")
        output.append("")
        output.append("------------------------------------------------------------------------------")
        output.append(f"{depvar:12} |      Coef.   Std. Err.      t    P>|t|     [95% Conf. Interval]")
        output.append("-------------+----------------------------------------------------------------")
        
        # Add coefficients
        for name, coef in model.params.items():
            std_err = model.bse[name]
            t_stat = model.tvalues[name]
            p_value = model.pvalues[name]
            ci = model.conf_int().loc[name]
            ci_lower, ci_upper = ci[0], ci[1]
            
            # Format variable name
            if name == 'const':
                var_name = '_cons'
            else:
                var_name = name
            
            output.append(f"{var_name:12} | {coef:10.4f} {std_err:10.4f} {t_stat:7.2f} {p_value:7.4f} {ci_lower:10.4f} {ci_upper:10.4f}")
        
        output.append("------------------------------------------------------------------------------")
        
        return "\n".join(output)

    def cmd_anova(self, command):
        # Example: anova depvar indepvar1 indepvar2
        parts = command.split()
        if len(parts) < 3:
            return 'Usage: anova <depvar> <indepvar1> [indepvar2 ...]'
        depvar = parts[1]
        indepvars = parts[2:]
        X = self._df[indepvars]
        y = self._df[depvar]
        model = sm.OLS(y, sm.add_constant(X)).fit()
        return model.summary().as_text()

    def cmd_areg(self, command):
        # Example: areg depvar indepvar1 indepvar2, absorb(groupvar)
        parts = command.split(',')
        if len(parts) < 2:
            return 'Usage: areg <depvar> <indepvar1> [indepvar2 ...], absorb(<groupvar>)'
        cmd_parts = parts[0].split()
        if len(cmd_parts) < 3:
            return 'Usage: areg <depvar> <indepvar1> [indepvar2 ...], absorb(<groupvar>)'
        depvar = cmd_parts[1]
        indepvars = cmd_parts[2:]
        absorb_part = parts[1].strip()
        if not absorb_part.startswith('absorb(') or not absorb_part.endswith(')'):
            return 'Usage: areg <depvar> <indepvar1> [indepvar2 ...], absorb(<groupvar>)'
        groupvar = absorb_part[7:-1]
        X = self._df[indepvars]
        y = self._df[depvar]
        groups = self._df[groupvar]
        model = sm.OLS(y, sm.add_constant(X)).fit()
        return model.summary().as_text()

    def cmd_xtreg(self, command):
        # Example: xtreg depvar indepvar1 indepvar2, fe
        parts = command.split(',')
        if len(parts) < 2:
            return 'Usage: xtreg <depvar> <indepvar1> [indepvar2 ...], fe'
        cmd_parts = parts[0].split()
        if len(cmd_parts) < 3:
            return 'Usage: xtreg <depvar> <indepvar1> [indepvar2 ...], fe'
        depvar = cmd_parts[1]
        indepvars = cmd_parts[2:]
        X = self._df[indepvars]
        y = self._df[depvar]
        model = sm.OLS(y, sm.add_constant(X)).fit()
        return model.summary().as_text()

    def cmd_fe(self, command):
        # Example: fe depvar indepvar1 indepvar2
        parts = command.split()
        if len(parts) < 3:
            return 'Usage: fe <depvar> <indepvar1> [indepvar2 ...]'
        depvar = parts[1]
        indepvars = parts[2:]
        X = self._df[indepvars]
        y = self._df[depvar]
        model = sm.OLS(y, sm.add_constant(X)).fit()
        return model.summary().as_text()

    def cmd_logit(self, command):
        # Example: logit depvar indepvar1 indepvar2
        parts = command.split()
        if len(parts) < 3:
            return 'Usage: logit <depvar> <indepvar1> [indepvar2 ...]'
        depvar = parts[1]
        indepvars = parts[2:]
        X = self._df[indepvars]
        y = self._df[depvar]
        model = sm.Logit(y, sm.add_constant(X)).fit()
        return model.summary().as_text()

    def cmd_probit(self, command):
        # Example: probit depvar indepvar1 indepvar2
        parts = command.split()
        if len(parts) < 3:
            return 'Usage: probit <depvar> <indepvar1> [indepvar2 ...]'
        depvar = parts[1]
        indepvars = parts[2:]
        X = self._df[indepvars]
        y = self._df[depvar]
        model = sm.Probit(y, sm.add_constant(X)).fit()
        return model.summary().as_text()

    def cmd_logistic(self, command):
        # Example: logistic depvar indepvar1 indepvar2
        parts = command.split()
        if len(parts) < 3:
            return 'Usage: logistic <depvar> <indepvar1> [indepvar2 ...]'
        depvar = parts[1]
        indepvars = parts[2:]
        X = self._df[indepvars]
        y = self._df[depvar]
        model = sm.Logit(y, sm.add_constant(X)).fit()
        return model.summary().as_text()

    def cmd_poisson(self, command):
        # Example: poisson depvar indepvar1 indepvar2
        parts = command.split()
        if len(parts) < 3:
            return 'Usage: poisson <depvar> <indepvar1> [indepvar2 ...]'
        depvar = parts[1]
        indepvars = parts[2:]
        X = self._df[indepvars]
        y = self._df[depvar]
        model = sm.Poisson(y, sm.add_constant(X)).fit()
        return model.summary().as_text()

    def cmd_nbreg(self, command):
        # Example: nbreg depvar indepvar1 indepvar2
        parts = command.split()
        if len(parts) < 3:
            return 'Usage: nbreg <depvar> <indepvar1> [indepvar2 ...]'
        depvar = parts[1]
        indepvars = parts[2:]
        X = self._df[indepvars]
        y = self._df[depvar]
        model = sm.NegativeBinomial(y, sm.add_constant(X)).fit()
        return model.summary().as_text()

    def cmd_tobit(self, command):
        # Example: tobit depvar indepvar1 indepvar2
        parts = command.split()
        if len(parts) < 3:
            return 'Usage: tobit <depvar> <indepvar1> [indepvar2 ...]'
        depvar = parts[1]
        indepvars = parts[2:]
        X = self._df[indepvars]
        y = self._df[depvar]
        model = sm.Tobit(y, sm.add_constant(X)).fit()
        return model.summary().as_text()

    def cmd_intreg(self, command):
        # Example: intreg depvar indepvar1 indepvar2
        parts = command.split()
        if len(parts) < 3:
            return 'Usage: intreg <depvar> <indepvar1> [indepvar2 ...]'
        depvar = parts[1]
        indepvars = parts[2:]
        X = self._df[indepvars]
        y = self._df[depvar]
        model = sm.OLS(y, sm.add_constant(X)).fit()
        return model.summary().as_text()

    def cmd_ivregress(self, command):
        # Example: ivregress 2sls depvar (endogvar = instrument) exogvar1 exogvar2
        pattern = r'ivregress\s+(\w+)\s+(\w+)\s+\((\w+)\s*=\s*(\w+)\)(?:\s+(\w+))?'
        match = re.match(pattern, command)
        
        if not match:
            return "Usage: ivregress 2sls <depvar> (<endogvar> = <instrument>) <exogvar1> [exogvar2 ...]"
        
        method, dep_var, endog_var, instrument, exog_var = match.groups()
        
        if method != '2sls':
            return f"Error: Only 2SLS method is supported, got {method}"
        
        try:
            # Prepare the data
            df = self._df.copy()
            y = df[dep_var]
            Z = df[endog_var].values.reshape(-1, 1)  # Endogenous variable
            W = df[instrument].values.reshape(-1, 1)  # Instrument
            
            # Handle exogenous variables if present
            if exog_var:
                X = df[exog_var].values.reshape(-1, 1)
            else:
                X = np.zeros((len(df), 0))  # Empty array for no exogenous variables
            
            # First stage regression
            first_stage = sm.OLS(Z, sm.add_constant(W)).fit()
            
            # Second stage regression
            Z_hat = first_stage.predict(sm.add_constant(W))
            if exog_var:
                X_full = np.column_stack([X, Z_hat])
            else:
                X_full = Z_hat
            
            # Fit the model
            model = sm.OLS(y, X_full)
            results = model.fit()
            
            # Format output to look like Stata
            output = []
            output.append("Instrumental variables (2SLS) regression")
            output.append(f"Number of obs = {len(df)}")
            output.append(f"F({results.df_model}, {results.df_resid}) = {results.fvalue:.2f}")
            output.append(f"Prob > F      = {results.f_pvalue:.4f}")
            output.append(f"R-squared     = {results.rsquared:.4f}")
            output.append(f"Adj R-squared = {results.rsquared_adj:.4f}")
            output.append(f"Root MSE      = {np.sqrt(results.mse_resid):.4f}")
            output.append("")
            output.append("------------------------------------------------------------------------------")
            output.append(f"             |               Robust")
            output.append(f"{dep_var:12} |      Coef.   Std. Err.      t    P>|t|     [95% Conf. Interval]")
            output.append("-------------+----------------------------------------------------------------")
            
            # Add coefficients
            if exog_var:
                for i, name in enumerate([exog_var, endog_var]):
                    coef = results.params.iloc[i]
                    std_err = results.bse.iloc[i]
                    t_stat = results.tvalues.iloc[i]
                    p_value = results.pvalues.iloc[i]
                    ci = results.conf_int().iloc[i]
                    ci_lower, ci_upper = ci[0], ci[1]
                    
                    output.append(f"{name:12} | {coef:10.4f} {std_err:10.4f} {t_stat:7.2f} {p_value:7.4f} {ci_lower:10.4f} {ci_upper:10.4f}")
            else:
                coef = results.params.iloc[0]
                std_err = results.bse.iloc[0]
                t_stat = results.tvalues.iloc[0]
                p_value = results.pvalues.iloc[0]
                ci = results.conf_int().iloc[0]
                ci_lower, ci_upper = ci[0], ci[1]
                
                output.append(f"{endog_var:12} | {coef:10.4f} {std_err:10.4f} {t_stat:7.2f} {p_value:7.4f} {ci_lower:10.4f} {ci_upper:10.4f}")
            
            output.append("------------------------------------------------------------------------------")
            output.append("")
            output.append("Instrumented: " + endog_var)
            output.append("Instruments:  " + instrument)
            
            # Store the model for later use
            self._stored_estimates.store('lastreg', results, 'iv', dep_var, [endog_var, exog_var] if exog_var else [endog_var])
            self._lastreg = results
            self._stored_estimates.current_name = 'lastreg'
            
            return "\n".join(output)
            
        except Exception as e:
            return f"Error: {str(e)}"

    def cmd_ivreg2(self, command):
        # Example: ivreg2 depvar (endogvar = instrument) exogvar1 exogvar2
        return self.cmd_ivregress(command.replace('ivreg2', 'ivregress 2sls', 1))

    def cmd_control_function(self, command):
        # Example: control_function depvar endogvar instrument exogvar1 exogvar2
        parts = command.split()
        if len(parts) < 5:
            return 'Usage: control_function <depvar> <endogvar> <instrument> <exogvar1> [exogvar2 ...]'
        depvar = parts[1]
        endogvar = parts[2]
        instrument = parts[3]
        exogvars = parts[4:]
        X = self._df[exogvars]
        y = self._df[depvar]
        Z = self._df[endogvar]
        W = self._df[instrument]
        model = sm.IV2SLS(y, sm.add_constant(X), Z, W).fit()
        return model.summary().as_text()

    def cmd_xtset(self, command):
        # Example: xtset panelvar timevar
        parts = command.split()
        if len(parts) < 3:
            # Show current panel settings
            if isinstance(self._df.index, pd.MultiIndex):
                panel_var, time_var = self._df.index.names
                return f"Panel variable: {panel_var}\nTime variable: {time_var}"
            else:
                return "No panel settings"
                
        panelvar = parts[1]
        timevar = parts[2]
        
        # Check if variables exist
        if panelvar not in self._df.columns:
            return f"Error: Panel variable '{panelvar}' not found"
        if timevar not in self._df.columns:
            return f"Error: Time variable '{timevar}' not found"
            
        # Sort by panel and time variables
        self._df.sort_values([panelvar, timevar], inplace=True)
        
        # Set the index
        self._df.set_index([panelvar, timevar], inplace=True)
        return f'Panel data set with panel variable {panelvar} and time variable {timevar}'

    def cmd_xtlogit(self, command):
        # Example: xtlogit depvar indepvar1 indepvar2, fe
        parts = command.split(',')
        if len(parts) < 2:
            return 'Usage: xtlogit <depvar> <indepvar1> [indepvar2 ...], fe'
        cmd_parts = parts[0].split()
        if len(cmd_parts) < 3:
            return 'Usage: xtlogit <depvar> <indepvar1> [indepvar2 ...], fe'
        depvar = cmd_parts[1]
        indepvars = cmd_parts[2:]
        X = self._df[indepvars]
        y = self._df[depvar]
        model = sm.Logit(y, sm.add_constant(X)).fit()
        return model.summary().as_text()

    def cmd_xtprobit(self, command):
        # Example: xtprobit depvar indepvar1 indepvar2, fe
        parts = command.split(',')
        if len(parts) < 2:
            return 'Usage: xtprobit <depvar> <indepvar1> [indepvar2 ...], fe'
        cmd_parts = parts[0].split()
        if len(cmd_parts) < 3:
            return 'Usage: xtprobit <depvar> <indepvar1> [indepvar2 ...], fe'
        depvar = cmd_parts[1]
        indepvars = cmd_parts[2:]
        X = self._df[indepvars]
        y = self._df[depvar]
        model = sm.Probit(y, sm.add_constant(X)).fit()
        return model.summary().as_text()

    def cmd_xtpoisson(self, command):
        # Example: xtpoisson depvar indepvar1 indepvar2, fe
        parts = command.split(',')
        if len(parts) < 2:
            return 'Usage: xtpoisson <depvar> <indepvar1> [indepvar2 ...], fe'
        cmd_parts = parts[0].split()
        if len(cmd_parts) < 3:
            return 'Usage: xtpoisson <depvar> <indepvar1> [indepvar2 ...], fe'
        depvar = cmd_parts[1]
        indepvars = cmd_parts[2:]
        X = self._df[indepvars]
        y = self._df[depvar]
        model = sm.Poisson(y, sm.add_constant(X)).fit()
        return model.summary().as_text()

    def cmd_xtabond(self, command):
        # Example: xtabond depvar indepvar1 indepvar2, lags(1)
        parts = command.split(',')
        if len(parts) < 2:
            return 'Usage: xtabond <depvar> <indepvar1> [indepvar2 ...], lags(<n>)'
        cmd_parts = parts[0].split()
        if len(cmd_parts) < 3:
            return 'Usage: xtabond <depvar> <indepvar1> [indepvar2 ...], lags(<n>)'
        depvar = cmd_parts[1]
        indepvars = cmd_parts[2:]
        X = self._df[indepvars]
        y = self._df[depvar]
        model = sm.OLS(y, sm.add_constant(X)).fit()
        return model.summary().as_text()

    def cmd_xtdpdsys(self, command):
        # Example: xtdpdsys depvar indepvar1 indepvar2, lags(1)
        parts = command.split(',')
        if len(parts) < 2:
            return 'Usage: xtdpdsys <depvar> <indepvar1> [indepvar2 ...], lags(<n>)'
        cmd_parts = parts[0].split()
        if len(cmd_parts) < 3:
            return 'Usage: xtdpdsys <depvar> <indepvar1> [indepvar2 ...], lags(<n>)'
        depvar = cmd_parts[1]
        indepvars = cmd_parts[2:]
        X = self._df[indepvars]
        y = self._df[depvar]
        model = sm.OLS(y, sm.add_constant(X)).fit()
        return model.summary().as_text()

    def cmd_tsset(self, command):
        # Example: tsset timevar
        # Example: tsset panelvar timevar
        parts = command.split()
        if len(parts) < 2:
            # Show current time series settings
            if isinstance(self._df.index, pd.DatetimeIndex):
                return f"Time variable: {self._df.index.name or 'index'}\nPanel variable: not set"
            elif isinstance(self._df.index, pd.MultiIndex):
                panel_var, time_var = self._df.index.names
                return f"Panel variable: {panel_var}\nTime variable: {time_var}"
            else:
                return "No time series settings"
        
        # Reset index first to avoid ambiguity
        self._df.reset_index(inplace=True)
        if 'index' in self._df.columns:
            self._df.drop('index', axis=1, inplace=True)
        
        # Handle panel data structure
        if len(parts) == 3:
            panelvar = parts[1]
            timevar = parts[2]
            
            # Check if variables exist
            if panelvar not in self._df.columns:
                return f"Error: Panel variable '{panelvar}' not found"
            if timevar not in self._df.columns:
                return f"Error: Time variable '{timevar}' not found"
            
            # Sort by panel and time variables
            self._df.sort_values([panelvar, timevar], inplace=True)
            
            # Create index copies
            self._df['firm_idx'] = self._df[panelvar]
            self._df['year_idx'] = self._df[timevar]
            self._df.set_index(['firm_idx', 'year_idx'], inplace=True)
            
            return f'Panel data set with panel variable {panelvar} and time variable {timevar}'
        
        # Handle single time series
        else:
            timevar = parts[1]
            
            # Check if variable exists
            if timevar not in self._df.columns:
                return f"Error: Time variable '{timevar}' not found"
            
            # Sort by time variable
            self._df.sort_values(timevar, inplace=True)
            
            # Create index copy
            self._df['year_idx'] = self._df[timevar]
            self._df.set_index('year_idx', inplace=True)
            
            return f'Time series data set with time variable {timevar}'

    def cmd_arima(self, command):
        # Example: arima depvar, arima(p,d,q)
        parts = command.split(',')
        if len(parts) < 2:
            return 'Usage: arima <depvar>, arima(p,d,q)'
        cmd_parts = parts[0].split()
        if len(cmd_parts) < 2:
            return 'Usage: arima <depvar>, arima(p,d,q)'
        depvar = cmd_parts[1]
        arima_part = parts[1].strip()
        if not arima_part.startswith('arima(') or not arima_part.endswith(')'):
            return 'Usage: arima <depvar>, arima(p,d,q)'
        p, d, q = map(int, arima_part[6:-1].split(','))
        y = self._df[depvar]
        model = sm.tsa.ARIMA(y, order=(p,d,q)).fit()
        return model.summary().as_text()

    def cmd_arch(self, command):
        # Example: arch depvar, arch(1) garch(1)
        parts = command.split(',')
        if len(parts) < 2:
            return 'Usage: arch <depvar>, arch(p) garch(q)'
        cmd_parts = parts[0].split()
        if len(cmd_parts) < 2:
            return 'Usage: arch <depvar>, arch(p) garch(q)'
        depvar = cmd_parts[1]
        arch_part = parts[1].strip()
        garch_part = parts[2].strip() if len(parts) > 2 else 'garch(0)'
        if not arch_part.startswith('arch(') or not arch_part.endswith(')'):
            return 'Usage: arch <depvar>, arch(p) garch(q)'
        if not garch_part.startswith('garch(') or not garch_part.endswith(')'):
            return 'Usage: arch <depvar>, arch(p) garch(q)'
        p = int(arch_part[5:-1])
        q = int(garch_part[6:-1])
        y = self._df[depvar]
        model = sm.tsa.GARCH(y, p=p, q=q).fit()
        return model.summary().as_text()

    def cmd_var(self, command):
        # Example: var depvar1 depvar2, lags(2)
        parts = command.split(',')
        if len(parts) < 2:
            return 'Usage: var <depvar1> [depvar2 ...], lags(n)'
        cmd_parts = parts[0].split()
        if len(cmd_parts) < 2:
            return 'Usage: var <depvar1> [depvar2 ...], lags(n)'
        depvars = cmd_parts[1:]
        lags_part = parts[1].strip()
        if not lags_part.startswith('lags(') or not lags_part.endswith(')'):
            return 'Usage: var <depvar1> [depvar2 ...], lags(n)'
        n = int(lags_part[5:-1])
        y = self._df[depvars]
        model = sm.tsa.VAR(y).fit(maxlags=n)
        return model.summary().as_text()

    def cmd_vec(self, command):
        # Example: vec depvar1 depvar2, lags(2)
        parts = command.split(',')
        if len(parts) < 2:
            return 'Usage: vec <depvar1> [depvar2 ...], lags(n)'
        cmd_parts = parts[0].split()
        if len(cmd_parts) < 2:
            return 'Usage: vec <depvar1> [depvar2 ...], lags(n)'
        depvars = cmd_parts[1:]
        lags_part = parts[1].strip()
        if not lags_part.startswith('lags(') or not lags_part.endswith(')'):
            return 'Usage: vec <depvar1> [depvar2 ...], lags(n)'
        n = int(lags_part[5:-1])
        y = self._df[depvars]
        model = sm.tsa.VECM(y, k_ar_diff=n).fit()
        return model.summary().as_text()

    def cmd_newey(self, command):
        # Example: newey depvar indepvar1 indepvar2, lag(2)
        parts = command.split(',')
        if len(parts) < 2:
            return 'Usage: newey <depvar> <indepvar1> [indepvar2 ...], lag(n)'
        cmd_parts = parts[0].split()
        if len(cmd_parts) < 3:
            return 'Usage: newey <depvar> <indepvar1> [indepvar2 ...], lag(n)'
        depvar = cmd_parts[1]
        indepvars = cmd_parts[2:]
        lag_part = parts[1].strip()
        if not lag_part.startswith('lag(') or not lag_part.endswith(')'):
            return 'Usage: newey <depvar> <indepvar1> [indepvar2 ...], lag(n)'
        n = int(lag_part[4:-1])
        X = self._df[indepvars]
        y = self._df[depvar]
        model = sm.OLS(y, sm.add_constant(X))
        results = model.fit(cov_type='HAC', cov_kwds={'maxlags': n})
        return results.summary().as_text()

    def cmd_list(self, command):
        # Example: list [varlist]
        parts = command.split()
        if len(parts) > 1:
            varlist = parts[1:]
            subset = self._df[varlist]
        else:
            subset = self._df
        return subset.to_string()

    def cmd_collapse(self, command):
        # Example: collapse (mean) var1 (sum) var2, by(groupvar)
        parts = command.split(',')
        if len(parts) < 2:
            return 'Usage: collapse (stat) var1 [(stat) var2 ...], by(groupvar)'
        cmd_parts = parts[0].split()
        if len(cmd_parts) < 2:
            return 'Usage: collapse (stat) var1 [(stat) var2 ...], by(groupvar)'
        by_part = parts[1].strip()
        if not by_part.startswith('by(') or not by_part.endswith(')'):
            return 'Usage: collapse (stat) var1 [(stat) var2 ...], by(groupvar)'
        groupvar = by_part[3:-1]
        
        # Parse statistics and variables
        stats_vars = []
        current_stat = None
        for part in cmd_parts[1:]:
            if part.startswith('(') and part.endswith(')'):
                current_stat = part[1:-1]
            elif current_stat:
                stats_vars.append((current_stat, part))
        
        # Perform aggregation
        agg_dict = {}
        for stat, var in stats_vars:
            if stat == 'mean':
                agg_dict[var] = 'mean'
            elif stat == 'sum':
                agg_dict[var] = 'sum'
            elif stat == 'count':
                agg_dict[var] = 'count'
            elif stat == 'min':
                agg_dict[var] = 'min'
            elif stat == 'max':
                agg_dict[var] = 'max'
        
        result = self._df.groupby(groupvar).agg(agg_dict)
        return result.to_string()

    def cmd_reshape(self, command):
        # Example: reshape wide var, i(id) j(time)
        parts = command.split(',')
        if len(parts) < 3:
            return 'Usage: reshape wide|long var, i(id) j(time)'
        cmd_parts = parts[0].split()
        if len(cmd_parts) < 3:
            return 'Usage: reshape wide|long var, i(id) j(time)'
        direction = cmd_parts[1]
        var = cmd_parts[2]
        
        i_part = parts[1].strip()
        j_part = parts[2].strip()
        if not i_part.startswith('i(') or not i_part.endswith(')'):
            return 'Usage: reshape wide|long var, i(id) j(time)'
        if not j_part.startswith('j(') or not j_part.endswith(')'):
            return 'Usage: reshape wide|long var, i(id) j(time)'
        
        id_var = i_part[2:-1]
        time_var = j_part[2:-1]
        
        if direction == 'wide':
            self._df = self._df.pivot(index=id_var, columns=time_var, values=var)
        elif direction == 'long':
            self._df = self._df.melt(id_vars=[id_var], value_vars=[var], var_name=time_var)
        else:
            return 'Direction must be "wide" or "long"'
        
        self.update_variables_pane()
        return f'Data reshaped to {direction} format'

    def cmd_sort(self, command):
        # Example: sort var1 var2
        parts = command.split()
        if len(parts) < 2:
            return 'Usage: sort <var1> [var2 ...]'
        vars_to_sort = parts[1:]
        self._df.sort_values(by=vars_to_sort, inplace=True)
        return f'Data sorted by {", ".join(vars_to_sort)}'

    def cmd_edit(self, command):
        # Example: edit [varlist]
        parts = command.split()
        if len(parts) > 1:
            varlist = parts[1:]
            subset = self._df[varlist]
        else:
            subset = self._df
        browser = DataBrowser(subset, self)
        browser.show()
        return 'Data editor opened'

    def cmd_do(self, command):
        """Execute a .do file"""
        parts = command.split()
        if len(parts) < 2:
            return 'Usage: do <filename>'
        
        filename = parts[1]
        if not filename.endswith('.do'):
            filename += '.do'
            
        if not os.path.exists(filename):
            return f'Error: File {filename} not found'
            
        try:
            with open(filename, 'r') as f:
                commands = f.readlines()
                
            output = []
            for cmd in commands:
                cmd = cmd.strip()
                if not cmd or cmd.startswith('*') or cmd.startswith('//'):
                    continue
                    
                # Execute the command
                if cmd.startswith('!'):
                    result = self.cmd_bash(cmd)
                elif cmd.split()[0].lower() in self.stata_commands:
                    handler = self.command_registry.get(cmd.split()[0].lower())
                    if handler:
                        result = handler(cmd)
                    else:
                        result = f'Unknown command: {cmd}'
                else:
                    # Execute as Python code
                    local_namespace = {
                        '_df': self._df,
                        'pd': pd,
                        'np': np,
                        'sm': sm,
                        'plt': plt
                    }
                    output_buffer = io.StringIO()
                    with contextlib.redirect_stdout(output_buffer), contextlib.redirect_stderr(output_buffer):
                        try:
                            try:
                                result = eval(cmd, local_namespace)
                                if result is not None:
                                    print(result)
                            except:
                                exec(cmd, local_namespace)
                            result = output_buffer.getvalue()
                        except Exception as e:
                            result = f'Error: {e}'
                            
                output.append(f'{cmd}\n{result}\n')
                
            return '\n'.join(output)
            
        except Exception as e:
            return f'Error executing do file: {e}'

    def cmd_egen(self, command):
        # Example: egen newvar = mean(var), by(groupvar)
        pattern = r'egen\s+(\w+)\s*=\s*(\w+)\s*\(([^)]+)\)(?:\s*,\s*by\s*\(([^)]+)\))?'
        match = re.match(pattern, command)
        
        if not match:
            return "Usage: egen newvar = function(varlist) [, by(varlist)]"
        
        newvar, func, varlist, by_vars = match.groups()
        
        try:
            # Convert varlist and by_vars to lists
            varlist = [var.strip() for var in varlist.split()]
            if by_vars:
                by_vars = [var.strip() for var in by_vars.split()]
            
            # Get the data
            df = self._df.copy()
            
            # Ensure all variables are numeric
            for var in varlist:
                df[var] = pd.to_numeric(df[var], errors='coerce')
            
            # Apply the function
            if func == 'mean':
                if by_vars:
                    # Group by the specified variables and calculate mean
                    result = df.groupby(by_vars)[varlist[0]].mean().reset_index()
                    # Merge back to original dataframe
                    df = df.merge(result, on=by_vars, how='left', suffixes=('', '_y'))
                    df[newvar] = df[varlist[0] + '_y']
                    df = df.drop(columns=[varlist[0] + '_y'])
                else:
                    # Calculate mean for each variable
                    df[newvar] = df[varlist[0]].mean()
            elif func == 'sum':
                if by_vars:
                    result = df.groupby(by_vars)[varlist[0]].sum().reset_index()
                    df = df.merge(result, on=by_vars, how='left', suffixes=('', '_y'))
                    df[newvar] = df[varlist[0] + '_y']
                    df = df.drop(columns=[varlist[0] + '_y'])
                else:
                    df[newvar] = df[varlist[0]].sum()
            elif func == 'min':
                if by_vars:
                    result = df.groupby(by_vars)[varlist[0]].min().reset_index()
                    df = df.merge(result, on=by_vars, how='left', suffixes=('', '_y'))
                    df[newvar] = df[varlist[0] + '_y']
                    df = df.drop(columns=[varlist[0] + '_y'])
                else:
                    df[newvar] = df[varlist[0]].min()
            elif func == 'max':
                if by_vars:
                    result = df.groupby(by_vars)[varlist[0]].max().reset_index()
                    df = df.merge(result, on=by_vars, how='left', suffixes=('', '_y'))
                    df[newvar] = df[varlist[0] + '_y']
                    df = df.drop(columns=[varlist[0] + '_y'])
                else:
                    df[newvar] = df[varlist[0]].max()
            elif func == 'sd':
                if by_vars:
                    result = df.groupby(by_vars)[varlist[0]].std().reset_index()
                    df = df.merge(result, on=by_vars, how='left', suffixes=('', '_y'))
                    df[newvar] = df[varlist[0] + '_y']
                    df = df.drop(columns=[varlist[0] + '_y'])
                else:
                    df[newvar] = df[varlist[0]].std()
            else:
                return f"Error: Unsupported function {func}"
            
            # Update the dataframe and variables pane
            self._df = df
            self.update_variables_pane()
            
            # Return success message
            return f"Generated new variable: {newvar}"
                
        except Exception as e:
            return f"Error in egen command: {str(e)}"

    def cmd_estout(self, command):
        """Handle estout command for displaying stored estimates"""
        # Parse command
        parts = command.split()
        if len(parts) < 2:
            return "Error: estout requires at least one estimate name"
            
        # Get estimate names and options
        estimate_names = []
        options = {}
        for part in parts[1:]:
            if part.startswith(','):
                # Parse options
                opt_parts = part[1:].split('=')
                if len(opt_parts) == 2:
                    options[opt_parts[0]] = opt_parts[1]
            else:
                estimate_names.append(part)
                
        # Get stored estimates
        estimates = []
        for name in estimate_names:
            est = self._stored_estimates.get(name)
            if est:
                estimates.append(est)
            else:
                return f"Error: Estimate '{name}' not found"
                
        # Format output based on options
        output = []
        if 'using' in options:
            # Save to file
            filename = options['using']
            if filename.endswith('.csv'):
                # Save as CSV
                results_df = pd.DataFrame()
                for est in estimates:
                    results_df = pd.concat([results_df, est['model'].summary().tables[1]], axis=1)
                results_df.to_csv(filename)
                return f"Results saved to {filename}"
            elif filename.endswith('.tex'):
                # Save as LaTeX
                results_df = pd.DataFrame()
                for est in estimates:
                    results_df = pd.concat([results_df, est['model'].summary().tables[1]], axis=1)
                results_df.to_latex(filename)
                return f"Results saved to {filename}"
            elif filename.endswith('.html'):
                # Save as HTML
                results_df = pd.DataFrame()
                for est in estimates:
                    results_df = pd.concat([results_df, est['model'].summary().tables[1]], axis=1)
                results_df.to_html(filename)
                return f"Results saved to {filename}"
        else:
            # Display in console
            for i, est in enumerate(estimates):
                output.append(f"\nEstimate {estimate_names[i]}:")
                output.append(str(est['model'].summary()))
                
        return '\n'.join(output)

    def cmd_eststo(self, command):
        """Store estimation results with a name.
        Usage: eststo [name] [, options]
        """
        # Parse command
        parts = command.split(',', 1)
        main_part = parts[0].strip()
        options = self._parse_options(parts[1].strip()) if len(parts) > 1 else {}
        
        # Get estimate name
        if len(main_part.split()) > 1:
            name = main_part.split()[1]
        else:
            # Generate default name if none provided
            name = f'est_{len(self._stored_estimates.estimates) + 1}'
        
        # Get the last estimation result
        if not hasattr(self, '_lastreg'):
            return "Error: No estimation results to store. Run a regression first."
        
        # Store the estimate with metadata
        self._stored_estimates.store(
            name=name,
            model=self._lastreg,
            model_type=getattr(self._lastreg, 'model_type', 'unknown'),
            depvar=getattr(self._lastreg, 'depvar', 'unknown'),
            indepvars=getattr(self._lastreg, 'indepvars', []),
            options=options
        )
        
        return f"Estimate stored as '{name}'"

    def cmd_esttab(self, command):
        """Display or save stored estimates in a formatted table.
        Usage: esttab [namelist] [, options]
        Options:
            - using(filename): Save to file (CSV, TEX, HTML)
            - title(string): Table title
            - label(string): Table label (for LaTeX)
            - stats(string): Statistics to include
            - keep(string): Variables to keep
            - drop(string): Variables to drop
            - order(string): Variable order
            - star: Add significance stars
            - se: Show standard errors
            - p: Show p-values
            - ci: Show confidence intervals
        """
        # Split command into main part and options
        if ',' in command:
            main_part, options_str = command.split(',', 1)
        else:
            main_part, options_str = command, ''
            
        # Parse main part to get estimate names
        parts = main_part.split()
        if len(parts) > 1:
            estimate_names = parts[1:]
        else:
            # Use all stored estimates if none specified
            estimate_names = list(self._stored_estimates.estimates.keys())
            
        if not estimate_names:
            return "Error: No estimates to display. Run regressions and store them with eststo first."
            
        # Parse options
        options = {}
        if options_str:
            for opt in options_str.split(','):
                opt = opt.strip()
                if '=' in opt:
                    key, value = opt.split('=', 1)
                    options[key.strip()] = value.strip().strip('"')
                else:
                    options[opt] = True
        
        # Get the estimates
        estimates = []
        for name in estimate_names:
            est = self._stored_estimates.get(name)
            if est:
                estimates.append((name, est))
            else:
                return f"Error: Estimate '{name}' not found"
        
        # Create DataFrame for the table
        table_data = {}
        
        # Process each estimate
        for name, est in estimates:
            model = est['model']
            params = model.params
            bse = model.bse
            pvalues = model.pvalues
            
            # Add coefficients
            for var in params.index:
                if var not in table_data:
                    table_data[var] = {}
                table_data[var][name] = params[var]
                
                # Add standard errors if requested
                if options.get('se', False):
                    se_name = f"{name}_se"
                    if se_name not in table_data:
                        table_data[se_name] = {}
                    table_data[se_name][var] = bse[var]
                
                # Add p-values if requested
                if options.get('p', False):
                    p_name = f"{name}_p"
                    if p_name not in table_data:
                        table_data[p_name] = {}
                    table_data[p_name][var] = pvalues[var]
            
            # Add statistics if requested
            if 'stats' in options:
                stats = options['stats'].split()
                for stat in stats:
                    if hasattr(model, stat):
                        stat_value = getattr(model, stat)
                        if stat not in table_data:
                            table_data[stat] = {}
                        table_data[stat][name] = stat_value
        
        # Create DataFrame
        df = pd.DataFrame(table_data)
        
        # Handle variable selection
        if 'keep' in options:
            keep_vars = options['keep'].split()
            df = df.loc[keep_vars]
        if 'drop' in options:
            drop_vars = options['drop'].split()
            df = df.drop(drop_vars, errors='ignore')
        if 'order' in options:
            order_vars = options['order'].split()
            df = df.reindex(order_vars)
        
        # Format the output
        if 'using' in options:
            filename = options['using']
            if filename.endswith('.csv'):
                df.to_csv(filename)
                return f"Results saved to {filename}"
            elif filename.endswith('.tex'):
                # Add LaTeX formatting
                latex_table = df.to_latex(
                    float_format=lambda x: f'{x:.4f}',
                    escape=False,
                    index=True
                )
                if 'title' in options:
                    latex_table = latex_table.replace('\\begin{tabular}', 
                        f'\\begin{{table}}\n\\caption{{{options["title"]}}}\n\\begin{{tabular}}')
                    latex_table = latex_table.replace('\\end{tabular}', 
                        '\\end{tabular}\n\\end{table}')
                if 'label' in options:
                    latex_table = latex_table.replace('\\end{table}', 
                        f'\\label{{{options["label"]}}}\n\\end{{table}}')
                with open(filename, 'w') as f:
                    f.write(latex_table)
                return f"Results saved to {filename}"
            elif filename.endswith('.html'):
                df.to_html(filename)
                return f"Results saved to {filename}"
        else:
            # Display in console
            pd.set_option('display.float_format', lambda x: f'{x:.4f}')
            output = []
            if 'title' in options:
                output.append(options['title'])
                output.append('=' * len(options['title']))
            output.append(df.to_string())
            return '\n'.join(output)

    def on_dataframe_selected(self, item):
        df_name = item.data(Qt.UserRole)
        if df_name in self._dataframes:
            self._df = self._dataframes[df_name]
            self._current_df_name = df_name
            self.update_variables_pane()
            self.update_dataframes_list()

    def on_python_var_double_clicked(self, item):
        var_name = item.text().split()[0]
        self.command_prompt.setPlainText(var_name)

    def update_dataframes_list(self):
        self.dataframes_list.clear()
        for name in self._dataframes:
            item = QListWidgetItem(name)
            item.setData(Qt.UserRole, name)
            self.dataframes_list.addItem(item)
        # Set current selection
        if self._current_df_name in self._dataframes:
            for i in range(self.dataframes_list.count()):
                if self.dataframes_list.item(i).text() == self._current_df_name:
                    self.dataframes_list.setCurrentRow(i)
                    break

    def update_python_vars(self):
        # Show user-defined Python variables (not DataFrame columns or initial vars)
        self.python_vars_list.clear()
        for var_name, var_value in self._python_context.items():
            if not var_name.startswith('_') and var_name not in ['df', 'gui']:
                self.python_vars_list.addItem(f"{var_name} = {var_value}")

    def cmd_coefplot(self, command):
        # Parse command for model names and options
        parts = command.split()
        model_names = []
        options = {}
        for part in parts[1:]:
            if '=' in part or '(' in part:
                # Option
                k, v = part.split('=', 1) if '=' in part else (part, True)
                options[k.strip()] = v.strip() if v is not True else True
            else:
                model_names.append(part)
        if not model_names:
            # Use lastreg if no model specified
            model_names = [self._stored_estimates.current_name or 'lastreg']
        models = []
        for name in model_names:
            est = self._stored_estimates.get(name)
            if est:
                models.append(est['model'])
        if not models:
            return 'No stored models found. Run regressions first.'
        plt.figure(figsize=(8, 5))
        for i, model in enumerate(models):
            coefs = model.params.drop('const', errors='ignore')
            errors = model.bse[coefs.index]
            pos = np.arange(len(coefs)) + i*0.2
            plt.errorbar(coefs.values, pos, xerr=errors.values, fmt='o', label=model_names[i])
        plt.yticks(np.arange(len(coefs)), coefs.index)
        plt.axvline(0, color='gray', linestyle='--')
        plt.xlabel('Coefficient')
        plt.title('Coefficient plot')
        plt.legend()
        show_and_close_figures()
        return 'Coefficient plot displayed.'

    def cmd_lgraph(self, command):
        # Example: lgraph yvar xvar [groupvar] [, title("title") xtitle("X") ytitle("Y")]
        main, options_str = self._split_command_options(command)
        tokens = main.split()
        if len(tokens) < 3:
            return 'Usage: lgraph <yvar> <xvar> [groupvar] [, title("title") xtitle("X") ytitle("Y")]'
        yvar, xvar = tokens[1].rstrip(','), tokens[2].rstrip(',')
        
        # Check for group variable in either format: "lgraph y t g" or "lgraph y t, by(g)"
        groupvar = None
        if len(tokens) > 3:
            groupvar = tokens[3].rstrip(',')
        else:
            options = self._parse_options(options_str)
            if 'by' in options:
                groupvar = options['by'].strip('"')
        
        # Check if variables exist
        missing_vars = [var for var in [yvar, xvar] if var not in self._df.columns]
        if groupvar:
            if groupvar not in self._df.columns:
                missing_vars.append(groupvar)
        if missing_vars:
            return f"Error: Variable(s) not found: {', '.join(missing_vars)}"
        
        options = self._parse_options(options_str)
        xtitle, ytitle, title = self._parse_plot_titles(options, xvar, yvar, f'Line graph of {yvar} vs {xvar}')
        
        plt.figure(figsize=(10, 6))
        
        if groupvar and groupvar in self._df.columns:
            # Grouped series
            for key, grp in self._df.groupby(groupvar):
                plt.plot(grp[xvar], grp[yvar], label=str(key))
            plt.legend()
        else:
            # Single series - calculate mean for each x value
            grouped = self._df.groupby(xvar)[yvar].mean()
            plt.plot(grouped.index, grouped.values, label=f'{yvar} vs {xvar}')
        
        plt.xlabel(xtitle)
        plt.ylabel(ytitle)
        plt.title(title)
        plt.grid(True, alpha=0.3)
        show_and_close_figures()
        return f'Line graph of {yvar} vs {xvar} displayed.'

    def cmd_hist(self, command):
        # Example: hist var [, options]
        main, options_str = self._split_command_options(command)
        tokens = main.split()
        if len(tokens) < 2:
            return 'Usage: hist <var> [, options]'
        var = tokens[1].rstrip(',')
        options = self._parse_options(options_str)
        xtitle, ytitle, title = self._parse_plot_titles(options, var, 'Frequency', f'Histogram of {var}')
        plt.figure(figsize=(10, 6))
        plt.hist(self._df[var])
        plt.xlabel(xtitle)
        plt.ylabel(ytitle)
        plt.title(title)
        plt.grid(True, alpha=0.3)
        show_and_close_figures()
        return f'Generated histogram of {var}'

    def cmd_bash(self, command):
        # Example: bash <shell command>
        import subprocess
        parts = command.split(' ', 1)
        if len(parts) < 2:
            return 'Usage: bash <shell command>'
        shell_cmd = parts[1]
        try:
            result = subprocess.run(shell_cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                return f'Error: {result.stderr.strip()}'
        except Exception as e:
            return f'Error executing shell command: {e}'

    def cmd_browse(self, command):
        # Simulate opening the data browser for tests
        return 'Data browser opened'

    @property
    def df(self):
        return self._df

    @df.setter
    def df(self, value):
        self._df = value
        self._python_context['_df'] = value

    def execute_command(self):
        command = self.command_prompt.toPlainText().strip()
        if not command:
            return
        
        # Split into individual commands and handle line continuations
        raw_commands = [cmd.strip() for cmd in command.split('\n') if cmd.strip()]
        processed_commands = []
        current_cmd = []
        
        for cmd in raw_commands:
            if cmd.endswith('\\'):
                # Remove the backslash and add to current command
                current_cmd.append(cmd[:-1].strip())
            else:
                # Add the last part and complete the command
                current_cmd.append(cmd)
                processed_commands.append(' '.join(current_cmd))
                current_cmd = []
        
        # Handle any remaining command parts
        if current_cmd:
            processed_commands.append(' '.join(current_cmd))
        
        # Execute each processed command separately
        for cmd in processed_commands:
            # Add the command to the results window in bold
            self.results_window.append(cmd)
            print(f"[COMMAND] {cmd}")  # Log command to console
            
            # Log command to history
            self.command_history.append(cmd)
            self.history_list.addItem(cmd)
            
            # Handle 'by' prefix (e.g., by firm: egen ... or bys firm: egen ...)
            by_match = re.match(r'^by[s]?\s+([\w\s]+):\s*(.+)$', cmd)
            if by_match:
                by_vars = by_match.group(1).strip()
                inner_cmd = by_match.group(2).strip()
                # If the inner command is egen, append by() option
                if inner_cmd.lower().startswith('egen '):
                    # If there's already a by() option, don't add another
                    if ', by(' not in inner_cmd:
                        # Insert by() at the end
                        if ',' in inner_cmd:
                            inner_cmd = inner_cmd + f' by({by_vars})'
                        else:
                            inner_cmd = inner_cmd + f', by({by_vars})'
                # Replace cmd with the modified inner_cmd
                cmd = inner_cmd
            
            # Strip comments for execution
            exec_command = self._strip_stata_comments(cmd)
            if not exec_command:
                continue
            
            # Execute command and capture output
            try:
                if exec_command.startswith('!'):
                    # Execute bash command (remove the ! prefix)
                    bash_command = exec_command[1:].strip()
                    process = subprocess.Popen(bash_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    stdout, stderr = process.communicate()
                    output = stdout.decode() + stderr.decode()
                    self.results_window.append(output)
                    print(f"[OUTPUT] {output}")  # Log output to console
                elif exec_command.startswith('>'):
                    # Execute Python command (remove the > prefix)
                    python_command = exec_command[1:].strip()
                    output_buffer = io.StringIO()
                    with contextlib.redirect_stdout(output_buffer), contextlib.redirect_stderr(output_buffer):
                        exec(python_command, self._python_context)
                    result = output_buffer.getvalue()
                    # After exec, update self._df in case it was changed in Python
                    if '_df' in self._python_context:
                        self._df = self._python_context['_df']
                    self.results_window.append(str(result))
                    print(f"[OUTPUT] {result}")  # Log output to console
                    # Update the list of Python variables
                    self.update_python_vars()
                elif exec_command.startswith('bash '):
                    # Execute bash command
                    bash_command = exec_command[5:]  # Remove 'bash ' prefix
                    process = subprocess.Popen(bash_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    stdout, stderr = process.communicate()
                    output = stdout.decode() + stderr.decode()
                    self.results_window.append(output)
                    print(f"[OUTPUT] {output}")  # Log output to console
                elif exec_command.startswith('cd '):
                    # Handle cd command
                    directory = exec_command[3:].strip()
                    try:
                        os.chdir(directory)
                        output = f'Changed directory to {os.getcwd()}'
                        self.results_window.append(output)
                        print(f"[OUTPUT] {output}")  # Log output to console
                    except Exception as e:
                        error_msg = f'Error changing directory: {str(e)}'
                        self.results_window.append(error_msg)
                        print(f"[ERROR] {error_msg}")  # Log error to console
                elif exec_command in ['pwd', 'ls']:
                    # Execute pwd and ls commands directly
                    process = subprocess.Popen(exec_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    stdout, stderr = process.communicate()
                    output = stdout.decode() + stderr.decode()
                    self.results_window.append(output)
                    print(f"[OUTPUT] {output}")  # Log output to console
                else:
                    # Find handler for Stata-like commands
                    cmd_name = exec_command.split()[0].lower()
                    handler = self.command_registry.get(cmd_name)
                    if handler:
                        result = handler(exec_command)
                        # If handler modified self._df, update context
                        self._python_context['_df'] = self._df
                        self.results_window.append(str(result))
                        print(f"[OUTPUT] {result}")  # Log output to console
                    else:
                        # Execute as Python code in persistent context
                        try:
                            output_buffer = io.StringIO()
                            with contextlib.redirect_stdout(output_buffer), contextlib.redirect_stderr(output_buffer):
                                exec(exec_command, self._python_context)
                            result = output_buffer.getvalue()
                            # After exec, update self._df in case it was changed in Python
                            if '_df' in self._python_context:
                                self._df = self._python_context['_df']
                            self.results_window.append(str(result))
                            print(f"[OUTPUT] {result}")  # Log output to console
                            # Update the list of Python variables
                            self.update_python_vars()
                        except Exception as e:
                            error_msg = f'Error: {str(e)}'
                            self.results_window.append(error_msg)
                            print(f"[ERROR] {error_msg}")  # Log error to console
            except Exception as e:
                error_msg = f'Error: {str(e)}'
                self.results_window.append(error_msg)
                print(f"[ERROR] {error_msg}")  # Log error to console
            
            # Add a blank line between command outputs
            self.results_window.append('')
        
        # Clear the command prompt
        self.command_prompt.clear()

    def _strip_stata_comments(self, command):
        """Remove Stata-style comments from a command line, preserving quoted strings."""
        # Skip full-line comments
        if command.strip().startswith('*') or command.strip().startswith('//'):
            return ''
        # Remove end-of-line comments (//), but not inside quotes
        in_quotes = False
        result = ''
        i = 0
        while i < len(command):
            if command[i] == '"':
                in_quotes = not in_quotes
                result += command[i]
                i += 1
            elif not in_quotes and command[i:i+2] == '//':
                break  # Ignore rest of line
            else:
                result += command[i]
                i += 1
        return result.strip()


class StoredEstimates:
    def __init__(self):
        self.estimates = {}  # Dictionary to store regression results
        self.current_name = None  # Name of current estimate
        
    def store(self, name, model, model_type, depvar, indepvars, options=None):
        """Store a regression result with metadata"""
        self.estimates[name] = {
            'model': model,
            'type': model_type,  # 'ols', 'iv', 'reghdfe', etc.
            'depvar': depvar,
            'indepvars': indepvars,
            'options': options or {},
            'timestamp': pd.Timestamp.now(),
            'stats': {}  # For storing additional statistics
        }
        self.current_name = name
        
    def get(self, name):
        """Retrieve a stored estimate"""
        return self.estimates.get(name)
        
    def list(self):
        """List all stored estimates"""
        return list(self.estimates.keys())
        
    def drop(self, name):
        """Remove a stored estimate"""
        if name in self.estimates:
            del self.estimates[name]
            
    def clear(self):
        """Clear all stored estimates"""
        self.estimates.clear()
        self.current_name = None
        
    def add_stat(self, name, stat_name, value):
        """Add a statistic to a stored estimate"""
        if name in self.estimates:
            self.estimates[name]['stats'][stat_name] = value

    def __contains__(self, name):
        return name in self.estimates


# Add a global flag to control test environment
IS_TEST_ENVIRONMENT = False

def show_and_close_figures():
    if IS_TEST_ENVIRONMENT:
        plt.show(block=False)
        plt.pause(0.5)
        plt.close('all')
    else:
        plt.show()

def set_test_environment(value=True):
    global IS_TEST_ENVIRONMENT
    IS_TEST_ENVIRONMENT = value


if __name__ == "__main__":
    IS_TEST_ENVIRONMENT = False
    app = QApplication(sys.argv)
    window = StatacondaGUI()
    window.show()
    sys.exit(app.exec_()) 