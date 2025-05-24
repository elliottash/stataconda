# Stataconda

## Overview
Stataconda is a powerful desktop GUI application that provides a Stata-like interface while leveraging Python's data science ecosystem. It accepts Stata commands, translates them into Python (using pandas, statsmodels, seaborn, etc.), executes the Python code, and formats the output to match Stata's familiar style.

## Features
- **Stata-like Interface**: Familiar command syntax and output formatting
- **Python Backend**: Leverages pandas, statsmodels, scikit-learn, and other Python libraries
- **Interactive GUI**: Command prompt, results window, variable list, and data browser
- **Data Management**: Support for CSV, Excel, and Stata .dta files
- **Statistical Analysis**: Comprehensive set of statistical commands
- **Visualization**: Rich plotting capabilities with matplotlib and seaborn
- **Bash Integration**: Execute shell commands directly from the interface

## Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/stataconda.git
cd stataconda

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

## Quick Start
1. Launch the application
2. Load a dataset using `use filename.dta` or `import filename.csv`
3. Explore data with `browse`, `summarize`, or `describe`
4. Run analyses using Stata-like commands
5. Save results using `save filename.dta` or `export using filename.rtf`

## Detailed Command Documentation

### Data Management Commands

#### `use` / `import`
Load a dataset from a file.
```stata
use filename.dta
import filename.csv
```
- Supports .dta, .csv, and .xlsx files
- Automatically detects file type from extension
- Updates variable list and data browser

#### `save`
Save the current dataset to a file.
```stata
save filename.dta
```
- Supports .dta, .csv, and .xlsx files
- Defaults to .dta if no extension provided

#### `export`
Export results to various formats.
```stata
export using filename.rtf
```
- Supports .rtf, .csv, .xlsx, .html, .tex formats
- Used primarily for exporting regression results

### Data Exploration Commands

#### `browse`
Open the data browser to view and edit data.
```stata
browse
browse var1 var2
```

#### `describe`
Display overview of dataset variables.
```stata
describe
describe var1 var2
```

#### `summarize`
Calculate summary statistics.
```stata
summarize
summarize var1 var2
```

#### `tabulate`
Create frequency tables.
```stata
tabulate var1
tabulate var1 var2
```

### Data Manipulation Commands

#### `generate`
Create new variables.
```stata
generate newvar = expression
```

#### `replace`
Modify existing variables.
```stata
replace var = expression
```

#### `drop`
Remove variables or observations.
```stata
drop var1 var2
drop if condition
```

#### `keep`
Keep specified variables or observations.
```stata
keep var1 var2
keep if condition
```

#### `rename`
Rename variables.
```stata
rename oldvar newvar
```

#### `recode`
Recode variable values.
```stata
recode var (1=2) (2=1)
```

#### `clonevar`
Create a copy of a variable.
```stata
clonevar newvar = oldvar
```

### Label Commands

#### `label variable`
Add variable labels.
```stata
label variable var "label"
```

#### `label values`
Add value labels.
```stata
label values var valuelist
```

### Data Combination Commands

#### `append`
Append datasets.
```stata
append using filename
```

#### `merge`
Merge datasets.
```stata
merge 1:1 keyvar using filename
```

#### `joinby`
Join datasets by key variables.
```stata
joinby keyvar using filename
```

#### `cross`
Create cross-product of datasets.
```stata
cross using filename
```

### Statistical Analysis Commands

#### `regress`
Linear regression.
```stata
regress depvar indepvar1 indepvar2 [, options]
```

#### `anova`
Analysis of variance.
```stata
anova depvar indepvar1 indepvar2
```

#### `areg`
Linear regression with absorbed fixed effects.
```stata
areg depvar indepvar1 indepvar2, absorb(groupvar)
```

#### `xtreg`
Panel data regression.
```stata
xtreg depvar indepvar1 indepvar2, fe
```

#### `fe`
Fixed effects regression.
```stata
fe depvar indepvar1 indepvar2
```

#### `logit` / `probit` / `logistic`
Binary outcome models.
```stata
logit depvar indepvar1 indepvar2
probit depvar indepvar1 indepvar2
logistic depvar indepvar1 indepvar2
```

#### `poisson` / `nbreg`
Count data models.
```stata
poisson depvar indepvar1 indepvar2
nbreg depvar indepvar1 indepvar2
```

#### `tobit` / `intreg`
Censored/interval regression models.
```stata
tobit depvar indepvar1 indepvar2
intreg depvar indepvar1 indepvar2
```

#### `ivregress` / `ivreg2`
Instrumental variables regression.
```stata
ivregress 2sls depvar (endogvar = instrument) exogvar1 exogvar2
ivreg2 depvar (endogvar = instrument) exogvar1 exogvar2
```

### Time Series Commands

#### `xtset` / `tsset`
Set panel/time series structure.
```stata
xtset panelvar timevar
tsset timevar
```

#### `arima`
ARIMA modeling.
```stata
arima depvar, arima(p,d,q)
```

#### `arch`
ARCH/GARCH modeling.
```stata
arch depvar, arch(1) garch(1)
```

#### `var` / `vec`
Vector autoregression/error correction.
```stata
var depvar1 depvar2, lags(2)
vec depvar1 depvar2, lags(2)
```

#### `newey`
Newey-West regression.
```stata
newey depvar indepvar1 indepvar2, lag(2)
```

### Visualization Commands

#### `scatter`
Create scatter plots.
```stata
scatter yvar xvar [, options]
```

#### `histogram`
Create histograms.
```stata
histogram var [, options]
```

#### `graph bar`
Create bar graphs.
```stata
graph bar var [, options]
```

#### `binscatter`
Create binned scatter plots.
```stata
binscatter yvar xvar [, options]
```

#### `coefplot`
Plot regression coefficients.
```stata
coefplot model1 model2 [, options]
```

#### `lgraph`
Create line graphs.
```stata
lgraph yvar xvar [groupvar] [, options]
```

### Estimation Results Commands

#### `estout` / `esttab`
Format and export estimation results.
```stata
estout using filename.rtf
esttab using filename.rtf
```

#### `estadd`
Add statistics to stored estimates.
```stata
estadd local statname value
```

#### `eststo`
Store estimation results.
```stata
eststo name: command
```

### Utility Commands

#### `bash`
Execute shell commands.
```stata
bash command
```

#### `do`
Execute a do-file.
```stata
do filename.do
```

#### `egen`
Generate extended variables.
```stata
egen newvar = function(arguments)
```

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details. 