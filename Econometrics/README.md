# Python (and R) for Econometrics
This module of the course covers non-structural econometric methods and how to apply these tools using open source software: both in Python and in R.

## 1. Installing R and Setting up R Kernel for Jupyter Notebooks

There are a couple options here and some are more or less straight forward depending on your OS and if you already have R installed already.

### Method 1:
* Follow [these instructions](https://docs.anaconda.com/anaconda/navigator/tutorials/r-lang/) to install the R kernel for Jupyter Notebook using the Anaconda navigator.

### Method 2:
Do the following:
1. [Install R from CRAN](https://cran.r-project.org)
2. If you want a popular IDE for R, [install RStudio](https://www.rstudio.com)
3. With R installed, go to your command line and do the following to create an R kernel for Jupyter Notebooks:
    * `$ R` # to launch R
    * `install.packages('IRkernel')`
    * `IRkernel::installspec()` or, to install system-wide for all users do `IRkernel::installspec(user = FALSE)`

Additionally, if you like VS Code and want to continue using it for writing R scripts, you will want to add the [R extension](https://marketplace.visualstudio.com/items?itemName=Ikuyadeu.r) to VS Code.

(Note that Anaconda's package manager Conda does manage R packages, but at least in my experience with OSX, this manager has not worked well.  Thus I recommend installing from CRAN as outlined above.)

## 2. Notebooks we worked through in class

Python:
* [Econometrics in Python](https://github.com/jdebacker/CompEcon_Fall21/blob/main/Econometrics/Python_Econometrics.ipynb)
* [RDD in Python](https://github.com/jdebacker/CompEcon_Fall21/blob/main/Econometrics/Python_RDD.ipynb)

R:
* [Intro to R data structures](https://github.com/jdebacker/CompEcon_Fall21/blob/main/Econometrics/R_Basics.ipynb)
* [Reading and writing data in R](https://github.com/jdebacker/CompEcon_Fall21/blob/main/Econometrics/R_Data.ipynb)
* [Writing functions, numerical optimization in R](https://github.com/jdebacker/CompEcon_Fall21/blob/main/Econometrics/R_Functions.ipynb)
* [Econometrics in R](https://github.com/jdebacker/CompEcon_Fall21/blob/main/Econometrics/R_Econometrics.ipynb)
* [RDD in R](https://github.com/jdebacker/CompEcon_Fall21/blob/main/Econometrics/R_RDD.ipynb)



## 3. Useful Links

* Python
  * [Stata to Python cheat sheet](https://cheatsheets.quantecon.org/stats-cheatsheet.html)
  * [StatsModel Package](http://www.statsmodels.org/dev/index.html)
  * [QuantEcon: Linear regression in Python](https://lectures.quantecon.org/py/ols.html)
  * Panel Data Models in Python via the [linearmodels package](https://pypi.python.org/pypi/linearmodels)
  * [rpy2 for calling R from Python](http://rpy2.readthedocs.io/en/version_2.8.x/overview.html)
  * [RDD packge for Python](https://github.com/evan-magnusson/rdd), still under development by the excellent Evan Magnusson
* R
  * [Guide to R for Stata Users](http://dss.princeton.edu/training/RStata.pdf)
  * [Panel Data Models in R](https://www.princeton.edu/~otorres/Panel101R.pdf)
  * [The R GGPlot Gallery](http://www.r-graph-gallery.com/portfolio/ggplot2-package/)
* General
  * Gentzkow and Shaprio, [Code and Data for the Social Sciences: A Practitioner's Guide](http://web.stanford.edu/~gentzkow/research/CodeAndData.pdf)
  * *Causal Inference: The Mixtape*, [code for examples in R and Python](https://github.com/scunning1975/mixtape)