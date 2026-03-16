<p align="center">
  <img src="bagging_figure/horizontalsimplebagging.png" width="1300" />
</p>


# Bagging for LID estimation

#### This repository has been developed for the publication: [On the Use of Bagging for Local Intrinsic Dimensionality Estimation](https://linktopaper) (link not yet active, paper submitted for review)

- To recreate the results, plots, and figures present in the publication, first follow the [Installation](#installation) steps. Then do either of the following:

  - Download ready experiment objects as explained at [Data availability](#data-availability), then load them to create the plots and figures via the instructions at [Reproducibility](#reproducibility).
  - Follow the instructions at [Reproducibility](#reproducibility) to resample new datasets and perform the LID estimation experiments in the publication from scratch.

- To use the Bagging_for_LID package for your own LID estimation experiments, or to examine downloaded experiment objects, first install the package via the [Installation](#installation) steps. Then follow the instructions at [Tutorial for the package](#tutorialforthepackage).

## Installation

#### Install the Bagging_for_LID package and its requirements.
 
To begin, clone the main repository to a selected folder

```bash
git clone https://github.com/anonymconference-star/Bagging_for_LID_Estimation.git
```

Navigate to selected folder

```bash
cd Bagging_for_LID_Estimation
```

Install requirements. Python version $\geq 3.11$ required.

```bash
pip install -r requirements.txt
```

Install package

```bash
pip install -e .
```

## Data availability

#### Downloading the exact experiment objects containing the data and already performed experiments for the publication [On the Use of Bagging for Local Intrinsic Dimensionality Estimation](https://linktopaper)

- Download the experiment objects from **Zenodo**: The source files (.pkl) available at [Zenodo link](https://zenodo.org/records/18847030?preview=1&token=eyJhbGciOiJIUzUxMiIsImlhdCI6MTc3MzE1OTc5MCwiZXhwIjoxNzk4NzYxNTk5fQ.eyJpZCI6ImRjZDdkM2UwLTFlZWMtNDE2MC1iN2NkLTlhNzJhNWVjZTkyYyIsImRhdGEiOnt9LCJyYW5kb20iOiJiMzgwMzZmZmM1M2QzNGIzM2U4OGM4Y2IwZDg0Yjc5ZCJ9.fF1JlcOf3kYWrotuWGq_efGicrFFp1gJlIRcy60pL-3f8ppJmUAjSn8Gakros4c7EWUdwBfzPIdh0YZJ0b20CQ) can be used together with our code to extract all the necessary information about the performed experiments, as well as to recreate the figures and the values in the tables that are also already viewable at [Output](Output). Amongst these downloadable files, the larger files with prefix 'mergedresult' contain all data required to obtain the results (including datasets). While the 'light_mergedresult' smaller files have only the data necessary to reconstruct the plots, they are more like data storage, not interactive class objects.

- Extract the experiment objects from the downloaded .zip file to a selected folder, which will serve as the directory for loading and saving experiments. You are now ready to give the path to the directory and load the files by either using [recreate_results_notebook](Reproducibility/recreate_results_notebook.ipynb) or [recreate_results](Reproducibility/recreate_results.py) and setting **load = True**.

- Alternatively, the files may be loaded for detailed inspection by following the instructions in the [single_experiment](Tutorials/single_experiment.ipynb) tutorial.

## Reproducibility

#### Recreating results and figures from the publication [On the Use of Bagging for Local Intrinsic Dimensionality Estimation](https://linktopaper) 

- The [recreate_results_notebook](Reproducibility/recreate_results_notebook.ipynb) Jupyter notebook file contains detailed, step-by-step instructions on how to
recreate the results and figures present in the paper from scratch, or by loading already computed experiment objects.
- The [recreate_example_figures](Reproducibility/recreate_example_figures.ipynb) Jupyter notebook file can be used to recreate the plots in the Introduction section of the publication.
- The [recreate_results](Reproducibility/recreate_results.py) Python file can be used to recreate the same results without the use of a Jupyter notebook.

## Tutorial for the package

#### Using the Bagging_for_LID package to run your own LID estimation experiments

- The [single_experiment](Tutorials/single_experiment.ipynb) jupyter notebook file can be used to learn how to use the repository for examining or performing single LID estimation experiments, or multiple ones at once for a range of parameter combinations.





