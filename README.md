# Raman Spectra Sampling Size Analysis

A sampling size analysis library for Single-cell Raman Spectroscopy data using
<b>Kernel divergence</b>, a novel dataset-wise metric.

# Installation

## Install dependencies

This library requires `python>=3.6`, with package dependencies:

* numpy
* scikit-learn
* tqdm (optional)

To instal the dependencies, run the command below in terminal:

```bash
pip install numpy scikit-learn tqdm
```

## Install scripts

This library uses standalone scripts only. The installation is as easy as
copying all scripts under the `script/` to the destination location.

# Usage

The sampling size analysis is conducted in two steps: (1) sampling simulation;
and (2) analysis of kernel divergence. The simulation step is the most
compulationally intensive and time consuming step.

## 1. Input data

The input data must be organized as a single tabular txt file of pure data,
<b>without</b> column header and row header, and with rows as samples (i.e. each
row is a sample).
The data must be real-valued, and demical values and integers are both accepted.
However the integers are assumed to be numerical rather than categorical or
bit-field.

An example of input data is given as `doc/k4normal.n_200.tsv`, with nxd=200x2
(200 samples, 2 dimensions).

## 2. Simulation

```bash
python ./script/kdiv_sampling.py \
	-k 5 \
	--depth-min 5 \
	--depth-max 200 \
	-r 1000 \
	-o sim_out.json \
	--progress \
	doc/k4normal.n_200.tsv
```

The command above takes the aforementioned example data as input, then runs 1000
repeats of sampling simulation, with batch size of 5, throughout the sampling
depth from 5 to 200. Its result output will be saved in sim_out.json. This step
may take a while based on the batch size, sampling depth range and number of
repeats so please be patient. The results are in json format, stores the
eigenvalues acquired at each sampling depth at each repeat of simulation.
The `--progress` option will show a progress bar (requires module `tqdm`) when
running this script from terminal.


## 3. Kernel divergence analysis

```bash
python ./script/kdiv_analysis.py \
	-n 3 \
	-o sim_out.kdiv.n_3.tsv \
	sim_out.json
```

The command above takes the simulation output generated in the previous step as
input, then calculates the kernel divergence at each sampling depth for each of
the 1000 repeats. The output is by default a tab-delimited table, with the
header row stating the simulated sampling depths. The number of data rows equals
the number of repeats in the previous simulation step. Option `-n 3` instructs
the kernel divergence calculation to the the first 3 eigenvalues; an alternative
option is `-p`, which instructs the calculation to use the largest eigenvalues
whose sum is greater than this fraction of total eigenvalue sum. For example
`-p 0.95` will use the first several eigenvalues that are larger than 95% of the
total sum.

By the end of this step, the user should be able to plot the mean kernel
divergence change at the sampling depth, or do a further analysis of the point
of convergence using a user-specified threshold.


# Publication

If you used this method in your research, please site our original paper:

```
Li, G., Wu, C., Wang, D., Srinivasan, V., Kaeli, D.R., Dy, J.G. and Gu, A.Z., 2022. Machine Learning-Based Determination of Sampling Depth for Complex Environmental Systems: Case Study with Single-Cell Raman Spectroscopy Data in EBPR Systems. Environmental Science & Technology, 56(18), pp.13473-13484.
```
