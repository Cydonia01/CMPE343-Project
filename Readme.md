# CMPE343 Statistical Computing Project

This repository contains three statistical computing projects implemented in Python as part of the CMPE343 course. Each project demonstrates different statistical concepts and machine learning techniques.

## Table of Contents

- [Project Overview](#project-overview)
- [Project Components](#project-components)
- [Installation](#installation)
- [Usage](#usage)
- [Data Files](#data-files)
- [Results](#results)
- [Dependencies](#dependencies)
- [Author](#author)

## Project Overview

This project consists of three main components:

1. **Language Model** - A probabilistic text generator using n-gram modeling
2. **Statistical Hypothesis Testing** - Monte Carlo simulation and confidence interval estimation
3. **Binary Classification** - Naive Bayes classifier for detection problems

## Project Components

### 1. Language Model (`model.py`)

A probabilistic language model that generates sentences based on word transition probabilities.

**Features:**

- Reads training data from `sentences.txt`
- Builds word transition probability matrices
- Generates new sentences using probabilistic sampling
- Calculates sentence probabilities

**Key Functions:**

- `generate_sentence()`: Generates a random sentence based on learned probabilities

**Output:** Generates 5 sample sentences with their associated probabilities

### 2. Statistical Analysis (`center.py`)

Implements statistical hypothesis testing and confidence interval estimation using Monte Carlo methods.

**Features:**

- Multivariate normal distribution sampling
- Hypothesis testing for difference in means
- Confidence interval estimation for different sample sizes
- Z-test implementation

**Key Components:**

- Function `g(x)`: Transforms input vectors using polynomial combination
- Sample size analysis: [50, 100, 1000, 10000]
- 95% confidence interval calculations
- Statistical significance testing (α = 0.05)

### 3. Binary Classification (`dune.py`)

A Naive Bayes classifier for binary detection problems using amplitude and distance features.

**Features:**

- Loads detection data from CSV files
- Implements Gaussian Naive Bayes classification
- Calculates classification accuracy
- Tests on both training and new data

**Key Functions:**

- `normal_pdf()`: Computes Gaussian probability density
- `classification()`: Makes binary predictions using Naive Bayes
- Performance evaluation on training and test sets

## Installation

1. Clone this repository:

```bash
git clone https://github.com/Cydonia01/CMPE343-Project.git
cd CMPE343-Project
```

2. Install required dependencies:

```bash
pip install numpy pandas
```

## Usage

### Running the Language Model

```bash
python model.py
```

This will generate 5 sample sentences based on the training data in `sentences.txt`.

### Running Statistical Analysis

```bash
python center.py
```

This will perform hypothesis testing and generate confidence intervals for different sample sizes.

### Running Binary Classification

```bash
python dune.py
```

This will train a Naive Bayes classifier and report accuracy on both training and test data.

## Data Files

### `sentences.txt`

- Contains 1000 training sentences for the language model
- Each sentence is bounded by `<|start|>` and `<|end|>` tokens
- Used to build word transition probabilities

### `detection_data.csv`

- Training dataset for binary classification
- **Columns:**
  - `Distance`: Numerical distance measurements
  - `Amplitude`: Numerical amplitude measurements
  - `Detection`: Binary labels ("Detect" or "No Detect")
- **Size:** 102 samples

### `detection_data_extra.csv`

- Test dataset for evaluating classifier performance
- Same structure as training data
- Used for independent performance validation

## Results

### Language Model

- Successfully generates grammatically coherent sentences
- Demonstrates probabilistic text generation
- Calculates sentence probabilities using log-likelihood

### Statistical Analysis

- Performs hypothesis testing with α = 0.05 significance level
- Generates confidence intervals for various sample sizes
- Critical value: z = 1.96 for 95% confidence level
- Demonstrates convergence properties with increasing sample size

### Binary Classification

- Implements Gaussian Naive Bayes assumption
- Reports classification accuracy on training and test sets
- Uses joint probability estimation for decision making
- Features: Distance and Amplitude measurements

## Dependencies

- **NumPy**: For numerical computations and array operations
- **Pandas**: For data manipulation and CSV file handling
- **Python 3.x**: Required runtime environment

## Technical Details

### Statistical Methods Used

- **Multivariate Normal Distribution Sampling**
- **Z-test for Hypothesis Testing**
- **Gaussian Naive Bayes Classification**
- **Monte Carlo Simulation**
- **Maximum Likelihood Estimation**

### Key Algorithms

- N-gram language modeling with probabilistic sampling
- Confidence interval estimation using Central Limit Theorem
- Binary classification using Bayes' theorem
