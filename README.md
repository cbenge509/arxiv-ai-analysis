arXiv.org AI/ML Analysis
==========================================================

![GitHub](https://img.shields.io/github/license/cbenge509/arxiv-ai-analysis) ![GitHub Pipenv locked Python version](https://img.shields.io/github/pipenv/locked/python-version/cbenge509/arxiv-ai-analysis) ![GitHub Pipenv locked dependency version](https://img.shields.io/github/pipenv/locked/dependency-version/cbenge509/arxiv-ai-analysis/pandas) ![GitHub Pipenv locked dependency version](https://img.shields.io/github/pipenv/locked/dependency-version/cbenge509/arxiv-ai-analysis/h5py) ![GitHub contributors](https://img.shields.io/github/contributors/cbenge509/arxiv-ai-analysis) ![GitHub repo size](https://img.shields.io/github/repo-size/cbenge509/arxiv-ai-analysis)

<img align="right" width="180" src="./images/ucb.png"/>

#### Authors : [Rahul Kulkarni](https://www.linkedin.com/in/rahul-kulkarni-6544545/) | [Anu Yadav](https://www.linkedin.com/in/anuyadav1/) | [Cristopher Benge](https://cbenge509.github.io/)

[![](https://sourcerer.io/fame/cbenge509/cbenge509/arxiv-ai-analysis/images/0)](https://sourcerer.io/fame/cbenge509/cbenge509/arxiv-ai-analysis/links/0)[![](https://sourcerer.io/fame/cbenge509/cbenge509/arxiv-ai-analysis/images/1)](https://sourcerer.io/fame/cbenge509/cbenge509/arxiv-ai-analysis/links/1)[![](https://sourcerer.io/fame/cbenge509/cbenge509/arxiv-ai-analysis/images/2)](https://sourcerer.io/fame/cbenge509/cbenge509/arxiv-ai-analysis/links/2)[![](https://sourcerer.io/fame/cbenge509/cbenge509/arxiv-ai-analysis/images/3)](https://sourcerer.io/fame/cbenge509/cbenge509/arxiv-ai-analysis/links/3)[![](https://sourcerer.io/fame/cbenge509/cbenge509/arxiv-ai-analysis/images/4)](https://sourcerer.io/fame/cbenge509/cbenge509/arxiv-ai-analysis/links/4)[![](https://sourcerer.io/fame/cbenge509/cbenge509/arxiv-ai-analysis/images/5)](https://sourcerer.io/fame/cbenge509/cbenge509/arxiv-ai-analysis/links/5)[![](https://sourcerer.io/fame/cbenge509/cbenge509/arxiv-ai-analysis/images/6)](https://sourcerer.io/fame/cbenge509/cbenge509/arxiv-ai-analysis/links/6)[![](https://sourcerer.io/fame/cbenge509/cbenge509/arxiv-ai-analysis/images/7)](https://sourcerer.io/fame/cbenge509/cbenge509/arxiv-ai-analysis/links/7)


U.C. Berkeley, Masters in Information & Data Science program - [datascience@berkeley](https://datascience.berkeley.edu/) <br>
Summer 2020, W209 - Data Visualization - [Andrew Reagan, PhD](https://www.linkedin.com/in/andyreagan/) - Section 4

---

## Description

This repo contains the draft work for the visualization of AI/ML research papers catalogued on [arXiv.org](https://arxiv.org/) for calendary years 1993 through 2019.  Categories under consideration have been limited to:

 - Computer Science: Artificial Intelligence [``cs: AI``]
 - Computer Science: Machine Learning [``cs: LG``]
 - Statistics: Machine Learning [``stat: ML``] 

---

#### Highlight of key files included in this repository:

  |File | Description |
  |:----|:------------|
  |[load_base_data.py](load_base_data.py)| Processes the base arXiv categories data, storing the output into a single Pandas.DataFrame (HDF5 file) |
  |[utils/preprocessing.py](utils/preprocessing.py)| Utility class; used for loading the raw arXiv data and storing the Pandas.DataFrame as HDF5 |
  |[Example - Load HDF5 File.ipynb](Example%20-%20Load%20HDF5%20File.ipynb)| Jupyter Notebook demonstrating how to read in the HDF5 stored Pandas.DataFrame |

---

## Visualization Samples

(NOTE: Work in Progress, below is not real!)

<br>
<img width="900" src="./images/25%20Topics%20COVID-19.png"/>
<br>

---

## References

Data was collected from the tremendous work provided by the [arxiv_archive](https://github.com/staeiou/arxiv_archive/) repo and all due credit is referred to: <br><br>``Geiger, R. Stuart (2020). ArXiV Archive: A Tidy and Complete Archive of Metadata for Papers on arxiv.org.`` [doi](10.5281/zenodo.1463242) | [url](http://doi.org/10.5281/zenodo.1463242).

---

License
-------
Licensed under the MIT License. See [LICENSE](LICENSE.txt) file for more details.
