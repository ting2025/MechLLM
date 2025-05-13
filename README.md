# MechLLM
This repository aims to help assist the extraction of information from chemical reaction mechanism images.
## Getting started
### Installing requirements
To ensure consistency and reproducibility, this project uses a virtual environment defined by the `4980_env.txt` file.
```bash
$ python -m venv env
$ source env/bin/activate
$ pip install -r 4980_env.txt
```
If you would like to use your own pre-installed environment, you may directly install the requirements by the following:
```
pip install -r requirements.txt
```
if there are conflicts in installing the requirements and could not initiate the installing,
please use the following and debug based on error outputs.
```
python requirements.py
```
### LLM utilization
This repository uses ChatGPT-4o as a demonstration of assisted information extraction by LLM in such visual chemistry tasks, therefore you will need to have your own API key in order to assess the features in this repository. Copy and paste your API key into `API_key.txt`.
## Atom labelling
The referencing of atoms uses number to label all the atoms, both graph based and text based molecules in the reaction.</br>

This repository highly leverages [MolScribe](https://github.com/thomas0809/MolScribe) and [RxnScribe](https://github.com/thomas0809/RxnScribe/tree/main). Refer to their paper and their code for a deeper understanding of how I leveraged their object recognition method and optical character recogniton. </br>
## Information Extraction
From Pistachio, this project generated 17k synthetic reaction images and drawed synthetic curved arrows based on the atom label. They are documented as a JSON file respectively. Please refer to [Huggingface](https://huggingface.co/datasets/Ting25/MechLLM), and download respecting files. </br>
Run `getReactionNumber.ipynb` to visualize the extraction.


   
