# Plagiarism Checker

## Project's Title

Plagiarism Checker is a Python program that compares two text files and calculates the similarity score between them. It uses the cosine similarity measure and TF-IDF (term frequency-inverse document frequency) to determine the degree of similarity between the texts.

## Project Description

Plagiarism is a serious academic offense that can have serious consequences, such as expulsion from school or legal action. The Plagiarism Checker project is designed to help students and educators check for plagiarism in their written work. By comparing two texts, the program can detect similarities and determine if plagiarism has occurred.

The program preprocesses the texts by removing non-alphabetic characters and stop words, and stemming the words. Then, it calculates the cosine similarity score between the texts using the TF-IDF measure. If the similarity score is above a certain threshold (0.8 by default), the program considers the texts to be highly similar and reports a potential case of plagiarism.

## Table of Contents

1. [How to Install and Run the Project](#installation)
2. [How to Use the Project](#usage)
3. [Credits](#credits)
4. [License](#license)
5. [Badges](#badges)

## How to Install and Run the Project

1. Clone the repository or download the ZIP file.
2. Install the required packages by running the command `pip install -r requirements.txt`.
3. Run the program by running the command `plagiarism_checker.py`.
4. Follow the prompts to enter the file paths of the texts to be checked.

## How to Use the Project

To use the Plagiarism Checker, follow these steps:

1. Open the command prompt or terminal and navigate to the directory where the program is saved.
2. Run the command `python plagiarism_checker.py`.
3. Enter the file paths of the texts to be checked when prompted.
4. Read the report generated by the program, which indicates the similarity score and whether the texts are similar or not.

## Credits

The Plagiarism Checker project was developed by GitProSolutions as a project for Plagiarism Checker. It uses the following third-party libraries:

- nltk
- scikit-learn

## License

The Plagiarism Checker project is licensed under the MIT License. See the `LICENSE` file for more details.

## Badges

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
