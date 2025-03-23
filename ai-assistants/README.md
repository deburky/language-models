# AI Underwriter Assistants

<img src="https://img.shields.io/badge/python-3.9.2-blue.svg" alt="python 3.9.2">

Author: [Denis Burakov (@deburky)](https:/www.github.com/deburky)

This project uses Assistants API to create a workflow to approve loan applications.

You can run the code from the notebook in the `notebooks` folder or run via the command line as `uv run main.py`.

The structure of the project is as follows:

```plain
├── README.md
├── data
│   ├── loans-9309cbc146a4.parquet
│   ├── loans-new-9309cbc146a4.parquet
│   └── scores-new-9309cbc146a4.parquet
├── instructions
│   ├── instructions_decision_maker.txt
│   └── instructions_underwriter.txt
├── main.py
├── notebooks
│   └── AI-Underwriter-Assistants.ipynb
├── pyproject.toml
├── report
│   └── underwriting_conversation.html
├── src
│   ├── __init__.py
│   └── utils.py
└── uv.lock
```