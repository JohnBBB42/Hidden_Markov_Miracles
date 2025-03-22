# Hidden Markov Neural Networks

A repository for developing Hidden Markov Neural Networks

# Instructions to run model
First git clone the repository.

Then run:

`pip install -e .`

Then run:

`python src/hidden_markov_neural_network/train.py` to train the initial training for the HMNN
`python src/hidden_markov_neural_network/train_nn.py` to train the initial training for the MLP
`python src/hidden_markov_neural_network/incremental_update_and_inference.py` to do the incremental updates, where the year interval is changed in the script
`python src/hidden_markov_neural_network/evaluate.py` to do evaluate the HMNN
`python src/hidden_markov_neural_network/evaluate_nn.py` to do evaluate the MLP, settings can be changed in the script
Hyperparameters and path are changed in the configs/config.yaml file

## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
