## Running tests locally

1. Ensure that you have [tox](https://tox.wiki/en/4.24.1/) inside your Python environment, i.e. `pip install tox`.
2. Run `tox` in the root directory of the repository.
3. The tests will run and the results will be displayed in the terminal.

## A note on the dummy models

Some of the tests require some dummy models to be present in the `tests/dummy_models` directory. Dummy models were generated using [the create_dummy_models script](https://github.com/huggingface/transformers/blob/main/utils/create_dummy_models.py) and placed in the `tests/dummy_models` directory. If you want to regenerate a dummy model from scratch: 

1. git clone the transformers repository: `git clone https://github.com/huggingface/transformers/tree/main`
2. install transformers in the development mode: `pip install -e transformers`
3. ensure to have the `torch` and `tensorflow` dependencies installed: `pip install torch tensorflow` 
4. ensure to have pytest installed: `pip install pytest`
5. run the script: `python utils/create_dummy_models.py --model_types <INSERT VALID MODEL NAME> $MODEL_DIRECTORY`