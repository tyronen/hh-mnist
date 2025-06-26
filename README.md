# hh-mnist
Hyperparameter Hippies MNIST project


## Tyrone and Dan's implementation

```
# on local machine
./send.sh
# on Nvidia machine
./ssh.sh
uv run train_model.py # encoder-only
uv run train_complex_model.py # encoder-decoder
```

Data will be placed in the `data` directory.

To run the webserver:

```
uv run --group inference -- streamlit run webserver.py
```