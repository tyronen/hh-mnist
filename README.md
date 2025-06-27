# hh-mnist
Hyperparameter Hippies MNIST project


## Tyrone and Dan's implementation

```
# on local machine
./send.sh
# on Nvidia machine
source ssh.sh
# encoder-only
uv run train_model.py --entity wandb-team --project wandb-project
# encoder-decoder
uv run create_composite_images.py
uv run train_complex_model.py --entity wandb-team --project wandb-project
```

Data will be placed in the `data` directory.

To run the webserver:

```
uv run --group inference -- streamlit run webserver.py
```

To run the slides

```
# must already have Node.js installed
npm install -g pnpm 
pnpm install
pnpm dev
```