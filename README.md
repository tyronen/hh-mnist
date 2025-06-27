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
uv run --group cuda124 create_composite_images.py
uv run --group cuda124 train_complex_model.py --entity wandb-team --project wandb-project
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

## Remote deploy

# Build and deploy instructions

Any hosting service accepting Docker will do. You need enough RAM for the model.

### Setup

1. Install Docker Engine on your server. Docker should be runnable by someone other than root. Follow the instructions
   on the Docker website.
2. Your server must have a static IP address.
3. Create an entry in your `~/.ssh/config` file, or use `ssh-agent` to alias a name to your server that incorporates its
   username and any private key needed to log into it.
4. `docker context create server-name --docker "ssh://server-name"`
