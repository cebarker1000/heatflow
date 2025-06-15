#!/bin/bash
set -eux

# Set micromamba root and add to PATH
export MAMBA_ROOT_PREFIX="$HOME/micromamba"
export PATH="$MAMBA_ROOT_PREFIX/bin:$PATH"

# Install micromamba if not already there
if [ ! -f ./bin/micromamba ]; then
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
fi

# Create the environment using the provided environment.yml
./bin/micromamba create -y -n codex-env -f environment.yml

# Hook into micromamba shell for activation
eval "$(./bin/micromamba shell hook -s bash)"
micromamba activate codex-env
