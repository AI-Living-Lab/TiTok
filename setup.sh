# rclone
export PATH="/workspace/home:$PATH"
export RCLONE_CONFIG="/workspace/home/rclone.conf"

if [ ! -f "/workspace/home/rclone" ]; then
  echo "[setup] Installing rclone to /workspace/home..."
  mkdir -p /workspace/home
  curl https://rclone.org/install.sh | bash
  mv /usr/bin/rclone /workspace/home/rclone
  echo "[setup] rclone installed. Run 'rclone config' to set up Google Drive."
fi

# Miniconda
export CONDA_HOME="/workspace/home/miniconda3"
export PATH="$CONDA_HOME/bin:$PATH"

if [ -f "$CONDA_HOME/etc/profile.d/conda.sh" ]; then
  . "$CONDA_HOME/etc/profile.d/conda.sh"
fi


export WANDB_ENTITY="guma017-ewha-womans-university"
export WANDB_API_KEY="<YOUR_WANDB_API_KEY>"   # https://wandb.ai/authorize 에서 발급 (커밋 금지)
