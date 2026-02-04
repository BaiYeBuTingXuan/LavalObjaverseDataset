# 1. Generate SSH key (if you don't have one)
ssh-keygen -t ed25519 -C "-renderer@$(hostname)" -f ~/.ssh/laval-objaverse.pub

# 2. Copy public key to remote server
ssh-copy-id -i ~/.ssh/laval-objaverse.pub hejun@10.21.4.23

mv ~/.ssh/laval-objaverse.pub pwd/
# 3. Update credentials.json to use key path instead of password:
{
  "remote_ip": "10.21.4.23",
  "remote_user": "render_user",
  "remote_key": "/home/hejun/.ssh/objaverse_render",
  "remote_path": "/data/objaverse_renders"
}