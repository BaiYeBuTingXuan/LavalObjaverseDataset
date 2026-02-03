# 1. Generate SSH key (if you don't have one)
ssh-keygen -t ed25519 -C "-renderer@$(hostname)" -f ~/.ssh/objaverse_render

# 2. Copy public key to remote server
ssh-copy-id -i ~/.ssh/objaverse_render.pub user@192.168.1.100

# 3. Update credentials.json to use key path instead of password:
{
  "remote_ip": "192.168.1.100",
  "remote_user": "render_user",
  "remote_key": "/home/hejun/.ssh/objaverse_render",
  "remote_path": "/data/objaverse_renders"
}