from huggingface_hub import snapshot_download
import os

if __name__ == '__main__':
    if not os.path.exists("./model_weight"):
        os.makedirs("./model_weight")
    snapshot_download(repo_id="echo840/MonkeyOCR",local_dir="./model_weight",local_dir_use_symlinks=False, resume_download=True)
