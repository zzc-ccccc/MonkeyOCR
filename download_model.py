from argparse import ArgumentParser
import os


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--type', '-t', type=str, default="huggingface") #modelscope
    args = parser.parse_args()
    if not os.path.exists("./model_weight"):
        os.makedirs("./model_weight")
    if args.type == "huggingface":
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id="echo840/MonkeyOCR",local_dir="./model_weight",local_dir_use_symlinks=False, resume_download=True)
    elif args.type == "modelscope":
        from modelscope import snapshot_download as modelscope_download
        modelscope_download(repo_id = 'l1731396519/MonkeyOCR',local_dir="./model_weight")
