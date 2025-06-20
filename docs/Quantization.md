# Quantization with AWQ

1.  Install the required packages.
    ```bash
    pip install datasets
    ```

2.  Enter the following in the terminal.
    ```bash
    lmdeploy lite auto_awq \
        ./model_weight/Recognition \
        --calib-dataset 'ptb' \
        --calib-samples 64 \
        --calib-seqlen 1024 \
        --w-bits 4 \
        --w-group-size 128 \
        --batch-size 1 \
        --work-dir ./monkeyocr_quantization
    ```
    Wait for the quantization to complete.
    * If the quantization process is killed, you need to check if you have sufficient memory.
    * For reference, the maximum VRAM usage for quantization with these parameters is approximately 6.47GB.

3.  You might encounter the following error:
    ```
    RuntimeError: Error(s) in loading state_dict for Linear:
        size mismatch for bias: copying a param with shape torch.Size([2048]) from checkpoint, the shape in current model is torch.Size([1280]).
    ```
    This is because your installed version of LMDeploy is not yet compatible with Qwen2.5VL. You need to install the latest development version from the GitHub repository.
    ```bash
    pip install git+https://github.com/InternLM/lmdeploy.git
    ```
    After the installation is complete, try quantizing again.

4.  After quantization is complete, replace the `Recognition` folder.
    ```bash
    mv model_weight/Recognition Recognition_backup

    mv monkeyocr_quantization model_weight/Recognition
    ```
    Then, you can try running the program again.
