# Running with WSL2 + Docker Desktop

## Installing WSL2 and Docker Desktop

First, ensure your version of Windows supports WSL2.

1.  Enable WSL.  
    Launch PowerShell with administrator privileges.
    * Enable the Virtual Machine Platform feature.
    ```PowerShell
    dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
    ```
    * Enable the Windows Subsystem for Linux feature.
    ```PowerShell
    dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
    ```
    * Restart your computer.

2.  Install a Linux distribution.
   ```PowerShell
   wsl --install -d Ubuntu
   ```

3.  Download Docker Desktop.  
    [Official Docker Website](https://www.docker.com/products/docker-desktop/)

4.  Configure WSL config (Optional).  

    If you need to quantize the model later, you might encounter issues with insufficient memory.

    * Open your user profile folder.
    * Enter `%UserProfile%` in the File Explorer address bar and press Enter.
    * Create a `.wslconfig` file.
    * Edit the file content with a code editor.
    ```.wslconfig
    [wsl2]
    memory=24GB
    ```
    Other parameters can be set as needed.

## Building the Container
1. Run Docker Desktop.

2. Enter WSL.
    ```PowerShell
    wsl
    cd ~
    ```
3. Clone the repository.
    ```PowerShell
    git clone https://github.com/Yuliang-Liu/MonkeyOCR

    cd MonkeyOCR
    ```
4. Follow the 'Docker Deployment' section in the [README.md](../README.md) file to create the Docker image.

After entering the container, you can run MonkeyOCR normally.

You can use the `Dev Containers extension` in VS Code to connect to the container for convenient editing and modification.

If you encounter the error `RuntimeError: No enough gpu memory for runtime.`, it indicates insufficient VRAM. You can try quantizing the model.
For details, see [Quantization Method](Quantization.md).