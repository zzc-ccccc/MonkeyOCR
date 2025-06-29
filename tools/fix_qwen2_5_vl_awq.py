#!/usr/bin/env python3
import os
import shutil
import sys

def find_lmdeploy_calibrate_file():
    """Automatically find the lmdeploy calibrate.py file in current environment"""
    try:
        import lmdeploy
        lmdeploy_path = os.path.dirname(lmdeploy.__file__)
        calibrate_file = os.path.join(lmdeploy_path, 'lite', 'apis', 'calibrate.py')
        
        if os.path.exists(calibrate_file):
            return calibrate_file
        else:
            print(f"Error: calibrate.py file not found, expected path: {calibrate_file}")
            return None
    except ImportError:
        print("Error: lmdeploy is not installed in current environment")
        return None

def main():
    print("Auto-detecting lmdeploy installation path...")
    
    # Automatically find calibrate.py file
    calibrate_file = find_lmdeploy_calibrate_file()
    if not calibrate_file:
        sys.exit(1)
    
    print(f"Found file: {calibrate_file}")
    
    # Backup original file
    backup_file = calibrate_file + ".backup"
    if not os.path.exists(backup_file):
        shutil.copy(calibrate_file, backup_file)
        print("✓ Original file backed up")
    else:
        print("✓ Backup file already exists, skipping backup")

    # Read file content
    try:
        with open(calibrate_file, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error: Failed to read file - {e}")
        sys.exit(1)

    # Check if already modified
    if "# if hasattr(vl_model, 'language_model')" in content:
        print("✓ File already modified, no need to modify again")
        return

    # Replace problematic code
    old_code = """        if hasattr(vl_model, 'language_model'):  # deepseek-vl, ...
            model = vl_model.language_model
        if hasattr(vl_model, 'llm'):  # MiniCPMV, ...
            model = vl_model.llm"""

    new_code = """        # if hasattr(vl_model, 'language_model'):  # deepseek-vl, ...
        #     model = vl_model.language_model
        # if hasattr(vl_model, 'llm'):  # MiniCPMV, ...
        #     model = vl_model.llm"""

    if old_code in content:
        content = content.replace(old_code, new_code)
        
        # Write back to file
        try:
            with open(calibrate_file, 'w', encoding='utf-8') as f:
                f.write(content)
            print("✓ Fix completed!")
        except Exception as e:
            print(f"Error: Failed to write file - {e}")
            sys.exit(1)
    else:
        print("⚠ Expected code snippet not found, lmdeploy version might be different")
        print("Please check the file content manually")

    print("\n🎉 Now you can run Qwen2.5-VL AWQ quantization!")
    print("Use command:")
    print("lmdeploy lite auto_awq \\")
    print("    ./model_weight/Recognition \\")
    print("    --calib-dataset 'ptb' \\")
    print("    --calib-samples 64 \\")
    print("    --calib-seqlen 1024 \\")
    print("    --w-bits 4 \\")
    print("    --w-group-size 128 \\")
    print("    --batch-size 1 \\")
    print("    --work-dir ./monkeyocr_quantization")

if __name__ == "__main__":
    main()
