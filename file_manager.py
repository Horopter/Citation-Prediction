import os
import sys
import glob
import hashlib

# Configuration
CHUNK_SIZE = 1024 * 1024 * 1024  # 1GB in bytes

def calculate_sha256(file_path):
    """Calculates the SHA256 hash of a file efficiently."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Read in 64MB chunks to be memory efficient
        for byte_block in iter(lambda: f.read(67108864), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def split_file(file_path):
    """Splits a file into chunks in the same directory and saves its checksum."""
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return

    # Get absolute path to ensure we are working in the correct folder
    abs_path = os.path.abspath(file_path)
    file_dir = os.path.dirname(abs_path)
    file_name = os.path.basename(abs_path)
    file_size = os.path.getsize(abs_path)

    print(f"Processing '{file_name}'...")
    print(f"  Location: {file_dir}")
    print(f"  Size:     {file_size / (1024*1024*1024):.2f} GB")
    
    # 1. Calculate and save checksum of the ORIGINAL file
    print("  Calculating original checksum...")
    original_hash = calculate_sha256(abs_path)
    checksum_filename = f"{abs_path}.sha256"
    
    with open(checksum_filename, 'w') as f:
        f.write(original_hash)
    print(f"  Saved checksum to: {os.path.basename(checksum_filename)}")

    # 2. Split the file
    print(f"  Splitting into {CHUNK_SIZE/(1024*1024*1024):.0f}GB chunks...")
    with open(abs_path, 'rb') as f:
        part_num = 0
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            
            # Save part in the same directory as the original
            part_filename = f"{abs_path}.part{part_num:03d}"
            with open(part_filename, 'wb') as chunk_file:
                chunk_file.write(chunk)
            
            print(f"    -> Created: {os.path.basename(part_filename)}")
            part_num += 1
    
    print(f"Done. Parts are in '{file_dir}'.\n")

def combine_files(original_file_path):
    """Combines split parts from the folder and verifies integrity."""
    # Handle path resolution
    abs_path = os.path.abspath(original_file_path)
    file_dir = os.path.dirname(abs_path)
    file_name = os.path.basename(abs_path)
    
    # Look for parts in the specific folder
    parts = sorted(glob.glob(f"{abs_path}.part*"))
    checksum_file = f"{abs_path}.sha256"
    
    if not parts:
        print(f"No parts found for '{file_name}' in '{file_dir}'.")
        return

    print(f"Combining {len(parts)} parts for '{file_name}'...")
    
    sha256_hash = hashlib.sha256()

    with open(abs_path, 'wb') as output_file:
        for part in parts:
            print(f"  Reading {os.path.basename(part)}...")
            with open(part, 'rb') as part_file:
                while True:
                    bytes_read = part_file.read(64 * 1024 * 1024) 
                    if not bytes_read:
                        break
                    output_file.write(bytes_read)
                    sha256_hash.update(bytes_read)
    
    calculated_hash = sha256_hash.hexdigest()
    print(f"Reconstruction complete. Verifying integrity...")

    if os.path.exists(checksum_file):
        with open(checksum_file, 'r') as f:
            expected_hash = f.read().strip()
        
        if calculated_hash == expected_hash:
            print(f"✅ SUCCESS: Checksum matches for '{file_name}'")
        else:
            print(f"❌ ERROR: Checksum MISMATCH for '{file_name}'")
            print(f"   Expected: {expected_hash}")
            print(f"   Actual:   {calculated_hash}")
            print("   The file may be corrupted.")
    else:
        print(f"⚠️  WARNING: No checksum file found at '{os.path.basename(checksum_file)}'. Cannot verify integrity.")
    
    print("-" * 40)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage:")
        print("  Split:   python file_manager.py split <path/to/file1> ...")
        print("  Combine: python file_manager.py combine <path/to/original_file1> ...")
        sys.exit(1)

    action = sys.argv[1].lower()
    files = sys.argv[2:]

    for file_path in files:
        if action == "split":
            split_file(file_path)
        elif action == "combine":
            combine_files(file_path)
        else:
            print("Invalid action. Use 'split' or 'combine'.")
            break
