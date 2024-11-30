import os

def create_image_processing_template(base_dir):
    # Define directory structure
    structure = {
        "data": ["raw", "processed", "output"],  # Data directories
        "src": ["preprocessing", "models", "utils", "visualization"],  # Source code directories
        "notebooks": [],  # Jupyter notebooks
        "tests": [],  # Unit tests
        "docs": [],  # Documentation
    }

    # Define main files
    main_files = {
        "README.md": "# Image Processing Project\n\nThis project is for processing and analyzing images.",
        "requirements.txt": "# Add your Python dependencies here\nnumpy\nopencv-python\nscikit-image\nmatplotlib\npillow",
        ".gitignore": "# Ignore unnecessary files\n__pycache__/\n*.pyc\n.env\n*.ipynb_checkpoints/\ndata/raw/",
    }

    # Create the base directory
    os.makedirs(base_dir, exist_ok=True)

    # Create directories and subdirectories
    for dir_name, sub_dirs in structure.items():
        base_path = os.path.join(base_dir, dir_name)
        os.makedirs(base_path, exist_ok=True)
        for sub_dir in sub_dirs:
            os.makedirs(os.path.join(base_path, sub_dir), exist_ok=True)

    # Create main files
    for file_name, content in main_files.items():
        file_path = os.path.join(base_dir, file_name)
        with open(file_path, "w") as file:
            file.write(content)

    # Create a sample Python script in src
    with open(os.path.join(base_dir, "src", "main.py"), "w") as file:
        file.write("""\
# Main script for the image processing project

import os

def main():
    print("Welcome to the Image Processing Project!")

if __name__ == "__main__":
    main()
""")

    print(f"Image processing project template created at {base_dir}")

# Example usage
if __name__ == "__main__":
    create_image_processing_template(".")

