import subprocess

# Function to install a package using pip
def install_package(package):
    try:
        print(f"Installing package: {package}")
        # Attempt to install the package using pip
        subprocess.check_call([f"pip", "install", package])
    except subprocess.CalledProcessError:
        # If there's an error during installation, catch it and skip
        print(f"Failed to install: {package}, skipping...")

# Read the requirements.txt file
def install_requirements(requirements_file):
    with open(requirements_file, 'r') as file:
        # Loop through each line (package)
        for line in file:
            package = line.strip()  # Remove extra spaces/newlines
            if package:  # Ignore empty lines
                install_package(package)

# Provide the path to your requirements.txt file
install_requirements('4980_env.txt')
