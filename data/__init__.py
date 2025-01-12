import platform
from pathlib import Path

# SET SOURCE PATHS HERE:
def get_data_path():
    """
    Returns the data path based on the operating system.
    """
    if platform.system().lower() == 'windows':
        data_path = Path('C:/', 'Users', 'SIEGLEW', 'OneDrive - Daimler Truck', 'MA', 'Code', 'MA-Data')
    elif platform.system().lower() == 'linux':
        data_path = Path('/mnt', 'nvme', 'datasets', 'sieglew') #/mnt/nvme/datasets/sieglew
    else: 
        print("Running on an unsupported platform")
        data_path = None
        
    return data_path