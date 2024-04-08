import torch

def check_cuda():
    """ 
    Check the CUDA availability and proper installation with getting the device ID and name 
    """
    
    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        print(f"The {torch.cuda.get_device_name(device_id)} is being used for GPU unit")
        return True, device_id
    return False, -1