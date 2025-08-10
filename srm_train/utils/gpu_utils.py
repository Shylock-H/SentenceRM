try:
    import psutil
    import GPUtil

    import_packages = True
except ImportError:
    import_packages = False
    
def get_cpu_info():
    cpu_info = {
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "cpu_freq": psutil.cpu_freq(),
        "usage": psutil.cpu_percent(interval=1),
    }
    return cpu_info


def get_memory_info():
    memory_info = {
        "total_memory": f"{psutil.virtual_memory().total / (1024 ** 3):.2f} GB",
        "used_memory": f"{psutil.virtual_memory().used / (1024 ** 3):.2f} GB",
    }
    return memory_info


def get_gpu_info():
    gpu_info = []
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        info = {
            "name": gpu.name,
            "load": f"{gpu.load * 100}%",
            "temperature": f"{gpu.temperature} C",
            "free_memory": f"{gpu.memoryFree}MB",
            "used_memory": f"{gpu.memoryUsed}MB",
            "total_memory": f"{gpu.memoryTotal}MB",
        }
        gpu_info.append(info)
    return gpu_info