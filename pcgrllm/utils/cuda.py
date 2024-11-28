import os

def get_cuda_version():
    try:
        cuda_version = os.popen("nvcc --version").read()
        for line in cuda_version.split("\n"):
            if "release" in line:
                version = line.split("release")[-1].strip().split(" ")[0]
                version = version.replace(",", "")
                return float(version)
        return False
    except Exception as e:
        print(f"Error getting CUDA version: {e}")
        return False

if __name__ == "__main__":
    print(get_cuda_version())
