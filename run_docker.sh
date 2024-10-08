docker run --rm -it --gpus all -v $(pwd):/app -v /mnt/nas:/mnt/nas --network host bic4907/pcgrl:latest /bin/bash
