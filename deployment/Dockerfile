FROM rayproject/ray:nightly-extra

WORKDIR /app

RUN sudo apt-get update && sudo apt-get upgrade -y && \
    sudo apt-get install -y fonts-dejavu-core libgl1 libglib2.0-0 && \
    sudo rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir numpy==1.26.4 pandas python-multipart ultralytics pillow httpx timm
RUN pip install --no-cache-dir torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu126

EXPOSE 8000 8265 10003

CMD ray start \
    --head \
    --node-ip-address=127.0.0.1 \
    --dashboard-host=0.0.0.0 \
    --num-gpus=1 \
 && tail -f /dev/null
