FROM databricksruntime/air:dcs-base-aws-runtime

RUN python3 -m pip install --no-cache-dir --break-system-packages \
    torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0

COPY train.py /app/train.py
WORKDIR /app
