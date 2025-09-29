# Efficient-HA




```
conda create -n EHA python==3.10
cd transformers
pip install -e .
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
pip install flash-attn==2.7.1.post4  --no-build-isolation
```