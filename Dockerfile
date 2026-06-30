# Use a lightweight PyTorch image with CUDA 12.1 and Python 3.10
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies (git is needed by Hugging Face datasets/transformers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install JupyterLab
RUN pip install --no-cache-dir jupyterlab

# Copy application source code, notebooks, and configurations
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY configs/ ./configs/
COPY ReLU_Tune_train.ipynb ./

# Set Python path to ensure src modules can be resolved
ENV PYTHONPATH=/app

# Expose Jupyter port
EXPOSE 8888

# Launch JupyterLab by default (with tokens disabled for easy local access)
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]
