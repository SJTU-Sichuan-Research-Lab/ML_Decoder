# Use an official Python runtime as a parent image
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

# Copy the current directory contents into the container at /app
COPY .. /app

# Set the working directory in the container
WORKDIR /app

# Set the CUDA_HOME environment variable
ENV CUDA_HOME=/usr/local/cuda-11.6
# https://blog.csdn.net/jiaoyangwm/article/details/134338386
# docker rm -f test && docker run -it --gpus all --name test mld python -c "import torch; print(torch.cuda.get_device_capability())"
ENV TORCH_CUDA_ARCH_LIST="8.0"
ENV IABN_FORCE_CUDA=1

# Optionally, set PATH and LD_LIBRARY_PATH to include CUDA binaries and libraries
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"


# Install any needed dependencies
RUN pip install -r requirements.txt
RUN pip install -r service/requirements.txt



# Run the Python script
#CMD ["python", "./your_script.py"]
