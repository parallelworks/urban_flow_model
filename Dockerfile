FROM tensorflow/tensorflow:latest-gpu-jupyter
RUN pip install pandas scikit-learn
ENTRYPOINT [ "/bin/bash" ]