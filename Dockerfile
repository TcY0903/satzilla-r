FROM python:3.11-slim

WORKDIR /app

COPY data/feature_names.txt ./data/
COPY training_data/pairwise_forests.pkl ./training_data/
COPY model_training/ ./model_training/
COPY solvers/ ./solvers/
COPY requirements.txt .
COPY satzilla_r.py .

RUN pip install --no-cache-dir -r requirements.txt

RUN chmod +x \
    solvers/AE_kissat2025_MAB/build/kissat \
    solvers/cadical-sc2025/build/cadical \
    solvers/Dynamiccadical/dynamiccadical \
    solvers/IsaSAT/bin/isasat \
    solvers/yalsat/yalsat

ENTRYPOINT ["python", "satzilla_r.py"]