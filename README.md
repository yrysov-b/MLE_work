## Project structure:

This project has a modular structure, where each folder has a specific duty.

```
MLE_basic_example
├── data                      
│   ├── inference_data.csv
│   └── train_data.csv
├── data_process              # Scripts used for data processing
│   ├── data_generation.py
│   └── __init__.py           
├── inference                 # Scripts and Dockerfiles used for inference
│   ├── Dockerfile
│   ├── run.py
│   └── __init__.py
├── models                    # Folder where trained models are stored
│   └── model.pickle
├── training                  # Scripts and Dockerfiles used for training
│   ├── Dockerfile
│   ├── train.py
│   └── __init__.py
├── utils.py                  # Utility functions and classes that are used in scripts
├── settings.json             # All configurable parameters and settings
└── README.md
```

## Data Generation: 

- Run the following script to generate data: 

```bash
python3 data_process/data_generation.py
```

 ## Training:

- Build the training Docker image:
```bash
 docker build -f ./training/Dockerfile -t training_image .
```
 - Run the training Docker image: 
```bash
 docker run -it training_image /bin/bash
```

- Then, move the trained model from the directory inside the Docker container `/app/models` to the local machine using:
```bash
docker cp <container_id>:/app/models/model.pickle ./models
```
Replace `<container_id>` with your running Docker container ID


- Copy training logs directory to local machine: 
```bash
docker cp <container_id>:/app/results/ ./results
```

Replace `<container_id>` with your running Docker container ID

## Inference:

- Build the inference Docker image:
```bash
 docker build -f ./inference/Dockerfile -t inference_image .
 ```

 - Run the inference Docker container:
```bash
 docker run -it inference_image /bin/bash 
 ```

- Copy results directory to local machine: 
```bash
docker cp <container_id>:/app/results/ ./results
```
Replace `<container_id>` with your running Docker container ID
