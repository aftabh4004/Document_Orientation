
### Triton Docker image

Pull the triton docker image from docker hub

```
sudo docker pull nvcr.io/nvidia/tritonserver:24.05-py3

```

### Export the model

```
# replace the 'model_path' with the state dict of your model in export.py

python3 export.py
```

### Run Triton server

```
sudo docker run --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -v ${PWD}/model_repository:/models nvcr.io/nvidia/tritonserver tritonserver --model-repository=/models
```


### Make Query

```
python3 client.py <path to image or pdf file>
```
