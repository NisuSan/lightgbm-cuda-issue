# Issue: CUDA Tree Learner Not Enabled in LightGBM Docker Image

## Description

When attempting to use LightGBM with CUDA in a Docker container, users may encounter an error stating: "CUDA Tree Learner was not enabled in this build. Please recompile with CMake option -DUSE_CUDA=1". This error indicates that the CUDA support was not properly enabled during the build process of the LightGBM library within the Docker image.

## Reproduction Steps

To reproduce this issue, follow these steps:

1. Clone the repository containing the Dockerfile with LightGBM installed.
2. Open a terminal and navigate to the directory containing the Dockerfile.
3. Build the Docker image by running the following command:

```bash
docker build .
```

4. Once the build is complete, run the Docker container with the following command:
```bash

docker run -p 8888:8888 -d --name lightgbm-gpu-cuda <hash_of_the_image>
```

5. Access Jupyter Notebook by opening a web browser and navigating to localhost:8888 and create a new notebook. In the notebook, write the following code snippet:

```python
import lightgbm as lgb
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=10_000)
dtrain = lgb.Dataset(X, label=y)
bst = lgb.train(
	params={
	   "objective": "regression",
	   "device": "cuda",
	   "verbose": 1
	},
	train_set=dtrain,
    num_boost_round=5
)
```
6. Execute the code cell.

### Expected Behavior

The code should execute without errors, and the LightGBM model should be trained using CUDA.

### Observed Behavior

An error message is displayed: "[LightGBM] [Fatal] CUDA Tree Learner was not enabled in this build. Please recompile with CMake option -DUSE_CUDA=1".