# Heart Disease Machine Learning Model
<img src="https://img.shields.io/badge/Python-14354C?style=for-the-badge&logo=python&logoColor=white"/> <img src="https://img.shields.io/badge/fastapi-109989?style=for-the-badge&logo=FASTAPI&logoColor=white"/> <img src="https://img.shields.io/badge/Docker-2CA5E0?style=for-the-badge&logo=docker&logoColor=white"/>

![machine-learning-logistic-regression](https://user-images.githubusercontent.com/73078250/201484493-06ce7a2b-3ea1-4ae9-a561-518f722b7d0f.svg)

The purpose of this project is to build a ML model with Python's scikit-learn, create an API with FastAPI and containerize with Docker.

Structure of the project:

<pre>
<code>
├── MLOps-heart_disease
│   ├── code
│   │    ├── complete_code.py
│   │    └── training.py
│   │        
│   ├── data   
│   │    └── heart.csv     
│   │    
│   ├── functions 
│   │    ├── __init__.py
│   │    └── functions.py   
│   │        
│   ├── model_to_run         
│   │    └── model.dat.gz          
│   │           
│   ├── __init__.py
│   ├── main.py
│   ├── Dockerfile
│   ├── README.md
│   └── requirements.txt
</code>
</pre>

- In <a href="https://github.com/naomyduarteg/MLOps-heart_disease/tree/main/code">code</a>, we have the complete code where columns of the dataset are described, the dataset is analyzed and ML models are trained. The best one, which is Linear Regression, is choosen, trained and exported with joblib to be reproduced. 
- In <a href="https://github.com/naomyduarteg/MLOps-heart_disease/tree/main/data">data</a>, we have the dataset used in this project.
- In <a href="https://github.com/naomyduarteg/MLOps-heart_disease/tree/main/functions">functions</a>, we load the model using joblib and write the functions that will be used in the API to perform the predictions.
- In <a href="https://github.com/naomyduarteg/MLOps-heart_disease/tree/main/model_to_run">model_to_run</a>, we have the compressed joblib file of the model.
- In <a href="https://github.com/naomyduarteg/MLOps-heart_disease/tree/main/model_to_run">main.py</a>, we have the main file where FastAPI is initialized, the classes for the input and output data are written using pydantic and the endpoints are written. 

## Running the API

1. Clone the repository

```
git@github.com:naomyduarteg/MLOps-heart_disease.git
```
2. Create a virtual environment

```
python3 -m venv <name_of_venv>
```
3. Go to the virtual environment's directory and activate it

For Windows:
```
Scripts/activate
```
For Linux/Mac:
```
bin/activate
```
4. Install the requirements

```
pip install -r requirements.txt
```

6. Run the API with uvicorn

```
uvicorn main:app --reload
```

From this point, one can use the Swagger documentation to test the API. 

## Docker 
We can use the <a href="https://github.com/naomyduarteg/MLOps-heart_disease/blob/main/Dockerfile">Dockerfile</a> to create an image for running our application inside a container

```
docker build . -t MLOps-heart_disease
```
And to run

```
docker run -p 8000:8000 MLOps-heart_disease
```
Now we have an API to predict heart disease deployed with Docker!

If you want to know more about FastAPI and the machine learning methods and metrics used in this project, take a look <a href="https://medium.com/@naomy-gomes">here</a> at my Medium webpage.
