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

- In <a href="https://github.com/naomyduarteg/MLOps-heart_disease/tree/main/code">code</a>, we have the complete code where the dataset is analyzed and ML models are trained. The best one, which is Linear Regression, is choosen, trained and exported with joblib to be reproduced. 
- In <a href="https://github.com/naomyduarteg/MLOps-heart_disease/tree/main/data">data</a>, we have the dataset used in this project.
- In <a href="https://github.com/naomyduarteg/MLOps-heart_disease/tree/main/functions">functions</a>, we load the model using joblib and write the functions that will be used in the API to perform the predictions.
- In <a href="https://github.com/naomyduarteg/MLOps-heart_disease/tree/main/model_to_run">model_to_run</a>, we have the compressed joblib file of the model.
- In <a href="https://github.com/naomyduarteg/MLOps-heart_disease/tree/main/model_to_run">main.py</a>, we have the main file where FastAPI is initialized, the classes for the input and output data are written using pydantic and the endpoints are written. 

If you want to know more about FastAPI and the machine learning methods and metrics used in this project, take a look <a href="https://medium.com/@naomy-gomes">here</a> at my Medium webpage.
