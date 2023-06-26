# model-service
Contains the wrapper service for the ML model

## How To Run It

#### Clone

Clone this repo to your local machine using 
```
git clone https://github.com/remla23-team15/model-service.git
```

#### Create Virtual Environment (venv)
Move to  the application folder and run in your terminal:
```
# Create virtualenv, make sure to use python3.8
$ virtualenv -p python3 venv
# Activate venv
$ source venv/bin/activate
```
Alternatively:
* Open the project with PyCharm (either Pro or CE)  or your favorite Python IDE
* Select python (>= 3.8) as project interpreter

#### Install Requirements
Move to  the application folder and run in your terminal:
```
pip install -r requirements.txt
```

#### Get The ML Models
Move to  the application folder and run in your terminal:
```
python get_ml_models.py
```

In case of problems, you can get the models from:
- The models repository https://liv.nl.tab.digital/s/TyPqR5HCjExqNQq
- The repository https://github.com/remla23-team15/model-training stored in the folder `ml_models`. 


Once downloaded, add them to a folder named `ml_models` in this project.

#### Run
Move to  the application folder and run in your terminal:
```
python app.py
```

The script will start a Flask server accessible at http://localhost:8080.

To access the Swagger API documentation, go to http://localhost:8080/apidocs.

#### Lint
To assess the code quality using PyLint with dslinter plugin, execute the following commands:
``` 
# Install CI requirements 
pip install -r requirements-ci.txt

# Execute pylint on the application files
pylint app.py get_ml_models.py
```

## Docker
To build and run a Docker image of the application, you can open the terminal (in the application folder) and run:
```shell script
docker build -t ghcr.io/remla23-team15/model-service:VERSION .

docker run -it --rm -p 8080:8080 ghcr.io/remla23-team15/model-service:VERSION
```

**VERSION indicates the version that you want to apply to the Docker image, for example 1.0.0, latest or so.**

## Contributors

REMLA 2023 - Group 15
