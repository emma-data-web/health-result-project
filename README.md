# Healthcare Prediction API

A **FastAPI-based healthcare prediction API** that integrates multiple machine learning models to provide predictions for malaria, malaria-related mortality, diabetes, and other health conditions.

The application also provides basic user management, including registration, login, and profile retrieval.

## Features
#i used basic auth system here, if you wanna contribute or build on better integrations, i recommend tokens be used. 
* User registration and login
* User profile management
* Malaria prediction with confidence score
* Malaria mortality prediction
* Diabetes prediction
* General health/disease prediction
* Multiple trained machine learning models served through a single FastAPI application
* SQLAlchemy database integration
* Password hashing
* CORS support
* Automatic API documentation through FastAPI

## Tech Stack

* **Python**
* **FastAPI**
* **SQLAlchemy**
* **Pydantic**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Joblib**
* **python-dotenv**
* **Uvicorn**

## Project Structure

```text
project/
│
├── app/
│   ├── ...
│
├── real_malaria_model.pkl
├── malaria_death_mmodel.pkl
├── diabetes_model.pkl
├── health_disease_pipeline.joblib
├── label_encoder.joblib
│
├── main.py
├── requirements.txt
└── README.md
```

> The exact directory structure may vary depending on how the project is organized.

## Machine Learning Models

The API exposes four prediction services.

### 1. Malaria Prediction

**Endpoint:** `POST /malpredict`

The malaria model accepts patient information including:

* Age
* Body temperature
* Hemoglobin
* RBC count
* Platelet count
* Fever
* Chills
* Vomiting
* Rainy season

The endpoint returns the predicted result together with the model's confidence score.

Example response:

```json
{
  "Result": 1,
  "confidence": "the confidences level of the result is 87%"
}
```

### 2. Malaria Mortality Prediction

**Endpoint:** `POST /predict`

This endpoint uses a separate trained model to predict a malaria-related death outcome.

Example response:

```json
{
  "death": 0
}
```

### 3. Diabetes Prediction

**Endpoint:** `POST /diapredict`

The diabetes model uses features such as:

* Pregnancies
* Glucose
* Blood pressure
* Skin thickness
* Insulin
* BMI
* Diabetes pedigree function
* Age

Example response:

```json
{
  "Outcome": 1
}
```

### 4. General Health/Disease Prediction

**Endpoint:** `POST /healthpredict`

This model accepts a broader set of patient information, including:

* Age
* Gender
* Temperature
* Heart rate
* Blood pressure
* Glucose level
* Oxygen level
* BMI
* Cough
* Fatigue
* Headache
* Nausea
* Chest pain
* Shortness of breath
* Vision problems
* Frequent urination
* Joint pain

The model's encoded prediction is converted back to the corresponding disease label using a label encoder.

Example response:

```json
{
  "predicted_disease": "Diabetes"
}
```

## Authentication

The API includes a simple user management system.

### Register

**Endpoint:** `POST /signup`

Users can register with:

* Name
* Email
* Password
* Position
* Department

Passwords are hashed before being stored in the database.

### Login

**Endpoint:** `POST /login`

Users provide their email and password. The API verifies the credentials and returns a welcome message.

### Profile

**Endpoint:** `POST /profile`

Retrieves the user's profile using their email.

The profile contains:

```json
{
  "name": "John Doe",
  "email": "john@example.com",
  "position": "Doctor",
  "Department": "Health"
}
```

## Database

The application uses **SQLAlchemy** for database interaction.

The `users` table contains:

| Column            | Description          |
| ----------------- | -------------------- |
| `id`              | Primary key          |
| `name`            | User's name          |
| `email`           | Unique email address |
| `hashed_password` | Hashed user password |
| `position`        | User's position      |
| `department`      | User's department    |



Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Linux/macOS:

```bash
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the project root and add the required database/environment configuration.

Example:

```env
DATABASE_URL=your_database_url
```

Add any other environment variables required by your database configuration.

## Running the API

Start the FastAPI application with Uvicorn:

```bash
uvicorn main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

## API Documentation

FastAPI automatically provides interactive API documentation.

### Swagger UI

```text
http://127.0.0.1:8000/docs
```

### ReDoc

```text
http://127.0.0.1:8000/redoc
```

These interfaces can be used to test the API endpoints directly from the browser.

## API Endpoints

| Method | Endpoint         | Description                           |
| ------ | ---------------- | ------------------------------------- |
| POST   | `/signup`        | Register a new user                   |
| POST   | `/login`         | Authenticate a user                   |
| POST   | `/profile`       | Retrieve user profile                 |
| POST   | `/predict`       | Predict malaria mortality             |
| POST   | `/malpredict`    | Predict malaria and return confidence |
| POST   | `/diapredict`    | Predict diabetes outcome              |
| POST   | `/healthpredict` | Predict a general health condition    |

## How It Works

The application loads the trained machine learning models when the FastAPI application starts.

When a prediction request is received:

1. FastAPI validates the request using Pydantic.
2. The validated input is converted into a Pandas DataFrame.
3. The appropriate trained model receives the input.
4. The model generates a prediction.
5. The prediction is returned as a JSON response.

For the general health model, the numerical prediction is passed through a label encoder to obtain the corresponding disease name.

```text
Client
   │
   │ HTTP Request
   ▼
FastAPI
   │
   │ Pydantic validation
   ▼
Pandas DataFrame
   │
   ▼
Trained ML Model
   │
   │ Prediction
   ▼
JSON Response
```

## Important Note

The predictions generated by this API are **machine-learning predictions and should not be treated as medical diagnoses**. The system is intended for software, research, educational, or demonstration purposes and should not replace professional medical evaluation.


