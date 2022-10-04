FROM python:3.9

RUN pip install -U pip poetry streamlit

WORKDIR /app

# Copy data, app and dependencies
COPY ./dashboard.py ./dashboard.py
COPY ./pyproject.toml ./pyproject.toml

# Install dependencies
RUN poetry install

ENTRYPOINT [ "streamlit", "run", "dashboard.py" ] 