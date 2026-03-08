PYTHON = python
PYTEST = pytest

.PHONY: setup train evaluate test api dashboard docker-up docker-down clean

setup:
	pip install -r requirements.txt

train:
	$(PYTHON) -m src.train

evaluate:
	$(PYTHON) -m src.evaluate

test:
	$(PYTEST) tests/ -v --tb=short

api:
	uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

dashboard:
	streamlit run streamlit_app/dashboard.py

docker-up:
	docker-compose up --build -d

docker-down:
	docker-compose down

clean:
	rm -rf models/*.pkl models/*.joblib reports/figures/*.png mlruns/ __pycache__
