# Makefile for Cal-C.Ai

init:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

migrate:
	alembic upgrade head

run:
	uvicorn main:app --reload --host 127.0.0.1 --port 8000

docker-build:
	docker build -t cal-cai:latest .

docker-run:
	docker run --rm --env-file .env -p 8000:8000 cal-cai:latest

