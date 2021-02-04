.PHONY: init interception_test

SHELL=/bin/bash

init:
	python3 -m venv venv
	source venv/bin/activate && pip install -r requirements.txt

interception_test:
	source venv/bin/activate && python3 interception_env.py