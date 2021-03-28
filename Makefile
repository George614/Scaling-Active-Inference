.PHONY: init interception_test

SHELL=/bin/bash

init:
	python3 -m venv venv
	source venv/bin/activate && pip install -r requirements.txt

interception_test:
	@source venv/bin/activate && python3 interception_py_env.py

interception-unity-env:
	unzip interception-unity-env.zip

interception_test: interception-unity-env
	@source venv/bin/activate && python3 interception_unity_env.py