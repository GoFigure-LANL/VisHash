install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt


test:
	#python -m pytest --vv --cov=util test_vishash.py
	python -m pytest --cov=vishash test_vishash.py


format:
	black *.py


lint:
	pylint --disable=R,C *.py


all: install lint test
