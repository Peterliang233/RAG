make_requirements:
	pipreqs . --encoding=utf8 --force
download_dependencies:
	pip install -r requirements.txt