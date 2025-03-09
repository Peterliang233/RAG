make_requirements:
	pipreqs ./ --encoding=utf-8 --force
download_dependencies:
	pip install -r requirements.txt