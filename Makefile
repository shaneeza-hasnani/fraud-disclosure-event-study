.PHONY: setup run clean

setup:
	pip install -r requirements.txt

run:
	jupyter nbconvert --to notebook --execute notebooks/event_study_analysis.ipynb --output event_study_analysis.ipynb

clean:
	rm -f output/*.csv output/figures/*.png
