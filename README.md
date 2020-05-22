# cat-AI-log. An AI-based product group allocation system

Capstone project.

Sebastian Thomas @ neue fische Bootcamp Data Science<br />
(datascience at sebastianthomas dot de)

When ordering medicines, hospitals have to deal with a multitude of different article descriptions for identical products. With cat-AI-log, article duplicates and similar articles can be found and articles can be catalogued in product groups using human-assisted artificial intelligence.

![cat-AI-log presentation][image]

[image]: cat-AI-log.png "cat-AI-log presentation"


## Results

cat-AI-log...
- finds similar articles in different forms
- recognizes dosage forms and physical quantities
- handles spelling mistakes of user
- handles spelling mistakes in data
- allocates known and similar articles correctly in many cases


## Content

Main analysis:
- [Part 1: Data mining](cat-AI-log_1_data_mining.ipynb)
- [Part 2: Data preprocessing](cat-AI-log_2_data_preprocessing.ipynb) ([Data cleaning](transformer/cleaning.py), [Feature engineering](transformer/feature_engineering.py))
- [Part 3: Exploratory data analysis](cat-AI-log_3_exploratory_data_analysis.ipynb)
- [Part 4: Predictive analysis](cat-AI-log_4_predictive_analysis.ipynb)
- [Part 5: The search engine](cat-AI-log_5_search_engine.ipynb)
- [Part 6: Visualization](cat-AI-log_6_visualization.ipynb)

Side path Data Mining:
- [Mining 1: pdf mining of IFA dosage forms](mining/cat-AI-log_mining_1_ifa_dosage_forms.ipynb)
- [Mining 2: html mining of DocMorris dosage forms](mining/cat-AI-log_mining_2_docmorris_dosage_forms.ipynb)
- [Mining 3: Construction of a replacement dictionary from DocMorris to IFA](mining/cat-AI-log_mining_3_replacement_docmorris_ifa.ipynb)

Main development:
- [Spelling correction](modules/spelling_correction.py)
- [Search](modules/search.py)
- [Quotient Extraction](modules/quotient_extraction.py)
- [Helper](modules/ds.py)
- [Another rough helper](modules/ds_rough.py)

Demo web frontend:
- [Flask run file](web/run.py)
- [HTML](web/templates/index.html)
- [CSS](web/static/style.css)

Due to publication restrictions, the data and the output of the project are not provided.


## Future Work

- iterative process of AI and specialist will improve data
- this in turn will improve AI
- ordering of search results could be improved
- with more data (e.g. active ingredients), recommender for generics could be built
- let cat-AI-log learn from former spelling mistakes to improve performance