# Sebastian Thomas (datascience at sebastianthomas dot de)

# coding=utf-8
# webapp
from flask import Flask, render_template, request, make_response

# reload developed objects
from joblib import load

# data
import numpy as np
import pandas as pd

# machine learning
from sklearn.pipeline import make_pipeline

# custom modules
from transformer.cleaning import clean_mira
from transformer.feature_engineering import extract_features_mira

app = Flask(__name__)

mira = pd.read_pickle('../data/mira_processed.pickle')
mira['certainty print'].replace({
    'very uncertain': 'sehr unsicher',
    'uncertain': 'unsicher',
    'certain': 'sicher',
    'very certain': 'sehr sicher',
    'confirmed': 'bestätigt'
}, inplace=True)
corpus = mira['article']

expansions_df = pd.read_csv('../data/abbreviations.csv', sep=';',
                            index_col='abbreviation')
expansions_df.set_index(expansions_df.index.str.replace(
    r'\.$', '', regex=True).str.lower(), inplace=True)
expansions = dict(expansions_df['expansion'])

replacements_dm_ifa_df = pd.read_csv('../data/replacements_dm_ifa.csv',
                                     sep=';', index_col='abbreviation_dm')
replacements_dm_ifa_df.set_index(replacements_dm_ifa_df.index.str.replace(
    r'\.$', '', regex=True).str.lower(), inplace=True)
replacements_dm_ifa = dict(replacements_dm_ifa_df['abbreviation_ifa'])

replacements_dosage_form_abbreviation_df \
    = pd.read_csv('../data/dosage_forms_ifa.csv', sep=';',
                  index_col='dosage form')
replacements_dosage_form_abbreviation_df.set_index(
    replacements_dosage_form_abbreviation_df.index.str.replace(
        r'\.$', '', regex=True).str.lower(), inplace=True)
replacements_dosage_form_abbreviation \
    = dict(replacements_dosage_form_abbreviation_df['abbreviation'])
replacement_dicts = [expansions, replacements_dm_ifa,
                     replacements_dosage_form_abbreviation]

dosage_forms_ifa = pd.read_csv('../data/dosage_forms_ifa.csv', sep=';',
                               index_col='abbreviation')
manufacturers = pd.read_csv('../data/manufacturers.csv', sep=';')
laws = ['11.3', '73.3', '116', '116b', '129', '129a']
token_lists = [dosage_forms_ifa.index, manufacturers['manufacturer'], laws]
token_feature_names = ['dosage form', 'manufacturer', 'law']

dosage_forms_dict = dosage_forms_ifa['dosage form'].to_dict()

spelling_corrector = load('../objects/spelling_corrector.joblib')
search_engine = load('../objects/search_engine.joblib')

preprocessor = load('../objects/preprocessor.joblib')
classifier = load('../objects/classifier.joblib')
classifier = make_pipeline(preprocessor, classifier)
label_encoder = load('../objects/label_encoder.joblib')

# include_all = False
# correct_spelling = True

translation = {
    'dosage form': 'Darreichungsform',
    'manufacturer': 'Hersteller',
    'law': 'Gesetz',
    'mass concentration 0 [mg/ml]': 'Massenkonzentration [mg/ml]',
    'mass puff concentration 0 [mg/hub]': 'Massenhubkonzentration [mg/hub]',
    'mass flow 0 [mg/h]': 'Massenfluss [mg/h]',
    'volume flow 0 [ml/h]': 'Volumenfluss [ml/h]',
    'active ingredient percentage 0 [%]': 'Wirkstoffanteil [%]',
    'mass 0 [mg]': 'Masse [mg]',
    'volume 0 [ml]': 'Volumen [ml]',
    'count puff 0 [hub]': 'Sprühstoßanzahl [hub]',
    'percentage 0 [%]': 'Anteil [%]',
    'length 0 [cm]': 'Länge 1 [cm]',
    'length 1 [cm]': 'Länge 2 [cm]',
    'length 2 [cm]': 'Länge 3 [cm]',
    'count 0': 'Anzahl'
}


@app.route("/")
def retrieve_articles(query=None, include_all=False, correct_spelling=True,
                      prediction=None, extracted_features=None,
                      recommendations=None):

    return render_template('index.html', query=query, include_all=include_all,
                           correct_spelling=correct_spelling,
                           prediction=prediction,
                           extracted_features=extracted_features,
                           recommendations=recommendations,
                           dosage_forms_dict=dosage_forms_dict)


@app.route("/", methods=['POST'])
def create_data():
    query = request.form.get('query')
    include_all = request.form.get('include_all') == 'y'
    correct_spelling = request.form.get('correct_spelling') == 'y'

    corrected_query \
        = spelling_corrector._transform(query) if correct_spelling else query
    preprocessed_query = preprocessor.transform([corrected_query])[0]

    prediction = label_encoder.inverse_transform(
        classifier.predict([preprocessed_query]))[0]
    certainty = np.max(classifier.predict_proba([preprocessed_query]))
    if certainty >= 0.8:
        certainty = 'sehr sicher'
    elif certainty >= 0.5:
        certainty = 'sicher'
    elif certainty >= 0.2:
        certainty = 'unsicher'
    elif certainty >= 0:
        certainty = 'sehr unsicher'
    prediction = (prediction, certainty)

    extracted_features = dict(
        extract_features_mira(clean_mira(
            [query], replacement_dicts=replacement_dicts),
            token_lists, token_feature_names).iloc[0])
    extracted_features = [(translation.get(item[0], 'andere'), item[1])
                          for item in extracted_features.items()
                          if not(type(item[1]) == np.float64
                                 and np.isnan(item[1]) or item[1] is pd.NA
                                 or item[0]
                                 in ['manufacturer article number',
                                     'additional fee', 'ambulant',
                                     'stationary'])]

    indices = search_engine.recommend(corrected_query, max_count=None,
                                      output='indices',
                                      correct_spelling=correct_spelling,
                                      include_all=include_all)
    recommendations = mira.iloc[indices][['article', 'prediction print',
                                          'certainty print']].values

    return make_response(retrieve_articles(query=query,
                                           include_all=include_all,
                                           correct_spelling=correct_spelling,
                                           prediction=prediction,
                                           extracted_features=extracted_features,
                                           recommendations=recommendations))
    # return render_template('index.html', query=query, include_all=include_all,
    #                        correct_spelling=correct_spelling,
    #                        prediction=prediction,
    #                        extracted_features=extracted_features,
    #                        recommendations=recommendations,
    #                        dosage_forms_dict=dosage_forms_dict)


# @app.route("/")  # , methods=['GET', 'POST'])
# def run():
#     global include_all
#     global correct_spelling
#
#     #if request.method == 'POST':
#     #    query = request.form.get('query')
#
#     query = request.args.get('query')
#
#     if query is not None:
#         include_all = request.args.get('include_all') == 'y'
#         correct_spelling = request.args.get('correct_spelling') == 'y'
#
#         corrected_query = spelling_corrector._transform(query) \
#             if correct_spelling else query
#         preprocessed_query = preprocessor.transform([corrected_query])[0]
#
#         prediction = label_encoder.inverse_transform(
#             classifier.predict([preprocessed_query]))[0]
#         certainty = np.max(classifier.predict_proba([preprocessed_query]))
#         if certainty >= 0.8:
#             certainty = 'sehr sicher'
#         elif certainty >= 0.5:
#             certainty = 'sicher'
#         elif certainty >= 0.2:
#             certainty = 'unsicher'
#         elif certainty >= 0:
#             certainty = 'sehr unsicher'
#         prediction = (prediction, certainty)
#
#         extracted_features = dict(
#             extract_features_mira(clean_mira(
#                 [query], replacement_dicts=replacement_dicts),
#                 token_lists, token_feature_names).iloc[0])
#         extracted_features = [(translation.get(item[0], 'andere'), item[1])
#                               for item in extracted_features.items()
#                               if not(type(item[1]) == np.float64
#                                      and np.isnan(item[1]) or item[1] is pd.NA
#                                      or item[0]
#                                      in ['manufacturer article number',
#                                          'additional fee', 'ambulant',
#                                          'stationary'])]
#
#         indices = search_engine.recommend(corrected_query, max_count=None,
#                                           output='indices',
#                                           correct_spelling=correct_spelling,
#                                           include_all=include_all)
#         recommendations = mira.iloc[indices][['article', 'prediction print',
#                                               'certainty print']].values
#     else:
#         prediction = None
#         extracted_features = []
#         recommendations = []
#
#     return render_template('index.html', query=query, include_all=include_all,
#                            correct_spelling=correct_spelling,
#                            prediction=prediction,
#                            extracted_features=extracted_features,
#                            recommendations=recommendations,
#                            dosage_forms_dict=dosage_forms_dict)

#
# if __name__ == "__main__":
#     app.run(debug=True)
