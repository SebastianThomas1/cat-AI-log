# Sebastian Thomas (datascience at sebastianthomas dot de)

# regular expressions
import re

# data
import numpy as np
import pandas as pd


# inspired by https://stackoverflow.com/questions/44558215/#44559180
def left_justify_and_cut(a):
    mask = ~np.isnan(a)
    justified_mask = np.flip(np.sort(mask, axis=1), axis=1)
    b = np.full(a.shape, np.nan)
    b[justified_mask] = a[mask]

    return b[:, ~np.all(np.isnan(b), axis=0)]


def extract_tokens(corpus, tokens, feature_name):
    # idea for improvement: make two steps
    # first: strict case sensitive extraction
    # second: only if nothing is found, try a case insensitive
    # extraction
    
    corpus = pd.Series(corpus)

    pattern = r'\b(?P<token>' + '|'.join(tokens) + r')(?:\s|\.|\b)'
    matches = corpus.str.findall(pattern, flags=re.IGNORECASE)

    output = pd.DataFrame(matches.apply(lambda l: ' '.join(l))
                          .replace('', pd.NA)).astype('string')
    output = output.reindex(corpus.index)
    output.columns = [feature_name]

    return output


def extract_manufacturer_article_numbers(corpus):
    corpus = pd.Series(corpus)

    pattern = r'\[(?P<article_number>\d+)\]$'
    
    matches = corpus.str.extractall(pattern)
    matches = matches.unstack()

    if matches.shape[1] != 0:
        output = matches['article_number'].astype('string')
    else:
        output = pd.DataFrame(np.full((corpus.size, 1), pd.NA)) \
            .astype('string')
    output = output.reindex(corpus.index)
    output.columns = ['manufacturer article number']

    return output


def extract_additional_fee(corpus):
    return pd.Series(corpus).str.contains(r'(?:ZE|\(ZE\))$') \
        .to_frame(name='additional fee')


def extract_ambulant(corpus):
    return pd.Series(corpus).str.contains(r'\b(?:amb\.|ambulant|ambu)') \
        .to_frame(name='ambulant')


def extract_stationary(corpus):
    return pd.Series(corpus).str.contains(r'\b(?:stat\.|stationär|station)') \
        .to_frame(name='stationary')


def extract_mass_concentrations(corpus):
    # necessary since <NA>-value cannot be handled later
    corpus = pd.Series(corpus).astype(object)

    pattern = r'\b(?P<mass_value>\d+(?:[\.\,]\d*)?)\s?' \
              r'(?P<mass_unit>[uµm]?g)\s?\/\s?' \
              r'(?P<volume_value>\d*(?:[\.\,]\d*)?)\s?ml\b'
    matches = corpus.str.extractall(pattern, flags=re.IGNORECASE)
    matches['mass_unit'] = matches['mass_unit'].str.lower()
    matches['mass [mg]'] = (matches['mass_value'].str.replace(',', '.')
                            .astype(float)
                            * matches['mass_unit'].map({'ug': 0.001,
                                                        'µg': 0.001,
                                                        'mg': 1.,
                                                        'g': 1000.}))
    matches['volume [ml]'] = matches['volume_value'].str.replace(',', '.')\
        .astype(float).fillna(1.)
    matches['mass concentration [mg/ml]'] \
        = matches['mass [mg]'] / matches['volume [ml]']
    matches = matches.unstack()

    if matches.shape[1] != 0:
        output = matches[['mass concentration [mg/ml]']]
    else:
        output = pd.DataFrame(np.full((corpus.size, 1), np.nan))
    output = output.reindex(corpus.index)
    output.columns = ['mass concentration {} [mg/ml]'.format(idx)
                      for idx in range(output.shape[1])]

    return output


def extract_mass_puff_concentrations(corpus):
    # necessary since <NA>-value cannot be handled later
    corpus = pd.Series(corpus).astype(object)

    pattern = r'\b(?P<mass_value>\d+(?:[\.\,]\d+)?)\s?' \
              r'(?P<mass_unit>[uµm]?g)\s?\/\s?hub\b'
    matches = corpus.str.extractall(pattern, flags=re.IGNORECASE)
    matches['mass_unit'] = matches['mass_unit'].str.lower()
    matches['mass puff concentration [mg/hub]'] \
        = (matches['mass_value'].str.replace(',', '.').astype(float)
           * matches['mass_unit'].map({'ug': 0.001, 'µg': 0.001,  'mg': 1.,
                                       'g': 1000.}))
    matches = matches.unstack()

    if matches.shape[1] != 0:
        output = matches[['mass puff concentration [mg/hub]']]
    else:
        output = pd.DataFrame(np.full((corpus.size, 1), np.nan))
    output = output.reindex(corpus.index)
    output.columns = ['mass puff concentration {} [mg/hub]'.format(idx)
                      for idx in range(output.shape[1])]

    return output


def extract_mass_flows(corpus):
    # necessary since <NA>-value cannot be handled later
    corpus = pd.Series(corpus).astype(object)

    pattern = r'\b(?P<mass_value>\d+(?:[\.\,]\d+)?)\s?' \
              r'(?P<mass_unit>[uµm]?g)\s?\/\s?(?P<time_value>\d*)\s?h\b'
    matches = corpus.str.extractall(pattern, flags=re.IGNORECASE)
    matches['mass_unit'] = matches['mass_unit'].str.lower()
    matches['mass [mg]'] \
        = (matches['mass_value'].str.replace(',', '.').astype(float)
           * matches['mass_unit'].map({'ug': 0.001, 'µg': 0.001, 'mg': 1.,
                                       'g': 1000.}))
    matches['time [h]'] = matches['time_value'].astype(float).fillna(1.)
    matches['mass flow [mg/h]'] = matches['mass [mg]'] / matches['time [h]']
    matches = matches.unstack()

    if matches.shape[1] != 0:
        output = matches[['mass flow [mg/h]']]
    else:
        output = pd.DataFrame(np.full((corpus.size, 1), np.nan))
    output = output.reindex(corpus.index)
    output.columns = ['mass flow {} [mg/h]'.format(idx)
                      for idx in range(output.shape[1])]

    return output


def extract_volume_flows(corpus):
    # necessary since <NA>-value cannot be handled later
    corpus = pd.Series(corpus).astype(object)

    pattern = r'\b(?P<volume_value>\d+(?:[\.\,]\d+)?)\s?' \
              r'(?P<volume_unit>[mc]?l|liter)\s?\/\s?(?P<time_value>\d*)\s?h\b'
    matches = corpus.str.extractall(pattern, flags=re.IGNORECASE)
    matches['volume_unit'] = matches['volume_unit'].str.lower()
    matches['volume [ml]'] \
        = (matches['volume_value'].str.replace(',', '.').astype(float)
           * matches['volume_unit'].map({'ml': 1., 'cl': 10., 'l': 1000.,
                                         'liter': 1000.}))
    matches['time [h]'] = matches['time_value'].astype(float).fillna(1.)
    matches['volume flow [ml/h]'] \
        = matches['volume [ml]'] / matches['time [h]']
    matches = matches.unstack()

    if matches.shape[1] != 0:
        output = matches[['volume flow [ml/h]']]
    else:
        output = pd.DataFrame(np.full((corpus.size, 1), np.nan))
    output = output.reindex(corpus.index)
    output.columns = ['volume flow {} [ml/h]'.format(idx)
                      for idx in range(output.shape[1])]

    return output


def extract_active_ingredient_percentages(corpus):
    # necessary since <NA>-value cannot be handled later
    corpus = pd.Series(corpus).astype(object)

    pattern = r'\b(?P<mass_value>\d+(?:[\.\,]\d+)?)\s?mg\s?\/\s?g\b'
    matches = corpus.str.extractall(pattern, flags=re.IGNORECASE)
    matches['active ingredient percentage [%]'] \
        = matches['mass_value'].str.replace(',', '.').astype(float) / 10
    matches = matches.unstack()

    if matches.shape[1] != 0:
        output = matches[['active ingredient percentage [%]']]
    else:
        output = pd.DataFrame(np.full((corpus.size, 1), np.nan))
    output = output.reindex(corpus.index)
    output.columns = ['active ingredient percentage {} [%]'.format(idx)
                      for idx in range(output.shape[1])]

    return output


def extract_masses(corpus):
    # necessary since <NA>-value cannot be handled later
    corpus = pd.Series(corpus).astype(object)

    pattern = r'\b(?P<value>\d+(?:[\.\,]\d+)?)\s?' \
              r'(?P<unit>[uµmk]?g)\b(?!\s?\/\s?(?:ml|h|hub|g))'
    matches = corpus.str.extractall(pattern, flags=re.IGNORECASE)
    matches['unit'] = matches['unit'].str.lower()
    matches['mass [mg]'] \
        = (matches['value'].str.replace(',', '.').astype(float)
           * matches['unit'].map({'ug': 0.001, 'µg': 0.001, 'mg': 1.,
                                  'g': 1000., 'kg': 1000000.}))
    matches = matches.unstack()

    if matches.shape[1] != 0:
        output = matches[['mass [mg]']]
    else:
        output = pd.DataFrame(np.full((corpus.size, 1), np.nan))
    output = output.reindex(corpus.index)
    output.columns = ['mass {} [mg]'.format(idx)
                      for idx in range(output.shape[1])]

    return output


def extract_volumes(corpus):
    # necessary since <NA>-value cannot be handled later
    corpus = pd.Series(corpus).astype(object)

    pattern = r'\b(?P<value>\d+(?:[\.\,]\d+)?)\s?' \
              r'(?P<unit>[mc]?l|liter)\b(?!\s?\/\s?h)'
    matches = corpus.str.extractall(pattern, flags=re.IGNORECASE)
    matches['unit'] = matches['unit'].str.lower()
    matches['volume [ml]'] \
        = (matches['value'].str.replace(',', '.').astype(float)
           * matches['unit'].map({'ml': 1., 'cl': 10., 'l': 1000.,
                                  'liter': 1000.}))
    matches = matches.unstack()

    if matches.shape[1] != 0:
        output = matches[['volume [ml]']]
    else:
        output = pd.DataFrame(np.full((corpus.size, 1), np.nan))
    output = output.reindex(corpus.index)
    output.columns = ['volume {} [ml]'.format(idx)
                      for idx in range(output.shape[1])]

    return output


def extract_count_puffs(corpus):
    # necessary since <NA>-value cannot be handled later
    corpus = pd.Series(corpus).astype(object)

    pattern = r'\b(?P<value>\d+)\s?hub\b'
    matches = corpus.str.extractall(pattern, flags=re.IGNORECASE)
    matches['count puff [hub]'] = matches['value'].astype(int)
    matches = matches.unstack()

    if matches.shape[1] != 0:
        output = matches[['count puff [hub]']]
    else:
        output = pd.DataFrame(np.full((corpus.size, 1), np.nan))
    output = output.reindex(corpus.index)
    output.columns = ['count puff {} [hub]'.format(idx)
                      for idx in range(output.shape[1])]

    return output


def extract_percentages(corpus):
    # necessary since <NA>-value cannot be handled later
    corpus = pd.Series(corpus).astype(object)

    pattern = r'(?P<value>\d+(?:[\.\,]\d+)?)\s?(?:\%)'
    matches = corpus.str.extractall(pattern)
    matches['percentage [%]'] \
        = matches['value'].str.replace(',', '.').astype(float)
    matches = matches.unstack()

    if matches.shape[1] != 0:
        output = matches[['percentage [%]']]
    else:
        output = pd.DataFrame(np.full((corpus.size, 1), np.nan))
    output = output.reindex(corpus.index)
    output.columns = ['percentage {} [%]'.format(idx)
                      for idx in range(output.shape[1])]

    return output


def extract_lengths(corpus):
    # necessary since <NA>-value cannot be handled later
    corpus = pd.Series(corpus)  # .astype(object)

    pattern3d = r'\b(?P<length_value_1>\d+(?:[\.\,]\d+)?)\s?' \
                r'(?P<length_unit_1>[mc]?m)?\s?[x\/]\s?' \
                r'(?P<length_value_2>\d+(?:[\.\,]\d+)?)\s?' \
                r'(?P<length_unit_2>[mc]?m)?\s?[x\/]\s?' \
                r'(?P<length_value_3>\d+(?:[\.\,]\d+)?)\s?' \
                r'(?P<length_unit_3>[mc]?m)\b'
    matches3d = corpus.str.extractall(pattern3d, flags=re.IGNORECASE)
    matches3d['length_unit_1'] = matches3d['length_unit_1'].str.lower()
    matches3d['length_unit_2'] = matches3d['length_unit_2'].str.lower()
    matches3d['length_unit_3'] = matches3d['length_unit_3'].str.lower()
    matches3d['length_unit_1'].fillna(matches3d['length_unit_3'], inplace=True)
    matches3d['length_unit_2'].fillna(matches3d['length_unit_3'], inplace=True)
    matches3d['length 1 [cm]'] \
        = (matches3d['length_value_1'].str.replace(',', '.').astype(float)
           * matches3d['length_unit_1'].map({'mm': 0.1, 'cm': 1., 'm': 100.}))
    matches3d['length 2 [cm]'] \
        = (matches3d['length_value_2'].str.replace(',', '.').astype(float)
           * matches3d['length_unit_2'].map({'mm': 0.1, 'cm': 1., 'm': 100.}))
    matches3d['length 3 [cm]'] \
        = (matches3d['length_value_3'].str.replace(',', '.').astype(float)
           * matches3d['length_unit_3'].map({'mm': 0.1, 'cm': 1., 'm': 100.}))
    matches3d = matches3d.unstack().reindex(corpus.index)
    if matches3d.shape[1] != 0:
        a3 = matches3d[['length 1 [cm]', 'length 2 [cm]', 'length 3 [cm]']] \
            .stack(0).unstack().reindex(corpus.index).values
    else:
        a3 = np.full((corpus.size, 3), np.nan)

    corpus = corpus.str.replace(pattern3d, '', flags=re.IGNORECASE, regex=True)
    pattern2d = r'(?!\s?[x\/]\s?)\b(?P<length_value_1>\d+(?:[\.\,]\d+)?)\s?' \
                r'(?P<length_unit_1>[mc]?m)?\s?[x\/]\s?' \
                r'(?P<length_value_2>\d+(?:[\.\,]\d+)?)\s?' \
                r'(?P<length_unit_2>[mc]?m)\b'
    matches2d = corpus.str.extractall(pattern2d, flags=re.IGNORECASE)
    matches2d['length_unit_1'] = matches2d['length_unit_1'].str.lower()
    matches2d['length_unit_2'] = matches2d['length_unit_2'].str.lower()
    matches2d['length_unit_1'].fillna(matches2d['length_unit_2'], inplace=True)
    matches2d['length 1 [cm]'] \
        = (matches2d['length_value_1'].str.replace(',', '.').astype(float)
           * matches2d['length_unit_1'].map({'mm': 0.1, 'cm': 1., 'm': 100.}))
    matches2d['length 2 [cm]'] \
        = (matches2d['length_value_2'].str.replace(',', '.').astype(float)
           * matches2d['length_unit_2'].map({'mm': 0.1, 'cm': 1., 'm': 100.}))
    matches2d = matches2d.unstack().reindex(corpus.index)
    if matches2d.shape[1] != 0:
        a2 = matches2d[['length 1 [cm]', 'length 2 [cm]']].stack(0).unstack() \
            .reindex(corpus.index).values
    else:
        a2 = np.full((corpus.size, 2), np.nan)
   
    corpus = corpus.str.replace(pattern2d, '', flags=re.IGNORECASE, regex=True)
    pattern1d = r'(?:\b)(?P<length_value>\d+(?:[\.\,]\d+)?)\s?' \
                r'(?P<length_unit>[mc]?m)\b'
    matches1d = corpus.str.extractall(pattern1d, flags=re.IGNORECASE)
    matches1d['length_unit'] = matches1d['length_unit'].str.lower()
    matches1d['length [cm]'] \
        = (matches1d['length_value'].str.replace(',', '.').astype(float)
           * matches1d['length_unit'].map({'mm': 0.1, 'cm': 1., 'm': 100.}))
    matches1d = matches1d.unstack().reindex(corpus.index)
    if matches1d.shape[1] != 0:
        a1 = matches1d[['length [cm]']].values
    else:
        a1 = np.full((corpus.size, 1), np.nan)

    lengths = left_justify_and_cut(np.concatenate([a3, a2, a1], axis=1))

    if lengths.shape[1] != 0:
        output = pd.DataFrame(lengths)
    else:
        output = pd.DataFrame(np.full((corpus.size, 1), np.nan))
    output.set_index(corpus.index, inplace=True)
    output.set_axis(['length {} [cm]'.format(idx)
                     for idx in range(output.shape[1])], axis=1, inplace=True)

    return output


def extract_counts(corpus):
    # necessary since <NA>-value cannot be handled later
    corpus = pd.Series(corpus).astype(object)
    
    pattern1 = r'\b(?P<count>\d{1,2})st\.?\b'
    matches1 = corpus.str.extractall(pattern1, flags=re.IGNORECASE)
    matches1['count'] = matches1['count'].astype('float')
    matches1 = matches1.unstack().reindex(corpus.index)
    if matches1.shape[1] != 0:
        a1 = matches1[['count']].values
    else:
        a1 = np.full((corpus.size, 1), np.nan)

    pattern2 = r'\((?P<count>\d{1,3})\)'
    matches2 = corpus.str.extractall(pattern2, flags=re.IGNORECASE)
    matches2['count'] = matches2['count'].astype('float')
    matches2 = matches2.unstack().reindex(corpus.index)
    if matches2.shape[1] != 0:
        a2 = matches2[['count']].values
    else:
        a2 = np.full((corpus.size, 1), np.nan)

    counts = left_justify_and_cut(np.concatenate([a1, a2], axis=1))
    
    if counts.shape[1] != 0:
        output = pd.DataFrame(counts)
    else:
        output = pd.DataFrame(np.full((corpus.size, 1), np.nan))
    output.set_index(corpus.index, inplace=True)
    output.set_axis(['count {}'.format(idx)
                     for idx in range(output.shape[1])], axis=1, inplace=True)

    return output


def extract_features_mira(corpus, token_lists, token_feature_names=None):
    # necessary since <NA>-value cannot be handled later
    corpus = pd.Series(corpus).astype(object)
    features = pd.DataFrame(np.zeros((corpus.size, 0)), index=corpus.index)

    if token_feature_names is None \
            or len(token_feature_names) != len(token_lists):
        token_feature_names = ['token ' + str(idx)
                               for idx in range(len(token_lists))]

    for (token_list, feature_name) in zip(token_lists, token_feature_names):
        features = pd.concat([features, extract_tokens(corpus, token_list,
                                                       feature_name)], axis=1)

    features = pd.concat([features,
                          extract_manufacturer_article_numbers(corpus),
                          extract_additional_fee(corpus),
                          extract_ambulant(corpus),
                          extract_stationary(corpus),
                          extract_mass_concentrations(corpus),
                          extract_mass_puff_concentrations(corpus),
                          extract_mass_flows(corpus),
                          extract_volume_flows(corpus),
                          extract_active_ingredient_percentages(corpus),
                          extract_masses(corpus),
                          extract_volumes(corpus),
                          extract_count_puffs(corpus),
                          extract_percentages(corpus),
                          extract_lengths(corpus),
                          extract_counts(corpus)], axis=1)

    return features


def delete_tokens(corpus, tokens):
    return pd.Series(corpus).str.replace(r'\b(' + '|'.join(tokens)
                                         + r')(\s|\.|\b)', '',
                                         flags=re.IGNORECASE, regex=True)


def delete_manufacturer_article_numbers(corpus):
    return pd.Series(corpus).str.replace(r'\[(\d+)\]$', '', regex=True)


def delete_additional_fee(corpus):
    return pd.Series(corpus).str.replace(r'(?:ZE|\(ZE\))$', '', regex=True)


def delete_ambulant(corpus):
    return pd.Series(corpus).str.replace(r'\b(?:amb\.|ambulant|ambu)', '',
                                         regex=True)


def delete_stationary(corpus):
    return pd.Series(corpus).str.replace(r'\b(?:stat\.|stationär|station)', '',
                                         regex=True)


def delete_mass_concentrations(corpus):
    return pd.Series(corpus).str.replace(
        r'\b(?P<mass_value>\d+(?:[\.\,]\d*)?)\s?'
        r'(?P<mass_unit>[uµm]?g)\s?\/\s?'
        r'(?P<volume_value>\d*(?:[\.\,]\d*)?)\s?ml\b', '', flags=re.IGNORECASE,
        regex=True)


def delete_mass_puff_concentrations(corpus):
    return pd.Series(corpus).str.replace(
        r'\b(?P<mass_value>\d+(?:[\.\,]\d+)?)\s?'
        r'(?P<mass_unit>[uµm]?g)\s?\/\s?hub\b', '', flags=re.IGNORECASE,
        regex=True)


def delete_mass_flows(corpus):
    return pd.Series(corpus).str.replace(
        r'\b(?P<mass_value>\d+(?:[\.\,]\d+)?)\s?'
        r'(?P<mass_unit>[uµm]?g)\s?\/\s?(?P<time_value>\d*)\s?h\b', '',
        flags=re.IGNORECASE, regex=True)


def delete_volume_flows(corpus):
    return pd.Series(corpus).str.replace(
        r'\b(?P<volume_value>\d+(?:[\.\,]\d+)?)\s?'
        r'(?P<volume_unit>[mc]?l|liter)\s?\/\s?(?P<time_value>\d*)\s?h\b', '',
        flags=re.IGNORECASE, regex=True)


def delete_active_ingredient_percentages(corpus):
    return pd.Series(corpus).str.replace(
        r'\b(?P<mass_value>\d+(?:[\.\,]\d+)?)\s?mg\s?\/\s?g\b', '',
        flags=re.IGNORECASE, regex=True)


def delete_masses(corpus):
    return pd.Series(corpus).str.replace(
        r'\b(?P<value>\d+(?:[\.\,]\d+)?)\s?'
        r'(?P<unit>[uµmk]?g)\b(?!\s?\/\s?(?:ml|h|hub|g))', '',
        flags=re.IGNORECASE, regex=True)


def delete_volumes(corpus):
    return pd.Series(corpus).str.replace(
        r'\b(?P<value>\d+(?:[\.\,]\d+)?)\s?'
        r'(?P<unit>[mc]?l|liter)\b(?!\s?\/\s?h)', '', flags=re.IGNORECASE,
        regex=True)


def delete_count_puffs(corpus):
    return pd.Series(corpus).str.replace(
        r'\b(?P<value>\d+)\s?hub\b', '', flags=re.IGNORECASE, regex=True)


def delete_percentages(corpus):
    return pd.Series(corpus).str.replace(
        r'(?P<value>\d+(?:[\.\,]\d+)?)\s?(?:\%)', '', flags=re.IGNORECASE,
        regex=True)


def delete_lengths(corpus):
    corpus = pd.Series(corpus)
    corpus = corpus.str.replace(
        r'\b(?P<length_value_1>\d+(?:[\.\,]\d+)?)\s?'
        r'(?P<length_unit_1>[mc]?m)?\s?[x\/]\s?'
        r'(?P<length_value_2>\d+(?:[\.\,]\d+)?)\s?'
        r'(?P<length_unit_2>[mc]?m)?\s?[x\/]\s?'
        r'(?P<length_value_3>\d+(?:[\.\,]\d+)?)\s?'
        r'(?P<length_unit_3>[mc]?m)\b', '', flags=re.IGNORECASE, regex=True)
    corpus = corpus.str.replace(
        r'(?!\s?[x\/]\s?)\b(?P<length_value_1>\d+(?:[\.\,]\d+)?)\s?'
        r'(?P<length_unit_1>[mc]?m)?\s?[x\/]\s?'
        r'(?P<length_value_2>\d+(?:[\.\,]\d+)?)\s?'
        r'(?P<length_unit_2>[mc]?m)\b', '', flags=re.IGNORECASE, regex=True)
    corpus = corpus.str.replace(
        r'(?:\b)(?P<length_value>\d+(?:[\.\,]\d+)?)\s?'
        r'(?P<length_unit>[mc]?m)\b', '', flags=re.IGNORECASE, regex=True)
    return corpus


def delete_counts(corpus):
    corpus = pd.Series(corpus)
    corpus = corpus.str.replace(
        r'\b(?P<count>\d{1,2})st\.?\b', '', flags=re.IGNORECASE, regex=True)
    corpus = corpus.str.replace(
        r'\((?P<count>\d{1,3})\)', '', flags=re.IGNORECASE, regex=True)
    return corpus


def reduce_to_base(corpus, token_lists):
    for token_list in token_lists:
        corpus = delete_tokens(corpus, token_list)
    corpus = delete_manufacturer_article_numbers(corpus)
    corpus = delete_additional_fee(corpus)
    corpus = delete_ambulant(corpus)
    corpus = delete_stationary(corpus)
    corpus = delete_mass_concentrations(corpus)
    corpus = delete_mass_puff_concentrations(corpus)
    corpus = delete_mass_flows(corpus)
    corpus = delete_volume_flows(corpus)
    corpus = delete_active_ingredient_percentages(corpus)
    corpus = delete_masses(corpus)
    corpus = delete_volumes(corpus)
    corpus = delete_count_puffs(corpus)
    corpus = delete_percentages(corpus)
    corpus = delete_lengths(corpus)
    corpus = delete_counts(corpus)
    corpus = pd.Series(corpus).str.replace(r'\d+', ' ', regex=True)
    
    return corpus
