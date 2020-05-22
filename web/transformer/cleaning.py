# regular expressions
import re

# data
import numpy as np
import pandas as pd


def remove_strange_initial_final_characters(corpus):
    return pd.Series(corpus).str.replace(r'^(\!)+|^(\*)+|(\*)+$', '')

def replace_tokens(corpus, replacements):
    pattern = (r'\b(' + '|'.join(replacements.keys()).replace('.', '\.') + r')(?:\s|\.|\b)')
    return pd.Series(corpus).str.replace(pattern, lambda match: replacements[match.group(1).lower()] + ' ', flags=re.IGNORECASE)

def insert_space_after_abbreviation(corpus):
    return pd.Series(corpus).str.replace(r'([A-Za-z]\.)(\S)', lambda m: m.group(1) + ' ' + m.group(2))

def clean_mira(corpus, replacement_dicts):
    corpus = remove_strange_initial_final_characters(corpus)
    for replacement_dict in replacement_dicts:
        corpus = replace_tokens(corpus, replacement_dict)
#    corpus = insert_space_after_abbreviation(corpus)

    return corpus