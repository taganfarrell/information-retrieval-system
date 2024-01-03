# Import Statements
import re
import os
import time
from collections import Counter
from collections import defaultdict
import shutil
import uuid
import json
import argparse
import math
import pickle
from copy import deepcopy
from nltk.stem import PorterStemmer
ps = PorterStemmer()

start_time = time.time()

print("Project 2:")

def read_docs(directory):
    # Initializing regex expressions to extract information from documents
    text_exp = r'<TEXT>(.*?)</TEXT>'
    docno_exp = r'<DOCNO>(.*?)</DOCNO>'
    comment_exp = r'<.*?>'

    for file in os.listdir(directory):  # <-- You only need this loop
        file_path = os.path.join(directory, file)

        # Ignoring files that don't end with '.txt' or start with a dot '.'
        if file.startswith('.'):
            continue

        # print(f"Reading file: {file_path}")
        # checking if it is a file
        if os.path.isfile(file_path):
            try:
                with open(file_path, 'rt') as f:
                    text = f.read()
                    text_matches = re.finditer(text_exp, text, re.DOTALL)
                    docno_matches = re.finditer(docno_exp, text, re.DOTALL)

                    for text_match, docno_match in zip(text_matches, docno_matches):
                        extracted_text = text_match.group(1)  # takes the first group of <TEXT>
                        extracted_text = re.sub(comment_exp, '', extracted_text)  # removes comments and leaves only plain text
                        extracted_text = replace_sgml_escapes(extracted_text)  # Replace SGML escapes with actual characters
                        text_content = extracted_text.strip()  # removes leading and trailing whitespace
                        docno_content = docno_match.group(1).strip()  # gets first doc number
                        doc_dict = {'DOCNO': docno_content,
                                    'TEXT': text_content}  # constructs a dictionary containing extracted DOCNO and TEXT
                        yield doc_dict

            except UnicodeDecodeError:
                print(f"Error decoding file: {file_path}. Skipping.")


def read_queries_single(query_file, inverted_index_path):
    # Load the inverted index to access the idf values
    with open(inverted_index_path, 'rt') as f:
        inverted_index = {entry['term']: entry for entry in map(json.loads, f)}

    # Define regex patterns to extract information from the queries
    num_exp = r'<num> Number: (\d+)'
    title_exp = r'<title> Topic: ([\s\S]+?)\s*<'

    # Open the provided file and read its content
    with open(query_file, 'rt') as f:
        content = f.read()

        # Use regex to find all matches for the defined patterns
        num_matches = re.finditer(num_exp, content)
        title_matches = re.finditer(title_exp, content)

        # Loop through matches and extract desired data
        for num_match, title_match in zip(num_matches, title_matches):
            num_content = int(num_match.group(1))  # Extract number
            title_content = title_match.group(1).strip()  # Extract title
            processed_query = tokenize(title_content)

            # Calculate TF for the terms in the query
            term_frequencies = {}
            for term, positions in processed_query.items():
                term_frequencies[term] = len(positions)

            # Calculate TF-IDF weights for the terms in the query
            tf_idf_weights = {}
            for term, tf in term_frequencies.items():
                idf = inverted_index.get(term, {}).get('idf', 0)  # Get idf from index, default to 0 if not found
                tf_idf_weights[term] = tf * idf

            # Yield a dictionary containing the extracted number, processed query, and tf-idf weights
            yield {'num': num_content, 'title': processed_query, 'tf_idf_weights': tf_idf_weights}


def process_narrative_queries(query_file_path, idf_dict, threshold):
    # Regular expressions for extracting data from the query file
    num_pattern = re.compile(r"<num> Number: (\d+)")
    narr_pattern = re.compile(r"<narr> Narrative:\s*(.*?)\n\n", re.DOTALL)

    # Read and process the query file
    with open(query_file_path, 'r') as file:
        content = file.read()

        # Find all matches for query ID and narrative text
        query_ids = num_pattern.findall(content)
        narratives = narr_pattern.findall(content)

        for query_id, narrative in zip(query_ids, narratives):
            # Tokenize the narrative text
            processed_narrative = tokenize(narrative)

            # Calculate term frequencies and TF-IDF weights
            term_frequencies = {}
            tf_idf_weights = {}
            for term in processed_narrative:
                term_frequencies[term] = term_frequencies.get(term, 0) + 1
                idf = idf_dict.get(term, 0)
                tf_idf_weights[term] = term_frequencies[term] * idf

            # Filter terms based on top n terms
            # filtered_terms = select_top_n_terms(tf_idf_weights, threshold)
            # Filter based on tf-idf values above threshold
            # filtered_terms = {term: tf_idf_weights[term] for term in tf_idf_weights if tf_idf_weights[term] >= threshold}
            # Filter based on top n percentage terms of tf-idf values
            filtered_terms = select_top_percentage_terms(tf_idf_weights, threshold)
            # print(f"Top Terms (Threshold: {threshold}): {filtered_terms}\n")

            yield {
                'num': int(query_id),
                'title': {term: [i+1 for i in range(term_frequencies[term])] for term in filtered_terms},
                'tf_idf_weights': filtered_terms
            }


def select_top_n_terms(tf_idf_weights, top_n):
    # Sort terms by their TF-IDF weights in descending order and select the top N terms
    return dict(sorted(tf_idf_weights.items(), key=lambda item: item[1], reverse=True)[:top_n])


def select_top_percentage_terms(tf_idf_weights, percentage):
    # Calculate how many terms to keep
    number_of_terms = max(1, int(len(tf_idf_weights) * (percentage / 100)))

    # Sort terms by their TF-IDF weights in descending order and select the top percentage
    return dict(sorted(tf_idf_weights.items(), key=lambda item: item[1], reverse=True)[:number_of_terms])


def read_queries_stem(query_file, inverted_index_path):
    stopwords = ["is", "the", "and", "a", "an", "of", "in", "to", "for", "with", "on", "at", "by", "from", "about"]
    ps = PorterStemmer()

    # Load the inverted stem index to access the idf values
    with open(inverted_index_path, 'rt') as f:
        inverted_index = {entry['term']: entry for entry in map(json.loads, f)}

    # Define regex patterns to extract information from the queries
    num_exp = r'<num> Number: (\d+)'
    title_exp = r'<title> Topic: ([\s\S]+?)\s*<'

    # Open the provided file and read its content
    with open(query_file, 'rt') as f:
        content = f.read()

        # Use regex to find all matches for the defined patterns
        num_matches = re.finditer(num_exp, content)
        title_matches = re.finditer(title_exp, content)

        # Loop through matches and extract desired data
        for num_match, title_match in zip(num_matches, title_matches):
            num_content = int(num_match.group(1))  # Extract number
            title_content = title_match.group(1).strip()  # Extract title
            processed_query = tokenize(title_content)
            stemmed_terms = [ps.stem(term) for term in processed_query if term not in stopwords]

            # Calculate TF for the stemmed terms in the query
            term_frequencies = {}
            for term in stemmed_terms:
                term_frequencies[term] = term_frequencies.get(term, 0) + 1

            # Calculate TF-IDF weights for the terms in the query
            tf_idf_weights = {}
            for term, tf in term_frequencies.items():
                idf = inverted_index.get(term, {}).get('idf', 0)  # Get idf from index, default to 0 if not found
                tf_idf_weights[term] = tf * idf

            # Yield a dictionary containing the extracted number, the dictionary of term frequencies, and tf-idf weights
            yield {'num': num_content, 'title': term_frequencies, 'tf_idf_weights': tf_idf_weights}


def read_queries_dynamic(query_file, inverted_phrase_index_path):
    stopwords = ["is", "the", "and", "a", "an", "of", "in", "to", "for", "with", "on", "at", "by", "from", "about"]

    # Load the inverted stem index to access the idf values
    with open(inverted_phrase_index_path, 'rt') as f:
        inverted_index = {entry['phrase']: entry for entry in map(json.loads, f)}

    # Define regex patterns to extract information from the queries
    num_exp = r'<num> Number: (\d+)'
    title_exp = r'<title> Topic: ([\s\S]+?)\s*<'

    # Open the provided file and read its content
    with open(query_file, 'rt') as f:
        content = f.read()

        # Use regex to find all matches for the defined patterns
        num_matches = re.finditer(num_exp, content)
        title_matches = re.finditer(title_exp, content)

        # Loop through matches and extract desired data
        for num_match, title_match in zip(num_matches, title_matches):
            num_content = int(num_match.group(1))  # Extract number
            title_content = title_match.group(1).strip()  # Extract title
            phrases = get_phrases(title_content)

            # Calculate TF for the stemmed terms in the query
            phrase_frequencies = {}
            for phrase in phrases:
                phrase_frequencies[phrase] = phrase_frequencies.get(phrase, 0) + 1

            # Calculate TF-IDF weights for the terms in the query
            tf_idf_weights = {}
            for phrase, tf in phrase_frequencies.items():
                idf = inverted_index.get(phrase, {}).get('idf', 0)  # Get idf from index, default to 0 if not found
                tf_idf_weights[phrase] = tf * idf
                # print("tf = ", tf, " | idf = ", idf)

            # Yield a dictionary containing the extracted number, the dictionary of term frequencies, and tf-idf weights
            yield {'num': num_content, 'title': phrase_frequencies, 'tf_idf_weights': tf_idf_weights}


def is_valid_date(date_str):
    # Split the date into month, day, and year
    month, day, year = map(int, date_str.split('/'))
    # List of days in each month (non-leap year)
    days_in_month = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    # Check for leap year
    if year % 4 == 0:
        if year % 100 != 0 or (year % 100 == 0 and year % 400 == 0):
            days_in_month[2] = 29  # February has 29 days in a leap year

    # Check if month is valid
    if 1 <= month <= 12:
        # Check if day is valid for the given month
        if 1 <= day <= days_in_month[month]:
            return True

    return False

def replace_sgml_escapes(text):
    text = text.replace('&blank;', '&')
    text = text.replace('&hyph;', '-')
    text = text.replace('&sect;', '§')
    text = text.replace('&times;', '×')
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    text = text.replace('&apos;', '\'')
    text = text.replace('&quot;', '"')
    text = text.replace('&cent;', '¢')
    text = text.replace('&pound;', '£')
    text = text.replace('&yen;', '¥')
    text = text.replace('&euro;', '€')
    text = text.replace('&copy;', '©')
    text = text.replace('&reg;', '®')
    text = text.replace('&amp;', '&')
    return text


def tokenize(text):
    # Case Folding
    text = text.lower()

    # Special Tokens, Part F (Dates: Example: 07/24/2001 or January 20, 1995)
    # Map for month names
    month_map = {
        'january': '01', 'february': '02', 'march': '03', 'april': '04',
        'may': '05', 'june': '06', 'july': '07', 'august': '08',
        'september': '09', 'october': '10', 'november': '11', 'december': '12',
        'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04',
        'jun': '06', 'jul': '07', 'aug': '08', 'sep': '09', 'oct': '10',
        'nov': '11', 'dec': '12'
    }

    # Convert MM-DD-YYYY to MM/DD/YYYY (Example: 05-06-2001 -> 05/06/2001)
    text = re.sub(r'(?<!\d)(\d{1,2})-(\d{1,2})-(\d{4})(?!\d)', r'\1/\2/\3', text)

    # Convert MM-DD-YY to MM/DD/YYYY (Example: 05-06-01 -> 05/06/2001)
    # For 2000s
    text = re.sub(r'(\d{1,2})-(\d{1,2})-([0-2][0-3])(?!\d)', r'\1/\2/20\3', text)
    # For 1900s
    text = re.sub(r'(\d{1,2})-(\d{1,2})-([3-9][0-9]|[2][4-9])(?!\d)', r'\1/\2/19\3', text)

    # Convert Month DD, YYYY and MMM DD, YYYY to MM/DD/YYYY (Example: September 13, 2023 -> 09/13/2001 or Oct 9, 2020)
    month_pattern = (r'(?P<month>january|jan|february|feb|march|mar|april|apr|may|june|jun|july|jul|august|aug|'
                     r'september|sep|october|oct|november|nov|december|dec)\s+(?P<day>\d{1,2}),?\s+(?P<year>\d{4})[.,]?')
    text = re.sub(month_pattern,
                  lambda m: "{}/{}/{}".format(month_map[m.group('month').lower()], m.group('day'), m.group('year')),
                  text)

    # Convert MMM-DD-YYYY to MM/DD/YYYY (Example: Jun-25-1980 -> 06/25/1980)
    mmm_pattern = (r'(?P<month>jan|feb|mar|apr|may|jun|jul|aug|'
                   r'sep|oct|nov|dec)-(?P<day>\d{1,2})-(?P<year>\d{4})')
    text = re.sub(mmm_pattern,
                  lambda m: "{}/{}/{}".format(month_map[m.group('month').lower()], m.group('day'), m.group('year')),
                  text)

    # Special Tokens, Part A (Example: P.H.D)
    text = re.sub(r'\.(?=[a-z][^a-z]|\s|,|;|:|!|-)', '', text)

    # Special Tokens, Part H (store file extensions without the period)
    # text = re.sub(r'([A-Za-z]+)\.([A-Za-z]{2,})', r'\1\2', text)
    text = re.sub(r'([A-Za-z0-9_-]+)\.([A-Za-z0-9_-]+)(?=\s|$)', r'\1 \2', text)

    # Special Tokens, Part C (Alphabet-digit: Example: "F-16", “CDC-50”)
    # 1-2 letters->hyphen->digits
    short_alpha_digit_matches = re.findall(r'\b([a-z]{1,2})-(\d+)', text)
    for match in short_alpha_digit_matches:
        text = re.sub(r'{}-(\d+)'.format(match[0]), match[0] + r'\1', text) # gets rid of hyphen

    # 3+ letters->hyphen->digits
    long_alpha_digit_matches = re.findall(r'\b([a-z]{3,})-(\d+)', text)
    for match in long_alpha_digit_matches:
        text = re.sub(r'{}-(\d+)'.format(match[0]), match[0] + r'\1' + ' ' + match[0], text) # adds cdc50 and cdc

    # Special Tokens, Part D (Digit-alphabet: Example: “1-hour”)
    # numbers->hyphen->1-2 letters
    short_digit_alpha_matches = re.findall(r'\b(\d+)-([a-z]{1,2})\b', text)
    for match in short_digit_alpha_matches:
        text = re.sub(r'(\d+)-{}'.format(match[1]), r'\1' + match[1], text)

    # numbers->hyphen->3+ letters
    long_digit_alpha_matches = re.findall(r'\b(\d+)-([a-z]{3,})\b', text)
    for match in long_digit_alpha_matches:
        text = re.sub(r'(\d+)-{}'.format(match[1]), r'\1' + match[1] + ' ' + match[1], text)

    # Special Tokens, Part E (Hyphenated Terms: Example: pre-processing, black-tie)
    # Handle patterns with specific prefixes
    prefixes = ['pre', 'post', 're', 'co', 'non', 'inter', 'self', 'mid', 'out']
    for prefix in prefixes:
        prefix_matches = re.findall(r'\b{}-([a-z]+)\b'.format(prefix), text)
        for suffix in prefix_matches:
            hyphen_pattern = r'{}-{}'.format(prefix, suffix)
            replacement = prefix + suffix + ' ' + suffix
            text = re.sub(hyphen_pattern, replacement, text)

    # Handle generic hyphenated patterns for words with 1, 2 or 3 hyphens
    generic_hyphenated = re.findall(r'\b([a-z]+)-([a-z]+(?:-[a-z]+){0,2})\b', text)
    for start, rest in generic_hyphenated:
        if start not in prefixes:  # We've already handled specific prefixes
            end_term = rest.split('-')[-1]
            hyphen_pattern = r'{}-{}'.format(start, rest)
            replacement = start + rest.replace('-', '') + ' ' + start + ' ' + end_term
            text = re.sub(hyphen_pattern, replacement, text)

    # Special Tokens, Part G (Change digit formats: Example 1000.00, 1,000.00 and 1,000 to 1000)
    # Remove any commas within numbers, such as 1,000,000
    text = re.sub(r'(?<=\d),(?=\d{3})', '', text)

    # Handle currency
    text = re.sub(r'\$', ' $', text)

    # Pattern to match numbers with a single decimal, ensuring they don't belong to a sequence with more than one decimal.
    zerospattern = r'\b(?<!\.)\d+\.\d+(?=\s|,)'

    # Replacement function to process each matched number.
    def repl(matchobj):
        # Convert matched number to float and then back to string to strip trailing zeros.
        num_str = str(float(matchobj.group(0)))
        # For numbers like 1000.0, remove the trailing .0
        return num_str.rstrip('0').rstrip('.') if '.' in num_str else num_str

    # Get rid of any commas following a number
    text = re.sub(r'(?<=\d),(?=\s|$)', '', text)

    # Apply the regex substitution.
    text = re.sub(zerospattern, repl, text)

    # Define a regular expression pattern to split text into terms
    # This pattern matches spaces, punctuation, and symbols as delimiters
    pattern = r'\s+|(?<=[^\d\s])[,.](?=\s|$)|[!?^*#\/\(\)\[\]{};\'&_]+|:(?=\s)'
    # pattern = r'\\s+|(?<!\\S)[,.](?=\\s|$)|[!?^*#\\(\\)\\[\\]{};\\\'&_]+|:(?=\\s)|(?<=\\S)\\.(?=\\s|$)'

    # Split the text into terms using the pattern
    terms = re.split(pattern, text)
    # Validate dates
    terms = [term for term in terms if not re.match(r'\d+/\d+/\d+', term) or is_valid_date(term)]
    terms = [term for term in terms if term]

    term_positions = {}
    for position, term in enumerate(terms, 1):  # Starts the count from 1, which is the position in the document
        if term not in term_positions:
            term_positions[term] = []
        term_positions[term].append(position)

    return term_positions

def get_phrases(text):
    """
    Extract all 2-term and 3-term phrases from the given text, excluding phrases with stop words.
    """
    text = text.lower()

    special_symbols = {'.', ':', '@', '#', ',', ';', '!', '?', '&', '$', '*', '^', '(', ')', '[', ']', '{', '}'}

    stop_words = {"a", "a's", "able", "about", "above", "according", "accordingly", "across", "actually", "after",
                  "afterwards", "again", "against", "ain't", "all", "allow", "allows", "almost", "alone", "along",
                  "already", "also", "although", "always", "am", "among", "amongst", "an", "and", "another", "any",
                  "anybody", "anyhow", "anyone", "anything", "anyway", "anyways", "anywhere", "apart", "appear",
                  "appreciate", "appropriate", "are", "aren't", "around", "as", "aside", "ask", "asking", "associated",
                  "at", "available", "away", "awfully", "b", "be", "became", "because", "become", "becomes", "becoming",
                  "been", "before", "beforehand", "behind", "being", "believe", "below", "beside", "besides", "best",
                  "better", "between", "beyond", "both", "brief", "but", "by", "c", "c'mon", "c's", "came", "can",
                  "can't", "cannot", "cant", "cause", "causes", "certain", "certainly", "changes", "clearly", "co",
                  "com", "come", "comes", "concerning", "consequently", "consider", "considering", "contain",
                  "containing", "contains", "corresponding", "could", "couldn't", "course", "currently", "d",
                  "definitely", "described", "despite", "did", "didn't", "different", "do", "does", "doesn't", "doing",
                  "don't", "done", "down", "downwards", "during", "e", "each", "edu", "eg", "eight", "either", "else",
                  "elsewhere", "enough", "entirely", "especially", "et", "etc", "even", "ever", "every", "everybody",
                  "everyone", "everything", "everywhere", "ex", "exactly", "example", "except", "f", "far", "few",
                  "fifth", "first", "five", "followed", "following", "follows", "for", "former", "formerly", "forth",
                  "four", "from", "further", "furthermore", "g", "get", "gets", "getting", "given", "gives", "go",
                  "goes", "going", "gone", "got", "gotten", "greetings", "h", "had", "hadn't", "happens", "hardly",
                  "has", "hasn't", "have", "haven't", "having", "he", "he's", "hello", "help", "hence", "her", "here",
                  "here's", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "hi", "him", "himself",
                  "his", "hither", "hopefully", "how", "howbeit", "however", "i", "i'd", "i'll", "i'm", "i've", "ie",
                  "if", "ignored", "immediate", "in", "inasmuch", "inc", "indeed", "indicate", "indicated", "indicates",
                  "inner", "insofar", "instead", "into", "inward", "is", "isn't", "it", "it'd", "it'll", "it's", "its",
                  "itself", "j", "just", "k", "keep", "keeps", "kept", "know", "knows", "known", "l", "last", "lately",
                  "later", "latter", "latterly", "least", "less", "lest", "let", "let's", "like", "liked", "likely",
                  "little", "look", "looking", "looks", "ltd", "m", "mainly", "many", "may", "maybe", "me", "mean",
                  "meanwhile", "merely", "might", "more", "moreover", "most", "mostly", "much", "must", "my", "myself",
                  "n", "name", "namely", "nd", "near", "nearly", "necessary", "need", "needs", "neither", "never",
                  "nevertheless", "new", "next", "nine", "no", "nobody", "non", "none", "noone", "nor", "normally",
                  "not", "nothing", "novel", "now", "nowhere", "o", "obviously", "of", "off", "often", "oh", "ok",
                  "okay", "old", "on", "once", "one", "ones", "only", "onto", "or", "other", "others", "otherwise",
                  "ought", "our", "ours", "ourselves", "out", "outside", "over", "overall", "own", "p", "particular",
                  "particularly", "per", "perhaps", "placed", "please", "plus", "possible", "presumably", "probably",
                  "provides", "q", "que", "quite", "qv", "r", "rather", "rd", "re", "really", "reasonably", "regarding",
                  "regardless", "regards", "relatively", "respectively", "right", "s", "said", "same", "saw", "say",
                  "saying", "says", "second", "secondly", "see", "seeing", "seem", "seemed", "seeming", "seems", "seen",
                  "self", "selves", "sensible", "sent", "serious", "seriously", "seven", "several", "shall", "she",
                  "should", "shouldn't", "since", "six", "so", "some", "somebody", "somehow", "someone", "something",
                  "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", "specified", "specify",
                  "specifying", "still", "sub", "such", "sup", "sure", "t", "t's", "take", "taken", "tell", "tends",
                  "th", "than", "thank", "thanks", "thanx", "that", "that's", "thats", "the", "their", "theirs", "them",
                  "themselves", "then", "thence", "hopefully", "how", "howbeit", "however", "i", "i'd", "i'll", "i'm",
                  "i've", "ie", "if", "ignored", "immediate", "in", "inasmuch", "inc", "indeed", "indicate",
                  "indicated", "indicates", "inner", "insofar", "instead", "into", "inward", "is", "isn't", "it",
                  "it'd", "it'll", "it's", "its", "itself", "j", "just", "k", "keep", "keeps", "kept", "know", "knows",
                  "known", "l", "last", "lately", "later", "latter", "latterly", "least", "less", "lest", "let",
                  "let's", "like", "liked", "likely", "little", "look", "looking", "looks", "ltd", "m", "mainly",
                  "many", "may", "maybe", "me", "mean", "meanwhile", "merely", "might", "more", "moreover", "most",
                  "mostly", "much", "must", "my", "myself", "n", "name", "namely", "nd", "near", "nearly",
                  "necessary", "need", "needs", "neither", "never", "nevertheless", "new", "next", "nine", "no",
                  "nobody", "non", "none", "noone", "nor", "normally", "not", "nothing", "novel", "now", "nowhere",
                  "o", "obviously", "of", "off", "often", "oh", "ok", "okay", "old", "on", "once", "one", "ones",
                  "only", "onto", "or", "other", "others", "otherwise", "ought", "our", "ours", "ourselves", "out",
                  "outside", "over", "overall", "own", "p", "particular", "particularly", "per", "perhaps", "placed",
                  "please", "plus", "possible", "presumably", "probably", "provides", "q", "que", "quite", "qv", "r",
                  "rather", "rd", "re", "really", "reasonably", "regarding", "regardless", "regards", "relatively",
                  "respectively", "right", "s", "said", "same", "saw", "say", "saying", "says", "second", "secondly",
                  "see", "seeing", "seem", "seemed", "seeming", "seems", "seen", "self", "selves", "sensible",
                  "sent", "serious", "seriously", "seven", "several", "shall", "she", "should", "shouldn't", "since",
                  "six", "so", "some", "somebody", "somehow", "someone", "something", "sometime", "sometimes",
                  "somewhat", "somewhere", "soon", "sorry", "specified", "specify", "specifying", "still", "sub",
                  "such", "sup", "sure", "t", "t's", "take", "taken", "tell", "tends", "th", "than", "thank",
                  "thanks", "thanx", "that", "that's", "thats", "the", "their", "theirs", "them", "themselves",
                  "then", "thence", "there", "there's", "thereafter", "thereby", "therefore", "therein", "theres",
                  "thereupon", "these", "they", "they'd", "they'll", "they're", "they've", "think", "third", "this",
                  "thorough", "thoroughly", "those", "though", "three", "through", "throughout", "thru", "thus",
                  "to", "together", "too", "took", "toward", "towards", "tried", "tries", "truly", "try", "trying",
                  "twice", "two", "u", "un", "under", "unfortunately", "unless", "unlikely", "until", "unto", "up",
                  "upon", "us", "use", "used", "useful", "uses", "using", "usually", "uucp", "v", "value", "various",
                  "very", "via", "viz", "vs", "w", "want", "wants", "was", "wasn't", "way", "we", "we'd", "we'll",
                  "we're", "we've", "welcome", "well", "went", "were", "weren't", "what", "what's", "whatever",
                  "when", "whence", "whenever", "where", "where's", "whereafter", "whereas", "whereby", "wherein",
                  "whereupon", "wherever", "whether", "which", "while", "whither", "who", "who's", "whoever",
                  "whole", "whom", "whose", "why", "will", "willing", "wish", "with", "within", "without", "won't",
                  "wonder", "would", "would", "wouldn't", "x", "y", "yes", "yet", "you", "you'd", "you'll", "you're",
                  "you've", "your", "yours", "yourself", "yourselves", "z", "zero"}

    # Tokenize the text into terms, removing special symbols
    tokens = [word for word in re.findall(r'\b\w+\b', text) if word not in special_symbols]

    phrases = []

    # Go through each token and check for valid phrases
    for i in range(len(tokens)):
        # Two-term sequences
        if i < len(tokens) - 1 and tokens[i] not in stop_words and tokens[i+1] not in stop_words:
            phrases.append(f"{tokens[i]} {tokens[i+1]}")

        # Three-term sequences
        if i < len(tokens) - 2 and tokens[i] not in stop_words and tokens[i+1] not in stop_words and tokens[i+2] not in stop_words:
            phrases.append(f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}")

    return phrases


# Dictionary to keep track of term IDs and phrases. Global variable.
term_id_map = {}
next_term_id = 1  # Start term IDs from 1.
doc_id_map = {}
phrase_id_map = {}
next_phrase_id = 1


def create_positional_index_for_document(doc_text, doc_id, temp_directory):
    global next_term_id

    # Tokenize the document and get positions
    term_positions = tokenize(doc_text)

    # Create a list to store <term_id, doc_id, positions> entries.
    entries = []

    for term, positions in term_positions.items():
        # Assign a new ID to a term if it doesn't already have one.
        if term not in term_id_map:
            term_id_map[term] = next_term_id
            next_term_id += 1
        term_id = term_id_map[term]

        # Calculate the term frequency (TF) for the current term in the document.
        tf = len(positions)

        entries.append({
            'term_id': term_id,
            'doc_id': doc_id,
            'positions': positions,
            'tf': tf  # Add the term frequency (TF) to the entry
        })

    # Sort the entries based on term_id and doc_id.
    entries = sorted(entries, key=lambda x: (x['term_id'], x['doc_id']))

    # Write the sorted entries to a jsonl file inside the 'temp_files' directory.
    filename = os.path.join(temp_directory, f"doc_{doc_id}.jsonl")
    with open(filename, 'w') as f:
        for entry in entries:
            f.write(json.dumps(entry) + '\n')


def create_phrase_index_for_document(doc_text, doc_id, temp_phrase_directory):
    global next_phrase_id

    # Extract the phrases from the document.
    phrases = get_phrases(doc_text)
    phrase_frequencies = Counter(phrases)

    # Create a list to store <phrase_id, doc_id, frequency> entries.
    entries = []

    for phrase, freq in phrase_frequencies.items():
        # Assign a new ID to a phrase if it doesn't already have one.
        if phrase not in phrase_id_map:
            phrase_id_map[phrase] = next_phrase_id
            next_phrase_id += 1
        phrase_id = phrase_id_map[phrase]

        entries.append({
            'phrase_id': phrase_id,
            'doc_id': doc_id,
            'frequency': freq
        })

    # Sort the entries based on phrase_id and doc_id.
    entries = sorted(entries, key=lambda x: (x['phrase_id'], x['doc_id']))

    # Write the sorted entries to a jsonl file inside the 'temp_phrase_files' directory.
    filename = os.path.join(temp_phrase_directory, f"phrase_doc_{doc_id}.jsonl")
    with open(filename, 'w') as f:
        for entry in entries:
            f.write(json.dumps(entry) + '\n')


n = 0  # total number of documents


def process_documents(directory, temp_directory):
    # Make sure the term_id_map and next_term_id are globally accessible.
    global term_id_map, next_term_id, n, doc_id_map

    doc_counter = 0  # Counter for document IDs

    # Iterate over each document in the directory
    for doc in read_docs(directory):
        doc_counter += 1  # Increment for each new document
        doc_id = "d" + str(doc_counter)  # Constructing the docID in desired format

        # Storing the mapping of doc_id to its original document name
        doc_id_map[doc_id] = doc['DOCNO']
        # print("Storing to doc id map: ", doc['DOCNO'])

        # Call the positional index function for the current document
        create_positional_index_for_document(doc['TEXT'], doc_id, temp_directory)

    print(f"Processed {doc_counter} documents.")
    n = doc_counter


def process_documents_for_phrases(directory, temp_phrase_directory):
    doc_counter = 0

    for doc in read_docs(directory):
        doc_counter += 1
        doc_id = "d" + str(doc_counter)

        create_phrase_index_for_document(doc['TEXT'], doc_id, temp_phrase_directory)

    print(f"Processed {doc_counter} documents for phrase indexing.")


# global list of the intermediate merged files so they can be deleted at the end
intermediate_merged_files = []
intermediate_merged_phrase_files = []


def two_way_merge(file1, file2, memory_limit=None):
    # Merge two sorted run files into a single sorted file.
    output_file_name = "merged_run_{}.jsonl".format(uuid.uuid4())  # Ensure a unique filename
    intermediate_merged_files.append(output_file_name)

    def load_chunk(file, limit):
        # Load a chunk of lines up to the given limit.
        lines = []
        for _ in range(limit):
            line = file.readline()
            if not line:
                break
            lines.append(line)
        return lines

    with open(file1, 'r') as f1, open(file2, 'r') as f2, open(output_file_name, 'w') as output_file:
        while True:
            chunk1 = load_chunk(f1, memory_limit)
            chunk2 = load_chunk(f2, memory_limit)

            if not chunk1 and not chunk2:  # If both chunks are empty, we are done.
                break

            # Merge the two chunks
            merged_chunk = []
            i, j = 0, 0
            while i < len(chunk1) and j < len(chunk2):
                parsed_line1 = json.loads(chunk1[i])
                parsed_line2 = json.loads(chunk2[j])

                if (parsed_line1['term_id'], parsed_line1['doc_id']) < (parsed_line2['term_id'], parsed_line2['doc_id']):
                    merged_chunk.append(chunk1[i])
                    i += 1
                else:
                    merged_chunk.append(chunk2[j])
                    j += 1

            # Append the remaining lines from the chunks
            while i < len(chunk1):
                merged_chunk.append(chunk1[i])
                i += 1
            while j < len(chunk2):
                merged_chunk.append(chunk2[j])
                j += 1

            # Write the merged chunk to the output
            output_file.writelines(merged_chunk)

    return output_file_name



def merge_all_files(files):
    # Recursively merge files using two_way_merge until only one file remains.
    while len(files) > 1:
        new_files = []

        # Merge in pairs
        for i in range(0, len(files), 2):
            if i + 1 < len(files):
                merged_file = two_way_merge(files[i], files[i+1], 1000000)
                new_files.append(merged_file)
            else:
                new_files.append(files[i])  # In case there's an odd number of files

        files = new_files

    return files[0]


def two_way_merge_phrases(file1, file2):
    # Merge two sorted phrase run files into a single sorted file.
    output_file_name = "merged_phrase_run_{}.jsonl".format(uuid.uuid4())  # Ensure a unique filename
    intermediate_merged_phrase_files.append(output_file_name)

    with open(file1, 'r') as f1, open(file2, 'r') as f2, open(output_file_name, 'w') as output_file:
        line1 = f1.readline()
        line2 = f2.readline()

        while line1 and line2:
            parsed_line1 = json.loads(line1)
            parsed_line2 = json.loads(line2)

            # Adjust the conditions for phrase_id and doc_id
            if (parsed_line1['phrase_id'], parsed_line1['doc_id']) < (
            parsed_line2['phrase_id'], parsed_line2['doc_id']):
                output_file.write(line1)
                line1 = f1.readline()
            else:
                output_file.write(line2)
                line2 = f2.readline()

        # If file1 still has content, dump it into the merged file
        while line1:
            output_file.write(line1)
            line1 = f1.readline()

        # If file2 still has content, dump it into the merged file
        while line2:
            output_file.write(line2)
            line2 = f2.readline()

    return output_file_name


def merge_all_phrase_files(files):
    # Recursively merge phrase files using two_way_merge_phrases until only one file remains.
    while len(files) > 1:
        new_files = []

        # Merge in pairs
        for i in range(0, len(files), 2):
            if i + 1 < len(files):
                merged_file = two_way_merge_phrases(files[i], files[i + 1])
                new_files.append(merged_file)
            else:
                new_files.append(files[i])  # In case there's an odd number of files

        files = new_files

    return files[0]


# Single Term Positional Index
def create_inverted_positional_index(merged_filename, term_id_map, output_directory):
    # Create a reverse mapping from term_id to term
    id_term_map = {v: k for k, v in term_id_map.items()}
    inverted_index = {}
    doc_lengths = {}  # This dictionary will store the total term count for each document

    with open(merged_filename, 'r') as f:
        for line in f:
            entry = json.loads(line.strip())
            term_id = entry["term_id"]
            term = id_term_map[term_id]
            doc_info = {
                "doc_id": entry["doc_id"],
                "positions": entry["positions"],
                "tf": entry["tf"]  # Include the term frequency
            }

            # If term is not in inverted_index yet, add it.
            if term not in inverted_index:
                inverted_index[term] = {"term": term, "postings": [doc_info]}
            else:
                inverted_index[term]["postings"].append(doc_info)

    # Calculate df, idf and tf-idf for each term and store them in the inverted index
    for term, data in inverted_index.items():
        df = len(data["postings"])  # df is the length of the postings list for the term
        idf = math.log(n / df)  # Using the global variable 'n'
        data["idf"] = idf  # Storing idf in the inverted index

        # Compute the tf-idf weight for each document where the term appears
        for posting in data["postings"]:
            tf = posting["tf"]
            posting["tf_idf"] = tf * idf  # Adding the tf-idf weight to each posting

            # Accumulate the term frequencies to compute document lengths
            doc_id = posting["doc_id"]
            if doc_id in doc_lengths:
                doc_lengths[doc_id] += tf
            else:
                doc_lengths[doc_id] = tf

            avg_doc_length = sum(doc_lengths.values()) / len(doc_lengths)

            data_to_save = {
                "doc_lengths": doc_lengths,
                "avg_doc_length": avg_doc_length
            }

            # Create the full path to save the data
            file_path = os.path.join(output_directory, "data.pkl")
            with open(file_path, "wb") as f:
                pickle.dump(data_to_save, f)

    # Writing the inverted index to a new file
    output_file_name = os.path.join(output_directory, "inverted_positional_index.jsonl")
    with open(output_file_name, 'w') as f:
        for term, data in inverted_index.items():
            f.write(json.dumps(data) + "\n")

    print(f"Inverted positional index written to {output_file_name}")


def create_inverted_phrase_index(merged_phrase_filename, phrase_id_map, output_directory, n, threshold=10):
    # Create a reverse mapping from phrase_id to phrase
    id_phrase_map = {v: k for k, v in phrase_id_map.items()}
    inverted_phrase_index = {}
    overall_phrase_frequencies = defaultdict(int)
    document_frequencies = defaultdict(set)  # Store unique doc_ids for each phrase to calculate df

    with open(merged_phrase_filename, 'r') as f:
        for line in f:
            entry = json.loads(line.strip())
            phrase_id = entry["phrase_id"]
            phrase = id_phrase_map[phrase_id]
            doc_id = entry["doc_id"]
            frequency = entry["frequency"]

            # Add doc_id to the set corresponding to the phrase for df calculation
            document_frequencies[phrase].add(doc_id)

            # Update the overall frequency for this phrase.
            overall_phrase_frequencies[phrase] += frequency

            # Prepare the document info with frequency (TF)
            doc_info = {"doc_id": doc_id, "tf": frequency}

            # If phrase is not in inverted_phrase_index yet, add it.
            if phrase not in inverted_phrase_index:
                inverted_phrase_index[phrase] = {"phrase": phrase, "postings": [doc_info]}
            else:
                inverted_phrase_index[phrase]["postings"].append(doc_info)

    # Calculate DF, IDF, and TF-IDF for each phrase
    for phrase, data in inverted_phrase_index.items():
        # df = len(document_frequencies[phrase])
        df = len(data["postings"])
        n = 1768
        idf = math.log(n / df)  # Calculate IDF value
        data['df'] = df  # Store the document frequency
        data['idf'] = idf  # Store raw IDF value for each phrase
        # Calculate TF-IDF for each document the phrase appears in
        for posting in data["postings"]:
            tf = posting["tf"]
            posting["tf_idf"] = tf * idf  # Store the TF-IDF weight

    # Writing the inverted index for phrases to a new file
    output_file_name = os.path.join(output_directory, "inverted_phrase_index.jsonl")
    with open(output_file_name, 'w') as f:
        for phrase, data in list(inverted_phrase_index.items()):
            # Check the threshold before writing to the file.
            if overall_phrase_frequencies[phrase] >= threshold:
                f.write(json.dumps(data) + "\n")
            else:
                # Remove this phrase from the index if it doesn't surpass the threshold
                del inverted_phrase_index[phrase]

    print(f"Inverted phrase index written to {output_file_name}")


# Single Term Index without stop words
def create_single_term_index_without_stopwords(merged_filename, term_id_map, output_directory):
    stopwords = [
        "a", "a's", "able", "about", "above", "according", "accordingly", "across", "actually", "after", "afterwards",
        "again", "against", "ain't", "all", "allow", "allows", "almost", "alone", "along", "already", "also",
        "although", "always", "am", "among", "amongst", "an", "another", "any", "anybody", "anyhow", "anyone",
        "anything", "anyway", "anyways", "anywhere", "apart", "appear", "appreciate", "appropriate", "are", "aren't",
        "around", "as", "aside", "ask", "asking", "associated", "at", "available", "away", "awfully", "b", "be",
        "became",
        "because", "become", "becomes", "becoming", "been", "before", "beforehand", "behind", "being", "believe",
        "below",
        "beside", "besides", "best", "better", "between", "beyond", "both", "brief", "but", "by", "c", "c'mon", "c's",
        "came",
        "can", "can't", "cannot", "cant", "cause", "causes", "certain", "certainly", "changes", "clearly", "co", "com",
        "come",
        "comes", "concerning", "consequently", "consider", "considering", "contain", "containing", "contains",
        "corresponding",
        "could", "couldn't", "course", "currently", "d", "definitely", "described", "despite", "did", "didn't",
        "different",
        "do", "does", "doesn't", "doing", "don't", "done", "down", "downwards", "during", "e", "each", "edu", "eg",
        "eight",
        "either", "else", "elsewhere", "enough", "entirely", "especially", "et", "etc", "even", "ever", "every",
        "everybody",
        "everyone", "everything", "everywhere", "ex", "exactly", "example", "except", "f", "far", "few", "fifth",
        "first",
        "five", "followed", "following", "follows", "for", "former", "formerly", "forth", "four", "from", "further",
        "furthermore", "g", "get", "gets", "getting", "given", "gives", "go", "goes", "going", "gone", "got", "gotten",
        "greetings", "h", "had", "hadn't", "happens", "hardly", "has", "hasn't", "have", "haven't", "having", "he",
        "he's",
        "hello", "help", "hence", "her", "here", "here's", "hereafter", "hereby", "herein", "hereupon", "hers",
        "herself",
        "hi", "him", "himself", "his", "hither", "hopefully", "how", "howbeit", "however", "i", "i'd", "i'll", "i'm",
        "i've",
        "ie", "if", "ignored", "immediate", "in", "inasmuch", "inc", "indeed", "indicate", "indicated", "indicates",
        "inner",
        "insofar", "instead", "into", "inward", "is", "isn't", "it", "it'd", "it'll", "it's", "its", "itself", "j",
        "just",
        "k", "keep", "keeps", "kept", "know", "knows", "known", "l", "last", "lately", "later", "latter", "latterly",
        "least", "less", "lest", "let", "let's", "like", "liked", "likely", "little", "look", "looking", "looks", "ltd",
        "m",
        "mainly", "many", "may", "maybe", "me", "mean", "meanwhile", "merely", "might", "more", "moreover", "most",
        "mostly",
        "much", "must", "my", "myself", "n", "name", "namely", "nd", "near", "nearly", "necessary", "need", "needs",
        "neither",
        "never", "nevertheless", "new", "next", "nine", "no", "nobody", "non", "none", "noone", "nor", "normally",
        "not",
        "nothing", "novel", "now", "nowhere", "o", "obviously", "of", "off", "often", "oh", "ok", "okay", "old", "on",
        "once",
        "one", "ones", "only", "onto", "or", "other", "others", "otherwise", "ought", "our", "ours", "ourselves", "out",
        "outside", "over", "overall", "own", "p", "particular", "particularly", "per", "perhaps", "placed", "please",
        "plus",
        "possible", "presumably", "probably", "provides", "q", "que", "quite", "qv", "r", "rather", "rd", "re",
        "really",
        "reasonably", "regarding", "regardless", "regards", "relatively", "respectively", "right", "s", "said", "same",
        "saw",
        "say", "saying", "says", "second", "secondly", "see", "seeing", "seem", "seemed", "seeming", "seems", "seen",
        "self",
        "selves", "sensible", "sent", "serious", "seriously", "seven", "several", "shall", "she", "should", "shouldn't",
        "since",
        "six", "so", "some", "somebody", "somehow", "someone", "something", "sometime", "sometimes", "somewhat",
        "somewhere",
        "soon", "sorry", "specified", "specify", "specifying", "still", "sub", "such", "sup", "sure", "t", "t's",
        "take", "taken",
        "tell", "tends", "th", "than", "thank", "thanks", "thanx", "that", "that's", "thats", "the", "their", "theirs",
        "them",
        "themselves", "then", "thence", "there", "there's", "thereafter", "thereby", "therefore", "therein", "theres",
        "thereupon",
        "these", "they", "they'd", "they'll", "they're", "they've", "think", "third", "this", "thorough", "thoroughly",
        "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "took", "toward",
        "towards", "tried", "tries", "truly", "try", "trying", "twice", "two", "u", "un", "under"]
    # Create a reverse mapping from term_id to term
    id_term_map = {v: k for k, v in term_id_map.items()}
    stopword_ids = {term_id_map[word] for word in stopwords if word in term_id_map}
    inverted_index = {}
    doc_freqs = {}  # Document frequency for each term
    doc_lengths = {}  # Store total term count for each document

    with open(merged_filename, 'r') as f:
        for line in f:
            entry = json.loads(line.strip())
            term_id = entry["term_id"]
            if term_id in stopword_ids:
                continue

            term = id_term_map[term_id]
            doc_id = entry["doc_id"]
            term_freq = len(entry["positions"])

            # Update document frequencies and document lengths
            doc_freqs[term] = doc_freqs.get(term, 0) + 1
            doc_lengths[doc_id] = doc_lengths.get(doc_id, 0) + term_freq

            if term not in inverted_index:
                inverted_index[term] = {"term": term, "postings": []}

            inverted_index[term]["postings"].append({"doc_id": doc_id, "term_freq": term_freq})

    # Calculate IDF for each term and update TF-IDF for each posting
    for term, data in inverted_index.items():
        df = doc_freqs[term]
        idf = math.log(n / df)
        data["idf"] = idf  # Store IDF in the inverted index

        for posting in data["postings"]:
            posting["tf_idf"] = posting["term_freq"] * idf

    # Write the inverted index to a file
    output_file_name = os.path.join(output_directory, "single_term_index_without_stopwords.jsonl")
    with open(output_file_name, 'w') as f:
        for term, data in inverted_index.items():
            f.write(json.dumps(data) + "\n")

    # Save additional data like document lengths
    data_to_save = {
        "doc_lengths": doc_lengths,
        "avg_doc_length": sum(doc_lengths.values()) / len(doc_lengths)
    }
    file_path = os.path.join(output_directory, "additional_data.pkl")
    with open(file_path, "wb") as f:
        pickle.dump(data_to_save, f)

    print(f"Single term index without stop words written to {output_file_name}")


# Create the stem index
def create_stem_index(merged_filename, term_id_map, output_directory):
    stopwords = ["is", "the", "and", "a", "an", "of", "in", "to", "for", "with", "on", "at", "by", "from", "about"]

    # Create a reverse mapping from term_id to term
    id_term_map = {v: k for k, v in term_id_map.items()}

    # Identify term_ids of stopwords
    stopword_ids = {term_id_map[word] for word in stopwords if word in term_id_map}

    inverted_index = {}

    with open(merged_filename, 'r') as f:
        for line in f:
            entry = json.loads(line.strip())
            term_id = entry["term_id"]
            term = id_term_map[term_id]  # Get the actual term
            stemmed_term = ps.stem(term)  # Stem the term

            # Skip stop words
            if term_id in stopword_ids:
                continue

            doc_info = {"doc_id": entry["doc_id"], "tf": len(entry["positions"])}

            # If stemmed_term is not in inverted_index yet, add it.
            if stemmed_term not in inverted_index:
                inverted_index[stemmed_term] = {"term": stemmed_term, "postings": [doc_info]}
            else:
                inverted_index[stemmed_term]["postings"].append(doc_info)

    # Calculate df, idf and tf-idf for each term and store in inverted index
    for term, data in inverted_index.items():
        df = len(data["postings"])  # df is the length of the postings list for the term
        idf = math.log(n / df)  # Using the global variable 'n'
        data["idf"] = idf  # Storing idf in the inverted index

        # Compute the tf-idf weight for each document where the term appears
        for posting in data["postings"]:
            tf = posting["tf"]
            posting["tf_idf"] = tf * idf  # Adding the tf-idf weight to each posting

    # Writing the inverted index to a new file
    output_file_name = os.path.join(output_directory, "stem_index.jsonl")
    with open(output_file_name, 'w') as f:
        for term, data in inverted_index.items():
            f.write(json.dumps(data) + "\n")

    print(f"Stem index written to {output_file_name}")


def create_document_vectors(inverted_index_path):
    print("creating doc vectors")
    # Load the inverted index
    with open(inverted_index_path, 'rt') as f:
        inverted_index = {entry['term']: entry for entry in map(json.loads, f)}

    # Determine the total number of terms
    total_terms = len(inverted_index)

    # Initialize document vectors for all terms
    document_vectors = defaultdict(lambda: [0] * total_terms)

    # Create a mapping of term to index
    term_to_index = {term: idx for idx, term in enumerate(inverted_index.keys())}

    # Update vectors with TF-IDF weights for all terms in each document
    for term, data in inverted_index.items():
        term_idx = term_to_index[term]
        for posting in data["postings"]:
            doc_id = posting["doc_id"]
            tf_idf_weight = posting["tf_idf"]
            document_vectors[doc_id][term_idx] = tf_idf_weight

    print("finished creating doc vectors")
    return document_vectors


def initialize_query_vector(all_query_terms):
    # Initialize a vector of zeros with length of all_query_terms
    return [0] * len(all_query_terms)


def populate_query_vector(query, inverted_index_path):
    # Load the inverted index to get all terms
    with open(inverted_index_path, 'rt') as f:
        inverted_index = {entry['term']: entry for entry in map(json.loads, f)}

    # Initialize the query vector with zeros for all terms
    total_terms = len(inverted_index)
    query_vector = [0] * total_terms

    # Create a mapping of term to index
    term_to_index = {term: idx for idx, term in enumerate(inverted_index.keys())}

    # Populate the vector with TF-IDF weights for terms present in the query
    for term in query['title']:
        if term in term_to_index:
            term_idx = term_to_index[term]
            query_vector[term_idx] = query['tf_idf_weights'].get(term, 0)

    return query_vector


def cosine_similarity(vector_a, vector_b):
    # Compute the dot product
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))

    # Compute the magnitudes (or lengths) of the two vectors
    magnitude_a = sum(a ** 2 for a in vector_a) ** 0.5
    magnitude_b = sum(b ** 2 for b in vector_b) ** 0.5

    # Avoid division by zero
    if not magnitude_a or not magnitude_b:
        return 0.0

    # Compute the cosine similarity
    similarity = dot_product / (magnitude_a * magnitude_b)
    return similarity


def bm25(query, doc_id, inverted_index, doc_lengths, avg_doc_length, k1=1.2, k2=1000, b=0.75):
    score = 0.0
    total_docs = len(doc_lengths)

    for term, positions in query["title"].items():
        # qtf_j: Term frequency of term j in the query
        qtf_j = len(positions)

        # Check if the term exists in the inverted index
        if term in inverted_index:
            postings_list = inverted_index[term]["postings"]

            # n: Number of documents containing term
            n = len(postings_list)

            # tf_ij: Term frequency of term j in document i
            tf_ij = next((posting["term_freq"] for posting in postings_list if posting["doc_id"] == doc_id), 0)

            # IDF weight for the term
            w = math.log((total_docs - n + 0.5) / (n + 0.5))

            # Term's contribution to the score using BM25 formula
            term_score = w * ((k1 + 1) * tf_ij) / (
                        tf_ij + k1 * (1 - b + b * (doc_lengths[doc_id] / avg_doc_length))) * ((k2 + 1) * qtf_j) / (
                                     k2 + qtf_j)
            score += term_score

    return score


def bm25phrase(query, doc_id, inverted_index, doc_lengths, avg_doc_length, k1=1.2, k2=1000, b=0.75):
    score = 0.0
    total_docs = len(doc_lengths)

    for phrase in query['title']:  # Assuming 'title' holds the phrases from the query
        # Check if the phrase exists in the inverted index
        if phrase in inverted_index:
            postings_list = inverted_index[phrase]["postings"]
            df = inverted_index[phrase]["df"]
            idf = inverted_index[phrase]["idf"]

            # tf_ij: Term frequency of phrase j in document i
            tf_ij = next((posting["tf"] for posting in postings_list if posting["doc_id"] == doc_id), 0)

            # Document length normalization
            doc_length = doc_lengths.get(doc_id, 0)
            len_norm = (1 - b) + b * (doc_length / avg_doc_length)

            # Term's contribution to the score using BM25 formula
            term_score = idf * ((tf_ij * (k1 + 1)) / (tf_ij + k1 * len_norm))
            score += term_score

    return score


def bm25stem(query, doc_id, inverted_index, doc_lengths, avg_doc_length, k1=1.2, b=0.75):
    score = 0.0
    total_docs = len(doc_lengths)

    for term, qtf_j in query["title"].items():  # qtf_j: Term frequency in the query
        # Check if the term exists in the inverted index
        if term in inverted_index:
            postings_list = inverted_index[term]["postings"]
            # n: Number of documents containing term
            n = len(postings_list)

            # tf_ij: Term frequency of term j in document i
            tf_ij = next((posting["tf"] for posting in postings_list if posting["doc_id"] == doc_id), 0)

            # Ensure the argument inside the log function is greater than zero
            log_argument = (total_docs - n + 0.5) / (n + 0.5)
            if log_argument <= 0:
                continue  # Skip this term if the log argument is not positive

            # IDF weight for the term
            idf = math.log(log_argument)

            # Term's contribution to the score using BM25 formula
            term_score = idf * ((k1 + 1) * tf_ij) / (
                tf_ij + k1 * (1 - b + b * (doc_lengths.get(doc_id, 0) / avg_doc_length))) * ((k1 + 1) * qtf_j) / (
                k1 + qtf_j)
            score += term_score

    return score


def compute_bm25_scores_for_all_queries(queries, inverted_index, doc_lengths, avg_doc_length):
    results = {}

    # Load the inverted positional index from the provided path
    with open(inverted_index, 'rt') as f:
        inverted_index = {entry['term']: entry for entry in map(json.loads, f)}

    # Loop through each query
    for query in queries:
        query_num = query["num"]
        results[query_num] = {}  # Initialize a dictionary to store scores for this query

        # Loop through all documents
        for doc_id in doc_lengths.keys():
            # Compute BM25 score for this document and query
            score = bm25stem(query, doc_id, inverted_index, doc_lengths, avg_doc_length)
            results[query_num][doc_id] = score

    return results


def compute_bm25_scores_for_all_queries_single(queries, inverted_index, doc_lengths, avg_doc_length):
    results = {}

    # Load the inverted positional index from the provided path
    with open(inverted_index, 'rt') as f:
        inverted_index = {entry['term']: entry for entry in map(json.loads, f)}

    # Loop through each query
    for query in queries:
        query_num = query["num"]
        results[query_num] = {}  # Initialize a dictionary to store scores for this query

        # Loop through all documents
        for doc_id in doc_lengths.keys():
            # Compute BM25 score for this document and query
            score = bm25(query, doc_id, inverted_index, doc_lengths, avg_doc_length)
            results[query_num][doc_id] = score

    return results


def rank_and_write_top_scores(results, output_directory, doc_id_map_dir):
    # Construct the path to the doc_id_map.txt file
    doc_id_map_file = os.path.join(doc_id_map_dir, "doc_id_map.txt")

    # Read the file and reconstruct the dictionary
    full_doc_id_map = {}
    with open(doc_id_map_file, 'r') as f:
        for line in f:
            key, value = line.strip().split(': ')
            full_doc_id_map[key] = value

    # File path for saving results
    results_file_path = output_directory

    # Open the results file for writing
    with open(results_file_path, 'w') as f:
        for query_num, scores in results.items():
            # Sort the scores for this query
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

            # Write the top 100 scores for this query to the results file
            for i, (doc_id, score) in enumerate(sorted_scores[:100]):
                doc_name = full_doc_id_map[str(doc_id)]
                f.write(f"{query_num} 0 {doc_name} {i+1} {score:.6f} BM25\n")

    print(f"BM25 results saved to {results_file_path}")


def rank_and_write_top_scores_reduction(results, output_directory, doc_id_map_dir):
    # Construct the path to the doc_id_map.txt file
    doc_id_map_file = os.path.join(doc_id_map_dir, "doc_id_map.txt")

    # Read the file and reconstruct the dictionary
    full_doc_id_map = {}
    with open(doc_id_map_file, 'r') as f:
        for line in f:
            key, value = line.strip().split(': ')
            full_doc_id_map[key] = value

    # File path for saving results
    results_file_path = output_directory

    # Open the results file for writing
    with open(results_file_path, 'w') as f:
        for query_num, scores in results.items():
            # Sort the scores for this query
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

            # Write the top 100 scores for this query to the results file
            for i, (doc_id, score) in enumerate(sorted_scores[:100]):
                doc_name = full_doc_id_map[str(doc_id)]
                f.write(f"{query_num} 0 {doc_name} {i+1} {score:.6f} BM25_Reduction\n")

    print(f"BM25 query reduction results saved to {results_file_path}")


def retrieve_top_20_docs(results_file):
    print("retrieving top 20 docs")
    top_docs_per_query = {}

    with open(results_file, 'r') as file:
        for line in file:
            parts = line.split()
            query_id, _, doc_id, rank, score, _ = parts
            rank = int(rank)  # Convert rank to an integer for comparison

            # Initialize the list for this query_id if it hasn't been already
            if query_id not in top_docs_per_query:
                top_docs_per_query[query_id] = []

            # If we don't have 20 documents yet, or if this document has a higher score than the last one
            if rank <= 20:
                # Append the document ID and its score as a tuple
                top_docs_per_query[query_id].append((doc_id, float(score)))

    return top_docs_per_query


def calculate_top_terms_for_documents(top_docs, doc_term_index, idf_dict, doc_id_map_dir, n_values, m_value):
    print("Calculating top terms for documents...")

    # Load and invert the document ID mapping
    doc_id_map_file = os.path.join(doc_id_map_dir, "doc_id_map.txt")
    doc_id_map = {}
    with open(doc_id_map_file, 'r') as f:
        for line in f:
            short_id, full_name = line.strip().split(': ')
            doc_id_map[full_name] = short_id

    top_terms_per_query = {}

    for query_id, docs in top_docs.items():
        top_terms_per_query[query_id] = {}

        for N in n_values:
            # print(f"  Using top N value: {N}")
            top_N_docs = [doc_id_map[doc_name] for doc_name, _ in docs[:N] if doc_name in doc_id_map]

            term_scores = {}

            for doc_id in top_N_docs:
                if doc_id in doc_term_index:
                    doc_terms = doc_term_index[doc_id]
                    for term, term_data in doc_terms.items():
                        term_freq = term_data['term_freq']

                        # Initialize or update term_scores with num.idf and num.ntf.idf
                        if term not in term_scores:
                            term_scores[term] = {'num.idf': 0, 'num.ntf.idf': 0}

                        term_scores[term]['num.idf'] += idf_dict.get(term, 0)
                        term_scores[term]['num.ntf.idf'] += term_freq * idf_dict.get(term, 0)

            # Sort and get the top M terms for num.idf and num.ntf.idf
            top_terms_num_idf = sorted(term_scores.items(), key=lambda item: item[1]['num.idf'], reverse=True)[:m_value]
            top_terms_num_ntf_idf = sorted(term_scores.items(), key=lambda item: item[1]['num.ntf.idf'], reverse=True)[:m_value]

            top_terms_per_query[query_id][N] = {
                'top_terms_num_idf': top_terms_num_idf,
                'top_terms_num_ntf.idf': top_terms_num_ntf_idf
            }

    return top_terms_per_query


def create_doc_term_index_and_idf_dict(inverted_index_path, output_directory):
    index_path = os.path.join(output_directory, inverted_index_path)
    print(f"index path = {index_path}")

    # Load the inverted index
    with open(index_path, 'rt') as f:
        inverted_index = {entry['term']: entry for entry in map(json.loads, f)}

    # Create an empty doc-term index and IDF dictionary
    doc_term_index = {}
    idf_dict = {}

    # Iterate over the inverted index to populate the doc-term index and IDF dictionary
    for term, data in inverted_index.items():
        # Populate the IDF dictionary
        idf_dict[term] = data['idf']

        for posting in data['postings']:
            doc_id = posting['doc_id']
            term_freq = posting['term_freq']
            tf_idf = posting['tf_idf']

            # Initialize the document entry if it doesn't exist
            if doc_id not in doc_term_index:
                doc_term_index[doc_id] = {}

            # Add the term information to the document entry
            doc_term_index[doc_id][term] = {'term_freq': term_freq, 'tf_idf': tf_idf}

    # Write the doc-term index to a file
    doc_term_index_file_path = os.path.join(output_directory, "doc_term_index.json")
    with open(doc_term_index_file_path, 'w') as f:
        json.dump(doc_term_index, f)
    print(f"Doc-term index written to {doc_term_index_file_path}")

    # Write the IDF dictionary to a file
    idf_dict_file_path = os.path.join(output_directory, "idf_dict.json")
    with open(idf_dict_file_path, 'w') as f:
        json.dump(idf_dict, f)
    print(f"IDF dictionary written to {idf_dict_file_path}")


def expand_and_rescore_queries(queries, top_terms, idf_dict, n_values, m_values, inverted_index, doc_lengths, avg_doc_length):
    expanded_query_results = {}

    # Iterate through each N value
    for N in n_values:
        for M in m_values:
            # For num.idf and num.ntf.idf
            for criteria in ['num_idf', 'num_ntf.idf']:
                expanded_queries = []

                # Go through each query
                for query in queries:
                    query_id = query['num']
                    expanded_query = deepcopy(query)  # Deep copy to avoid modifying the original query

                    # Add the top M terms for the current criteria
                    feedback_terms = [term for term, _ in top_terms[str(query_id)][N][f'top_terms_{criteria}'][:M]]
                    for term in feedback_terms:
                        if term not in expanded_query['title']:
                            expanded_query['title'][term] = [
                                len(expanded_query['title']) + 1]  # Add position for the new term
                            expanded_query['tf_idf_weights'][term] = idf_dict.get(term,
                                                                                  0)  # Use IDF as the TF-IDF value

                    expanded_queries.append(expanded_query)

                    # Print expanded query for inspection
                    # print(f"Expanded Query for N={N}, M={M}, Criteria={criteria}, Query ID={query_id}: {expanded_query}")

                # Compute BM25 scores for the expanded queries
                result_key = f"N={N}, M={M}, {criteria}"
                expanded_query_results[result_key] = compute_bm25_scores_for_all_queries_single(expanded_queries, inverted_index, doc_lengths, avg_doc_length)

    return expanded_query_results


def rank_and_write_all_top_prf_scores(expanded_query_results, output_directory, doc_id_map_dir):
    # Construct the path to the doc_id_map.txt file
    doc_id_map_file = os.path.join(doc_id_map_dir, "doc_id_map.txt")

    # Read the file and reconstruct the dictionary
    full_doc_id_map = {}
    with open(doc_id_map_file, 'r') as f:
        for line in f:
            key, value = line.strip().split(': ')
            full_doc_id_map[key] = value

    # Iterate through each set of results in expanded_query_results
    for criteria, results in expanded_query_results.items():
        # Construct a unique filename for each criteria
        sanitized_criteria = criteria.replace(" ", "_").replace(",", "")
        results_file_path = os.path.join(output_directory, f"results_{sanitized_criteria}.txt")

        # Open the results file for writing
        with open(results_file_path, 'w') as f:
            for query_num, scores in results.items():
                # Sort the scores for this query
                sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

                # Write the top 100 scores for this query to the results file
                for i, (doc_id, score) in enumerate(sorted_scores[:100]):
                    doc_name = full_doc_id_map.get(str(doc_id), "Unknown_Doc_ID")
                    f.write(f"{query_num} 0 {doc_name} {i+1} {score:.6f} BM25_{sanitized_criteria}\n")

        print(f"BM25 results for {criteria} saved to {results_file_path}")


def rank_and_write_top_scores_dynamic(results, output_directory, doc_id_map_dir):
    # Construct the path to the doc_id_map.txt file
    doc_id_map_file = os.path.join(doc_id_map_dir, "doc_id_map.txt")

    # Read the file and reconstruct the dictionary
    full_doc_id_map = {}
    with open(doc_id_map_file, 'r') as f:
        for line in f:
            key, value = line.strip().split(': ')
            full_doc_id_map[key] = value

    # File path for saving results
    results_file_path = output_directory

    with open(results_file_path, 'w') as f:
        for query_num, scores in results.items():
            # Convert scores to a list of (doc_id, (score, index_type)) tuples
            scores_list = list(scores.items())
            # Sort the scores_list by score in descending order
            sorted_scores = sorted(scores_list, key=lambda x: x[1][0], reverse=True)

            for i, (doc_id, (score, index_type)) in enumerate(sorted_scores[:100]):
                doc_name = full_doc_id_map[str(doc_id)]
                f.write(f"{query_num} 0 {doc_name} {i + 1} {score:.6f} {index_type}_BM25\n")

    print(f"BM25 dynamic results saved to {results_file_path}")


def compute_rankings(queries, document_vectors, doc_id_map_dir):
    rankings = {}
    print("compute rankings function")
    # Construct the path to the doc_id_map.txt file
    doc_id_map_file = os.path.join(doc_id_map_dir, "doc_id_map.txt")

    # Read the file and reconstruct the dictionary
    full_doc_id_map = {}
    with open(doc_id_map_file, 'r') as f:
        for line in f:
            key, value = line.strip().split(': ')
            full_doc_id_map[key] = value

    for query in queries:
        # Get the query vector
        print("for query in queries")
        query_vector = query['vector']

        # Compute cosine similarity scores for this query against all documents
        scores = {doc_id: cosine_similarity(query_vector, doc_vector)
                  for doc_id, doc_vector in document_vectors.items()}

        # Rank the documents for this query based on the similarity scores
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Convert docIDs in the sorted scores to original document names
        sorted_scores_with_original_names = [(full_doc_id_map[doc_id], score) for doc_id, score in sorted_scores]

        # Store the top 100 ranked documents for this query
        rankings[query['num']] = sorted_scores_with_original_names[:100]

    return rankings


def write_rankings_to_file(rankings, output_dir):
    # Create the output filepath
    output_filepath = output_dir

    # Open the file in write mode
    with open(output_filepath, 'w') as file:
        # Iterate over the rankings
        for query_id, ranked_docs in rankings.items():
            for i, (doc_id, score) in enumerate(ranked_docs):
                # Write the ranking in the desired format.
                file.write(f"{query_id} 0 {doc_id} {i+1} {score:.6f} COSINE\n")

    print(f"Rankings written to {output_filepath}")


def rank_docs_for_query_lm(query_terms, inverted_index, doc_lengths, mu):
    scores = {}  # to hold scores of documents
    collection_length = sum(doc_lengths.values())

    for term in query_terms:
        # If the term doesn't exist in the inverted index, skip
        if term not in inverted_index:
            continue

        term_freq_collection = sum(item['tf'] for item in inverted_index[term]['postings'])
        term_freq_documents = {item['doc_id']: item['tf'] for item in inverted_index[term]['postings']}

        for doc_id, length in doc_lengths.items():
            tf_qi_D = term_freq_documents.get(doc_id, 0)  # get tf of term in doc, 0 if term not in doc

            # Use the Dirichlet smoothed probability formula
            prob = (tf_qi_D + mu * (term_freq_collection / collection_length)) / (length + mu)
            scores[doc_id] = scores.get(doc_id, 0) + math.log(prob)

    # Sort documents by scores in descending order and take top 100
    ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:100]
    return ranked_docs


def process_all_queries(queries, inverted_index_path, doc_lengths, avg_doc_length, output_file_path, doc_id_map_dir):
    # Load the inverted index directly within the function
    with open(inverted_index_path, 'rt') as f:
        inverted_index = {entry['term']: entry for entry in map(json.loads, f)}

    all_rankings = {}

    # Construct the path to the doc_id_map.txt file
    doc_id_map_file = os.path.join(doc_id_map_dir, "doc_id_map.txt")

    # Read the file and reconstruct the dictionary
    full_doc_id_map = {}
    with open(doc_id_map_file, 'r') as f:
        for line in f:
            key, value = line.strip().split(': ')
            full_doc_id_map[key] = value

    for query in queries:
        query_num = query['num']
        query_terms = list(query['title'].keys())

        # Rank docs for this query
        ranked_docs = rank_docs_for_query_lm(query_terms, inverted_index, doc_lengths, avg_doc_length)

        # Store rankings for this query number
        all_rankings[query_num] = ranked_docs

    # Write to output file
    with open(output_file_path, 'w') as f:
        for query_num, rankings in all_rankings.items():
            top_100_docs = rankings[:100]
            for rank, (doc_id, score) in enumerate(top_100_docs, start=1):
                doc_name = full_doc_id_map[str(doc_id)]
                f.write(f"{query_num} 0 {doc_name} {rank} {score:.6f} LM\n")

    print(f"Rankings written to {output_file_path}")


def check_document_frequency(phrase, phrase_index):
    # Load the inverted index to access the idf values
    # Look up the DF of the phrase in the phrase index
    df = phrase_index.get(phrase, {}).get("df", 0)
    is_high_df = df > 7
    return is_high_df


def main_query_processing_dynamic(query_file, phrase_index, positional_index, doc_lengths, avg_doc_length):
    queries = list(read_queries_dynamic(query_file, phrase_index))
    queries_positional = list(read_queries_single(query_file, positional_index))
    # print("Queries: ", queries)

    results = {}

    with open(phrase_index, 'rt') as f:
        inverted_phrase_index = {entry['phrase']: entry for entry in map(json.loads, f)}

    with open(positional_index, 'rt') as f:
        inverted_positional_index = {entry['term']: entry for entry in map(json.loads, f)}

    for query in queries:
        query_num = query["num"]
        results[query_num] = {}
        documents_scored = 0

        # Find the corresponding positional query by 'num'
        query_positional = next((q for q in queries_positional if q['num'] == query_num), None)

        # Calculate phrase and positional scores for each document
        for doc_id in doc_lengths.keys():
            # Phrase score
            phrase_score = sum(
                bm25phrase(query, doc_id, inverted_phrase_index, doc_lengths, avg_doc_length) for phrase in
                query['title'] if phrase in inverted_phrase_index)
            # Positional score
            positional_score = bm25(query_positional, doc_id, inverted_positional_index, doc_lengths, avg_doc_length)

            # Decide which score to take (phrase score if it was calculated, else positional)
            if phrase_score > 0:
                score_type = 'phrase'
                score_to_use = phrase_score
            else:
                score_type = 'positional'
                score_to_use = positional_score

            # Keep track of scores
            documents_scored += 1
            if score_to_use > 0:
                if doc_id not in results[query_num]:
                    results[query_num][doc_id] = (score_to_use, score_type)

    # Return the intermediate results
    return results


def main():
    # print("Testing tokenization:")

    # sampleText = "Weight Control/Diets"

    # tokenized_output = tokenize(sampleText)
    # print(tokenized_output)

    # Initialize the main parser
    parser = argparse.ArgumentParser(description="Information Retrieval System")

    # Create subparsers for different modes of operation
    subparsers = parser.add_subparsers(dest="command")

    # Parser for the build command
    parser_build = subparsers.add_parser("build", help="Builds an index from the provided TREC files.")
    parser_build.add_argument("trec_files_directory_path", type=str,
                              help="Path to the directory containing raw documents.")
    parser_build.add_argument("index_type", type=str, choices=["single", "stem", "phrase", "positional", "doc_term"],
                              help="Type of index to be built.")
    parser_build.add_argument("output_dir", type=str,
                              help="Directory where the index and lexicon files will be written.")

    # Parser for the query_static command
    parser_query = subparsers.add_parser("query_static", help="Queries the index and outputs results.")
    parser_query.add_argument("index_directory_path", type=str, help="Path to the directory containing the index.")
    parser_query.add_argument("query_file_path", type=str, help="Path to the query file.")
    parser_query.add_argument("retrieval_model", type=str, choices=["cosine", "bm25", "lm"],
                              help="Retrieval model to be used.")
    parser_query.add_argument("index_type", type=str, choices=["single", "stem", "phrase", "positional"],
                              help="Type of index to be used.")
    parser_query.add_argument("results_file", type=str, help="Path to the output file for the query results.")
    parser_query.add_argument("ranking_type", type=str, choices=["regular", "prf", "reduction", "prf_reduction"],
                              help="Type of ranking to be used.")

    # Parser for the query_dynamic command
    parser_query = subparsers.add_parser("query_dynamic", help="Queries the index and outputs results.")
    parser_query.add_argument("index_directory_path", type=str, help="Path to the directory containing the index.")
    parser_query.add_argument("query_file_path", type=str, help="Path to the query file.")
    parser_query.add_argument("results_file", type=str, help="Path to the output file for the query results.")

    # Parse the arguments
    args = parser.parse_args()

    # Check which command is chosen and perform actions accordingly
    if args.command == "build":
        # Code for building the index
        temp_directory = os.path.join(args.output_dir, "temp_files")
        if not os.path.exists(temp_directory):
            os.makedirs(temp_directory)

        temp_phrase_directory = os.path.join(args.output_dir, "temp_phrase_files")
        if not os.path.exists(temp_phrase_directory):
            os.makedirs(temp_phrase_directory)

        merged_filename = os.path.join(args.output_dir, "merged_run.jsonl")

        # Create the output directory if it does not exist
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)  # Create new output directory

        # Create/clear the temp directory
        if os.path.exists(temp_directory):
            shutil.rmtree(temp_directory)  # This will delete the directory and its contents if it exists
        os.makedirs(temp_directory)  # Create new empty temp directory

        # Depending on the index type, call the necessary function(s)
        if args.index_type == "phrase":
            initial_time = time.time()
            process_documents_for_phrases(args.trec_files_directory_path, temp_phrase_directory)
            temp_phrase_files = [os.path.join(temp_phrase_directory, f) for f in os.listdir(temp_phrase_directory) if
                                 os.path.isfile(os.path.join(temp_phrase_directory, f))]
            temp_files_time = time.time()
            final_merged_phrase_file = merge_all_phrase_files(temp_phrase_files)
            files_merged_time = time.time()
            create_inverted_phrase_index(final_merged_phrase_file, phrase_id_map, args.output_dir, 25)
            index_made_time = time.time()
            print("Time taken to write phrase temp files:", (temp_files_time - initial_time) * 1000, "milliseconds")
            print("Time taken to write merge phrase files:", (files_merged_time - initial_time) * 1000, "milliseconds")
            print("Entire time taken to create phrase index:", (index_made_time - initial_time) * 1000, "milliseconds")
            pass

        elif args.index_type == "positional":
            initial_time = time.time()
            process_documents(args.trec_files_directory_path, temp_directory)
            temp_files = [os.path.join(temp_directory, f) for f in os.listdir(temp_directory) if
                          os.path.isfile(os.path.join(temp_directory, f))]
            temp_files_time = time.time()
            final_merged_file = merge_all_files(temp_files)
            files_merged_time = time.time()
            create_inverted_positional_index(final_merged_file, term_id_map, args.output_dir)

            index_made_time = time.time()

            doc_id_map_file = os.path.join(args.output_dir, "doc_id_map.txt")
            # Write the dictionary to the file
            with open(doc_id_map_file, 'w') as f:
                for key, value in doc_id_map.items():
                    f.write(f"{key}: {value}\n")

            print("Time taken to write positional temp files:", (temp_files_time - initial_time) * 1000, "milliseconds")
            print("Time taken to write merge positional files:", (files_merged_time - initial_time) * 1000,
                  "milliseconds")
            print("Entire time taken to create positional index:", (index_made_time - initial_time) * 1000,
                  "milliseconds")
            pass

        elif args.index_type == "single":
            initial_time = time.time()
            process_documents(args.trec_files_directory_path, temp_directory)
            temp_files = [os.path.join(temp_directory, f) for f in os.listdir(temp_directory) if
                          os.path.isfile(os.path.join(temp_directory, f))]
            temp_files_time = time.time()
            final_merged_file = merge_all_files(temp_files)
            files_merged_time = time.time()
            create_single_term_index_without_stopwords(final_merged_file, term_id_map, args.output_dir)
            index_made_time = time.time()
            print("Time taken to write single temp files:", (temp_files_time - initial_time) * 1000, "milliseconds")
            print("Time taken to write merge single files:", (files_merged_time - initial_time) * 1000, "milliseconds")
            print("Entire time taken to create single index:", (index_made_time - initial_time) * 1000, "milliseconds")
            pass

        elif args.index_type == "stem":
            initial_time = time.time()
            process_documents(args.trec_files_directory_path, temp_directory)
            temp_files = [os.path.join(temp_directory, f) for f in os.listdir(temp_directory) if
                          os.path.isfile(os.path.join(temp_directory, f))]
            temp_files_time = time.time()
            final_merged_file = merge_all_files(temp_files)
            files_merged_time = time.time()
            create_stem_index(final_merged_file, term_id_map, args.output_dir)
            index_made_time = time.time()
            print("Time taken to write stem temp files:", (temp_files_time - initial_time) * 1000, "milliseconds")
            print("Time taken to write merge stem files:", (files_merged_time - initial_time) * 1000, "milliseconds")
            print("Entire time taken to create stem index:", (index_made_time - initial_time) * 1000, "milliseconds")
            pass

        elif args.index_type == "doc_term":
            print("creating doc_term index")
            create_doc_term_index_and_idf_dict("single_term_index_without_stopwords.jsonl",
                                               args.output_dir)
            pass

        shutil.rmtree(temp_directory)  # delete the temp files
        shutil.rmtree(temp_phrase_directory)  # delete the temp phrase files
        for file in intermediate_merged_files:  # delete the intermediate merged files
            os.remove(file)
        for file in intermediate_merged_phrase_files:  # delete the intermediate merged phrase files
            os.remove(file)
        pass

    elif args.command == "query_static":
        # Code to process the queries, rank documents, and write results to the results file
        # index_directory_path is Output folder
        if args.index_type == "single":
            index_path = os.path.join(args.index_directory_path, "single_term_index_without_stopwords.jsonl")

            if args.ranking_type == "regular":
                print("regular...")
                queries = list(read_queries_single(args.query_file_path, index_path))
                if args.retrieval_model == "cosine":
                    print("Single Term Index Cosine: ")

                    # Gather all unique terms from all queries
                    all_query_terms = set()
                    for query in queries:
                        all_query_terms.update(query['title'].keys())

                    # Create query vectors for each query
                    for query in queries:
                        query_vector = populate_query_vector(query, index_path)
                        query['vector'] = query_vector  # Store the computed vector in the query dictionary

                    # Create document vectors for all these terms
                    document_vectors = create_document_vectors(index_path)

                    print("computing rankings")
                    # Compute the rankings
                    rankings = compute_rankings(queries, document_vectors, args.index_directory_path)
                    print("writing rankings")
                    # After computing the rankings:
                    write_rankings_to_file(rankings, args.results_file)
                    pass

                elif args.retrieval_model == "bm25":
                    print("Single Term Index BM25: ")

                    pickle_file = os.path.join(args.index_directory_path, "data.pkl")
                    with open(pickle_file, "rb") as f:
                        data = pickle.load(f)
                        doc_lengths = data["doc_lengths"]
                        avg_doc_length = data["avg_doc_length"]

                    # print("Queries:")
                    # print(queries)
                    # print("––––––––––––––––––––")

                    results = compute_bm25_scores_for_all_queries_single(queries, index_path, doc_lengths,
                                                                         avg_doc_length)

                    doc_id_map_directory_path = args.index_directory_path

                    rank_and_write_top_scores(results, args.results_file, doc_id_map_directory_path)

                elif args.retrieval_model == "lm":
                    print("Single Term Index LM: ")

                    pickle_file = os.path.join(args.index_directory_path, "data.pkl")
                    with open(pickle_file, "rb") as f:
                        data = pickle.load(f)
                        doc_lengths = data["doc_lengths"]
                        avg_doc_length = data["avg_doc_length"]

                    results = args.results_file
                    doc_id_map_dir = args.index_directory_path

                    # Process all queries and rank documents
                    process_all_queries(queries, index_path, doc_lengths, avg_doc_length, results, doc_id_map_dir)
                    pass

            elif args.ranking_type == "prf":
                print("prf...")
                queries = list(read_queries_single(args.query_file_path, index_path))
                pickle_file = os.path.join(args.index_directory_path, "data.pkl")
                with open(pickle_file, "rb") as f:
                    data = pickle.load(f)
                    doc_lengths = data["doc_lengths"]
                    avg_doc_length = data["avg_doc_length"]

                results = compute_bm25_scores_for_all_queries_single(queries, index_path, doc_lengths, avg_doc_length)

                doc_id_map_directory_path = args.index_directory_path

                rank_and_write_top_scores(results, args.results_file, doc_id_map_directory_path)

                top_20_docs = retrieve_top_20_docs(args.results_file)

                # Full list of N_values = [1, 3, 5, 10, 15, 20]
                n_values = [1, 3, 5, 10, 15, 20]
                # Full list of M_values = [1, 2, 3, 5]
                m_values = [1,2, 3, 5]
                m_value = 5

                # Paths for the doc-term index and IDF dictionary
                doc_term_index_path = os.path.join(args.index_directory_path, "doc_term_index.json")
                idf_dict_path = os.path.join(args.index_directory_path, "idf_dict.json")

                # Load the doc-term index
                with open(doc_term_index_path, 'rt') as f:
                    doc_term_index = json.load(f)

                # Load the IDF dictionary
                with open(idf_dict_path, 'rt') as f:
                    idf_dict = json.load(f)

                # Call the function with the new parameters
                top_terms = calculate_top_terms_for_documents(top_20_docs, doc_term_index, idf_dict,
                                                              doc_id_map_directory_path, n_values, m_value)

                print("Expanding Queries")
                expanded_results = expand_and_rescore_queries(queries, top_terms, idf_dict, n_values, m_values,
                                                              index_path, doc_lengths, avg_doc_length)

                print("Ranking and writing final prf scores")
                rank_and_write_all_top_prf_scores(expanded_results, args.index_directory_path, doc_id_map_directory_path)
                pass

            elif args.ranking_type == "reduction":
                print("reduction...")
                pickle_file = os.path.join(args.index_directory_path, "data.pkl")
                with open(pickle_file, "rb") as f:
                    data = pickle.load(f)
                    doc_lengths = data["doc_lengths"]
                    avg_doc_length = data["avg_doc_length"]

                idf_dict_path = os.path.join(args.index_directory_path, "idf_dict.json")
                doc_id_map_directory_path = args.index_directory_path

                # Load the IDF dictionary
                with open(idf_dict_path, 'rt') as f:
                    idf_dict = json.load(f)
                processed_narrative_queries = list(process_narrative_queries(args.query_file_path, idf_dict, 20))

                results = compute_bm25_scores_for_all_queries_single(processed_narrative_queries, index_path, doc_lengths, avg_doc_length)

                rank_and_write_top_scores(results, args.results_file, doc_id_map_directory_path)
                pass
            elif args.ranking_type == "prf_reduction":
                print("prf_reduction...")
                pickle_file = os.path.join(args.index_directory_path, "data.pkl")
                with open(pickle_file, "rb") as f:
                    data = pickle.load(f)
                    doc_lengths = data["doc_lengths"]
                    avg_doc_length = data["avg_doc_length"]

                idf_dict_path = os.path.join(args.index_directory_path, "idf_dict.json")
                doc_id_map_directory_path = args.index_directory_path

                # Load the IDF dictionary
                with open(idf_dict_path, 'rt') as f:
                    idf_dict = json.load(f)
                processed_narrative_queries = list(process_narrative_queries(args.query_file_path, idf_dict, 35))

                results = compute_bm25_scores_for_all_queries_single(processed_narrative_queries, index_path,
                                                                     doc_lengths, avg_doc_length)

                rank_and_write_top_scores(results, args.results_file, doc_id_map_directory_path)
                print(f"results file: {args.results_file}")
                top_20_docs = retrieve_top_20_docs(args.results_file)

                # Full list of N_values = [1, 3, 5, 10, 15, 20]
                n_values = [1, 3, 5, 10, 15, 20]
                # Full list of M_values = [1, 2, 3, 5]
                m_values = [1, 2, 3, 5]
                m_value = 5

                # Paths for the doc-term index and IDF dictionary
                doc_term_index_path = os.path.join(args.index_directory_path, "doc_term_index.json")
                idf_dict_path = os.path.join(args.index_directory_path, "idf_dict.json")

                # Load the doc-term index
                with open(doc_term_index_path, 'rt') as f:
                    doc_term_index = json.load(f)

                # Load the IDF dictionary
                with open(idf_dict_path, 'rt') as f:
                    idf_dict = json.load(f)

                # Call the function with the new parameters
                top_terms = calculate_top_terms_for_documents(top_20_docs, doc_term_index, idf_dict,
                                                              doc_id_map_directory_path, n_values, m_value)

                print("Expanding Queries")
                expanded_results = expand_and_rescore_queries(processed_narrative_queries, top_terms, idf_dict,
                                                              n_values, m_values,index_path, doc_lengths,
                                                              avg_doc_length)

                print("Ranking and writing final prf-reduction scores")
                rank_and_write_all_top_prf_scores(expanded_results, args.index_directory_path,
                                                  doc_id_map_directory_path)
                pass

        elif args.index_type == "stem":
            # Load positional index
            index_path = os.path.join(args.index_directory_path, "stem_index.jsonl")
            queries = list(read_queries_stem(args.query_file_path, index_path))

            if args.retrieval_model == "cosine":
                print("Stem Index Cosine: ")

                # Gather all unique terms from all queries
                all_query_terms = set()
                for query in queries:
                    all_query_terms.update(query['title'].keys())

                # Create query vectors for each query
                for query in queries:
                    query_vector = populate_query_vector(query, index_path)
                    query['vector'] = query_vector  # Store the computed vector in the query dictionary

                # Create document vectors for all these terms
                document_vectors = create_document_vectors(index_path)

                # Compute the rankings
                rankings = compute_rankings(queries, document_vectors, args.index_directory_path)

                # After computing the rankings:
                write_rankings_to_file(rankings, args.results_file)
                pass
            elif args.retrieval_model == "bm25":
                print("Stem Index BM25: ")

                pickle_file = os.path.join(args.index_directory_path, "data.pkl")
                with open(pickle_file, "rb") as f:
                    data = pickle.load(f)
                    doc_lengths = data["doc_lengths"]
                    avg_doc_length = data["avg_doc_length"]

                results = compute_bm25_scores_for_all_queries(queries, index_path, doc_lengths, avg_doc_length)

                doc_id_map_directory_path = args.index_directory_path

                rank_and_write_top_scores(results, args.results_file, doc_id_map_directory_path)
                pass
            elif args.retrieval_model == "lm":
                print("Stem Index LM: ")

                pickle_file = os.path.join(args.index_directory_path, "data.pkl")
                with open(pickle_file, "rb") as f:
                    data = pickle.load(f)
                    doc_lengths = data["doc_lengths"]
                    avg_doc_length = data["avg_doc_length"]

                results = args.results_file
                doc_id_map_dir = args.index_directory_path

                # Process all queries and rank documents
                process_all_queries(queries, index_path, doc_lengths, avg_doc_length, results, doc_id_map_dir)
                pass
            pass
        pass

        pass

    elif args.command == "query_dynamic":
        # The path to your query file
        query_file = args.query_file_path

        # Path to phrase index
        phrase_index_path = os.path.join(args.index_directory_path, "inverted_phrase_index.jsonl")
        positional_index_path = os.path.join(args.index_directory_path, "inverted_positional_index.jsonl")

        # Load the doc lengths and avg doc lengths
        pickle_file = os.path.join(args.index_directory_path, "data.pkl")
        with open(pickle_file, "rb") as f:
            data = pickle.load(f)
            doc_lengths = data["doc_lengths"]
            avg_doc_length = data["avg_doc_length"]

        results_file = args.results_file
        doc_id_map_dir = args.index_directory_path

        results = main_query_processing_dynamic(query_file, phrase_index_path, positional_index_path, doc_lengths,
                                                avg_doc_length)
        rank_and_write_top_scores_dynamic(results, results_file, doc_id_map_dir)
        pass


if __name__ == '__main__':
    main()


end_time = time.time()
print("Time taken:", end_time - start_time)

