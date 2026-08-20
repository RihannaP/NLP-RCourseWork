
import pandas as pd
import nltk
import spacy
from pathlib import Path
import string
import math
from collections import Counter


nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2000000



def fk_level(text, d):
    """Returns the Flesch-Kincaid Grade Level of a text (higher grade is more difficult).
    Requires a dictionary of syllables per word.

    Args:
    text (str): The text to analyze.
    d (dict): A dictionary of syllables per word.

    Returns:
    float: The Flesch-Kincaid Grade Level of the text. (higher grade is more difficult)
    """
    # FK Grade Level = 0.39*(words/sentences) + 11.8*(syllables/words) - 15.59
    sentences = nltk.sent_tokenize(text)
    words = [w for w in nltk.word_tokenize(text)
             if not all(ch in string.punctuation for ch in w)]

    n_sentences = len(sentences)
    n_words = len(words)
    if n_sentences == 0 or n_words == 0:
        return 0.0

    n_syllables = sum(count_syl(w, d) for w in words)
    return 0.39 * (n_words / n_sentences) + 11.8 * (n_syllables / n_words) - 15.59


def count_syl(word, d):
    """Counts the number of syllables in a word given a dictionary of syllables per word.
    if the word is not in the dictionary, syllables are estimated by counting vowel clusters

    Args:
        word (str): The word to count syllables for.
        d (dict): A dictionary of syllables per word.

    Returns:
        int: The number of syllables in the word.
        
    """
    word = word.lower()
    if word in d:
        # in the CMU dict, every phoneme ending in a stress digit (0/1/2) is a vowel
        # sound = one syllable. Count them in the first pronunciation variant.
        return len([ph for ph in d[word][0] if ph[-1].isdigit()])

    # fallback for words not in the dictionary: count runs of consecutive vowels.
    vowels = "aeiouy"
    count = 0
    prev_was_vowel = False
    for ch in word:
        is_vowel = ch in vowels
        if is_vowel and not prev_was_vowel:
            count += 1
        prev_was_vowel = is_vowel
    return max(1, count)   # every word has at least one syllable


def read_novels(path=Path.cwd() / "texts" / "novels"):
    """Reads texts from a directory of .txt files and returns a DataFrame with the text, title,
    author, and year"""
    rows = []
    for file in Path(path).glob("*.txt"):
        # filenames are "Title-Author-Year.txt". The title can have hyphens because of some of them divided by that
        # so I am taking year and author from the END and re-join whatever is left as the title.
        parts = file.stem.split("-")
        year = int(parts[-1])
        author = parts[-2]
        title = "-".join(parts[:-2]).replace("_", " ")
        text = file.read_text(encoding="utf-8", errors="ignore")
        rows.append({"text": text, "title": title, "author": author, "year": year})

    df = pd.DataFrame(rows, columns=["text", "title", "author", "year"])
    # (a)(ii): sort by year and reset the index
    df = df.sort_values("year").reset_index(drop=True)
    return df
    


def parse(df, store_path=Path.cwd() / "pickles", out_name="parsed.pickle"):
    """Parses the text of a DataFrame using spaCy, stores the parsed docs as a column and writes 
    the resulting  DataFrame to a pickle file"""
    
    df = df.copy()
    # Run each text through spaCy.
    df["parsed"] = list(nlp.pipe(df["text"], disable=["ner"])) #disabling NER, makes parsing faster

    # (d)(ii): serialise the whole df (Doc objects included) to a pickle file.
    store_path.mkdir(parents=True, exist_ok=True)
    df.to_pickle(store_path / out_name)

    return df   # (d)(iii)


def nltk_ttr(text):
    """Calculates the type-token ratio of a text. Text is tokenized using nltk.word_tokenize."""
    # lower-case first so types are counted case-insensitively ("The" == "the").
    tokens = nltk.word_tokenize(text.lower())
    # Drop tokens that are purely punctuation (e.g. "," ";" "--") so they aren't counted.
    tokens = [t for t in tokens if not all(ch in string.punctuation for ch in t)]
    if not tokens:
        return 0.0
    # types = unique tokens, tokens = all tokens
    return len(set(tokens)) / len(tokens)


def get_ttrs(df):
    """helper function to add ttr to a dataframe"""
    results = {}
    for i, row in df.iterrows():
        results[row["title"]] = nltk_ttr(row["text"])
    return results


def get_fks(df):
    """helper function to add fk scores to a dataframe"""
    results = {}
    cmudict = nltk.corpus.cmudict.dict()
    for i, row in df.iterrows():
        results[row["title"]] = round(fk_level(row["text"], cmudict), 4)
    return results


#.. add functions for part (e) here

def object_counts(doc):
    #counter of syntactic direct objects (dep_ == 'dobj') in a parsed Doc, counted by lemma
    objects = Counter()
    for token in doc:
        # is_alpha skips markup/OCR artifacts like '_' so only real word objects count
        if token.dep_ == "dobj" and token.is_alpha:
            objects[token.lemma_.lower()] += 1
    return objects


def object_dobjs(df, top_n=10):
    #(e)(i): prints each novel's title and its ten most common direct objects
    for _, row in df.iterrows():
        print(row["title"])
        print(object_counts(row["parsed"]).most_common(top_n))


def _verb_object_pairs(doc):
    """every (verb_lemma, object_text) pair joined by a direct-object relation.
    object is kept as surface text (lower-cased) so 'him'/'her' match directly
    (spaCy would otherwise lemmatise 'him' -> 'he')."""
    pairs = []
    for token in doc:
        if (token.dep_ == "dobj" and token.head.pos_ == "VERB"
                and token.is_alpha and token.head.is_alpha):
            pairs.append((token.head.lemma_.lower(), token.text.lower()))
    return pairs


def pmi_verbs_for_object(doc, target_object, top_n=10, min_count=2):
    """(e)(ii/iii): verbs most associated with target_object, ranked by PMI.
    PMI(verb, obj) = log2( P(verb, obj) / (P(verb) * P(obj)) ),
    estimated over all verb--direct-object pairs in the text."""
    pairs = _verb_object_pairs(doc)
    total = len(pairs)
    if total == 0:
        return []

    verb_counts = Counter(v for v, o in pairs)
    obj_counts = Counter(o for v, o in pairs)
    pair_counts = Counter(pairs)

    scores = {}
    for (verb, obj), joint in pair_counts.items():
        if obj != target_object or joint < min_count:
            continue
        # log2( (joint/total) / ((count(verb)/total) * (count(obj)/total)) )
        scores[verb] = math.log2((joint * total) / (verb_counts[verb] * obj_counts[obj]))

    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_n]


def subject_verbs_for_object(df, target_object, top_n=10, min_count=2):
    #prints each novel's title and its top verbs (by PMI) for a given object
    for _, row in df.iterrows():
        print(row["title"])
        print(pmi_verbs_for_object(row["parsed"], target_object, top_n, min_count))




if __name__ == "__main__":
    
    nltk.download("cmudict")
    nltk.download("punkt")
    nltk.download("punkt_tab")
    path = Path.cwd() / "texts" / "novels"
    print(path)
    df = read_novels(path)
    # print(df[["title", "author", "year"]].head())
    print(df[["title", "author", "year"]])
    print("shape:", df.shape)
    

    print("\n--- Type-Token Ratio ---")
    for title, ttr in get_ttrs(df).items():
        print(f"{ttr:.4f}  {title}")

    print("\n--- Flesch-Kincaid Grade Level ---")
    for title, score in get_fks(df).items():
        print(f"{score:6.2f}  {title}")
    
    parse(df)
    df = pd.read_pickle(Path.cwd() / "pickles" /"parsed.pickle")
    print("\n--- Parsed dataframe ---")
    print("columns:", df.columns.tolist())
    print(df[["title", "year"]].head())
    #quick check with first 14 tikends of first novel
    doc = df.loc[0, "parsed"] 
    print([(t.text, t.pos_, t.dep_) for t in doc[:15]])
    
    # call functions for part (e) here.
    # (e)(i) ten most common direct objects per novel
    print("\n--- (e)(i) Ten most common direct objects ---")
    object_dobjs(df)

    # (e)(ii) verbs most associated with the object 'him' (by PMI)
    print("\n--- (e)(ii) Verbs with object 'him' (by PMI) ---")
    subject_verbs_for_object(df, "him")

    # (e)(iii) verbs most associated with the object 'her' (by PMI)
    print("\n--- (e)(iii) Verbs with object 'her' (by PMI) ---")
    subject_verbs_for_object(df, "her")


#I r