import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import f1_score, classification_report
import spacy
_nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

def load_and_clean(path=Path.cwd() / "texts" / "hansard10000.csv"):
    #reads the Hansard CSV and applies the Part 2(a) cleaning steps
    df = pd.read_csv(path)

    # (a)(i) rename 'Labour (Co-op)' to 'Labour' so both count as one party
    df["party"] = df["party"].replace("Labour (Co-op)", "Labour")

    # (a)(ii) drop the 'Speaker' rows, then keep only the four most common parties
    df = df[df["party"] != "Speaker"]
    top4 = df["party"].value_counts().nlargest(4).index
    df = df[df["party"].isin(top4)]

    # (a)(iii) keep only rows whose speech_class is 'Speech'
    df = df[df["speech_class"] == "Speech"]

    # (a)(iv) keep only speeches at least 1000 characters long
    df = df[df["speech"].str.len() >= 1000]

    return df


def vectorise_and_classify(df, vectorizer=None):
    """Vectorises speeches with TF-IDF, splits train/test, trains two classifiers,
    and prints macro-F1 + classification report for each on the test set."""
    if vectorizer is None:
        # default params, but drop English stopwords and cap at 3000 features
        vectorizer = TfidfVectorizer(stop_words="english", max_features=3000)

    X = vectorizer.fit_transform(df["speech"])
    y = df["party"]

    # stratified split preserves the party proportions in train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=26
    )

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "ComplementNB": ComplementNB(),
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        print(f"\n===== {name} =====")
        #added , zero_division=0 to remove warnings
        print("macro-F1:", f1_score(y_test, preds, average="macro", zero_division=0))
        print(classification_report(y_test, preds, zero_division=0))


def custom_tokenizer(text):
    """Custom tokenizer for 2(d).
    Runs spaCy over the speech and returns lower-cased lemmas, keeping only
    content words (nouns, verbs, adjectives, proper nouns) and dropping stopwords.
    Restricting to content-word POS tags removes function words, punctuation and
    numbers, and lemmatising collapses inflections (tax/taxes/taxing -> tax) - so
    each feature captures a whole word family, concentrating signal into fewer,
    more informative features."""
    doc = _nlp(text)
    return [
        t.lemma_.lower()
        for t in doc
        if t.pos_ in {"NOUN", "VERB", "ADJ", "PROPN"} and not t.is_stop
    ]


def classify_best(df, vectorizer, label):
    """Train both classifiers, print the classification report for the better one (by macro-F1)."""
    X = vectorizer.fit_transform(df["speech"])
    y = df["party"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=26
    )
    print(f"\n########## {label} — features used: {len(vectorizer.get_feature_names_out())} ##########")

    best_name, best_f1, best_report = None, -1, None
    for name, model in {"LogisticRegression": LogisticRegression(max_iter=1000),
                        "ComplementNB": ComplementNB()}.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        macro = f1_score(y_test, preds, average="macro", zero_division=0)
        print(f"{name}: macro-F1 = {macro:.4f}")
        if macro > best_f1:
            best_name, best_f1 = name, macro
            best_report = classification_report(y_test, preds, zero_division=0)

    print(f"\n>>> Best: {best_name} (macro-F1 = {best_f1:.4f})")
    print(best_report)



if __name__ == "__main__":
    print("\n########## 2(a): Load and Clean Hansard CSV ##########")
    df = load_and_clean()
    print("shape:", df.shape)
    # print(df["party"].value_counts())
    print("\n########## 2(b): TF-IDF, unigrams ##########")
    #2(b) discussion 
    # Both models reach ~0.71 accuracy but low macro-F1 (LogReg ~0.43, ComplementNB
    # ~0.46). Heavy class imbalance explains the gap: Conservative dominates, so
    # accuracy stays high while macro-F1 (equal weight per party) exposes weak results on small classes. Liberal Democrat (15 test samples) is never predicted -> F1 0.00 (hence zero_division=0). ComplementNB spreads predictions more evenly and edges ahead, as expected for imbalanced text. Macro-F1 is the more honest metric.
    vectorise_and_classify(df)

    print("\n########## 2(c): TF-IDF, unigrams + bigrams + trigrams ##########")
    #2(c) discussion
    # Adding bigrams/trigrams (still 3000 features) lifts macro-F1 for both models
    # (LogReg 0.43->0.46, ComplementNB 0.46->0.49), driven almost entirely by better
    # SNP recall - its party-specific phrases only appear as n-grams. Majority classes
    # are unchanged and Liberal Democrat stays at 0.00 (too few samples).
    ngram_vec = TfidfVectorizer(
        stop_words="english",
        max_features=3000,
        ngram_range=(1, 3),   # unigrams, bigrams and trigrams
    )
    vectorise_and_classify(df, ngram_vec)

    print("\n########## 2(d): custom spaCy tokenizer ##########")
    custom_vec = TfidfVectorizer(
        tokenizer=custom_tokenizer,
        token_pattern=None,   # silence the warning when a custom tokenizer is given
        max_features=2000
    )
    classify_best(df, custom_vec, "2(d) custom tokenizer")