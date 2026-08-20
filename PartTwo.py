import pandas as pd
from pathlib import Path


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


if __name__ == "__main__":
    df = load_and_clean()
    print("shape:", df.shape)
    print(df["party"].value_counts())