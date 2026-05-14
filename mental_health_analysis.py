"""
Task 1.1 – Mental Health in Technology Related Jobs
Course: DLBDSMLUSL01 – Machine Learning: Unsupervised Learning and Feature Engineering
IU International University of Applied Sciences

Pipeline:
  1. Data Loading & Exploration
  2. Data Cleaning & Preprocessing
  3. Feature Engineering
  4. Dimensionality Reduction (PCA + t-SNE)
  5. Clustering (K-Means + Silhouette Analysis)
  6. Cluster Interpretation & Visualization
  7. Per-cluster Descriptive Statistics

Dataset: OSMI Mental Health in Tech Survey 2016
  https://www.kaggle.com/osmi/mental-health-in-tech-2016


Implementation references (see report for full citations):

  - Géron, A. (2022). Hands on Machine Learning with Scikit Learn,
    Keras, and TensorFlow (3rd ed.). O'Reilly Media
      Chapter.2  end to end pipeline structure, data cleaning, imputation
      Chapter.8  PCA with cumulative variance threshold
      Chapter.9  K-Means + elbow + silhouette analysis
  - McKinney, W. (2017). Python for Data Analysis (2nd ed.). O'Reilly.
      Pandas patterns (Chapter 5-7).
  - Scikit-learn user guide:
      https://scikit-learn.org/stable/user_guide.html
  - Scikit-learn silhouette analysis example (this is the basis for Figure 6):
      https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
  - Scikit-learn manifold guide (t-SNE recommendations):
      https://scikit-learn.org/stable/modules/manifold.html#t-sne
  - Kobak, D., & Berens, P. (2019). The art of using t-SNE for
    single-cell transcriptomics. Nature Communications, 10, 5416.
      (PCA initialization recommendation)
  - PEP 8 (https://peps.python.org/pep-0008/) for code structure.
 
Algorithmic references (foundational papers for techniques used):

  - Lloyd, S. P. (1982). IEEE Trans. Inf. Theory, 28(2), 129-137.
      (K-Means algorithm.)
  - Arthur & Vassilvitskii (2007). SODA, 1027-1035.
      (k-means++ initialization, used by default in scikit-learn.)
  - Rousseeuw, P. J. (1987). J. Comp. Appl. Math., 20, 53-65.
      (Silhouette coefficient.)
  - Jolliffe, I. T., & Cadima, J. (2016). Phil. Trans. R. Soc. A.
      (PCA review.)
  - Van der Maaten, L., & Hinton, G. (2008). JMLR, 9, 2579-2605.
      (t-SNE original paper.)


------------------ Section by section attribution -----------------------

------------------------- Imports block ---------------------------------

python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
...

This is the conventional scikit-learn import idiom. Pattern is referenced in:
* Géron (2022), Chapter 2 "End to End Machine Learning Project" uses this exact import block.
* Scikit learn official user guide, "Getting started" section: https://scikit-learn.org/stable/getting_started.html
* The aliases np, pd, plt, sns are PEP-8-style conventions documented in numpy, pandas, matplotlib, and seaborn docs respectively.

-------- Configuration constants (RANDOM_STATE, PALETTE, etc.) -------

python
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
plt.rcParams.update({"figure.dpi": 120, "font.size": 10})

* Using a global RANDOM_STATE for reproducibility: standard scikit learn convention, taught in Géron (2022), Chapter 2 and the scikit learn user guide on randomness: https://scikit-learn.org/stable/common_pitfalls.html#controlling randomness
* The value 42 is a long running programming joke from Douglas Adams but also is the de facto convention in tutorials and other user experiences referenced online.
* plt.rcParams.update(...) for global plot styling: matplotlib documentation on customization: https://matplotlib.org/stable/users/explain/customizing.html

--------------------- Data loading and initial exploration -----------

python
df = pd.read_csv(filepath)
df.shape, df.duplicated().sum(), df.dtypes.value_counts(), df.isnull().sum()
This exploration pattern is from:

* McKinney (2017), "Python for Data Analysis" (2nd ed.), Chapter 5 pandas basics
* Géron (2022), Chapter 2, "Take a Quick Look at the Data Structure" section
* pandas user guide: https://pandas.pydata.org/docs/user_guide/10min.html


-------------------- Age cleaning with outlier filtering ------------

python
s = pd.to_numeric(series, errors="coerce")
median_age = s[(s >= 18) & (s <= 75)].median()
s = s.where((s >= 18) & (s <= 75), other=np.nan)
return s.fillna(median_age)

* pd.to_numeric(..., errors="coerce"): pandas standard idiom for safe numeric conversion. Documented in pandas API reference.
* Median imputation is found in Géron (2022), Chapter 2, "Data Cleaning" section, explicitly uses median imputation as a baseline.
* Outlier filtering using range bounds: general data cleaning pattern from VanderPlas (2016), "Python Data Science Handbook", Chapter 3.

-------------------- Gender standardization -------------------------

python
def _map(v):
    if v.startswith("male") or v in (...): return "Male"

* The use of a mapping function applied with .map() is standard pandas: documented at https://pandas.pydata.org/docs/reference/api/pandas.Series.map.html
* The specific OSMI gender harmonization (mapping approx. ~50 free text entries to Male/Female/Other) is a well documented preprocessing step seen on public Kaggle notebooks for this exact dataset. Search "OSMI mental health Kaggle EDA" multiple public notebooks (notably by Kaggle users kairosart, gnathawat, and fabiendaniel) apply this same harmonization. The mapping logic is essentially community-standard for this dataset.

--------------------- Label encoding loop ---------------------------

python
for col in categorical_cols:
    d[col] = LabelEncoder().fit_transform(d[col].astype(str))

* Scikit-learn LabelEncoder documentation: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
* The astype(str) trick to handle NaN before encoding is widely used in Stack Overflow answers (a common gotcha).
* Géron (2022), Chapter 2, covers encoding approaches.


---------------------- SimpleImputer ---------------------------------

python
SimpleImputer(strategy="median").fit_transform(...)
SimpleImputer(strategy="most_frequent").fit_transform(...)

* Scikit-learn imputer documentation: https://scikit-learn.org/stable/modules/impute.html
* The median for numeric / mode for categorical convention: Géron (2022), Chapter 2.

---------------------- StandardScaler --------------------------------

python
X = StandardScaler().fit_transform(d)

* Scikit-learn preprocessing documentation: https://scikit-learn.org/stable/modules/preprocessing.html
* The necessity of scaling before PCA and K-Means: covered in IU DLBDSMLUSL01 course book Unit 2 and Unit 3.

---------------- PCA + variance threshold selection ------------------

python
pca_full = PCA(random_state=RANDOM_STATE).fit(X)
cumvar = np.cumsum(pca_full.explained_variance_ratio_)
n_comp = int(np.searchsorted(cumvar, 0.90) + 1)

The above is textbook pattern use reference of:

* Géron (2022), Chapter 8, "Choosing the Right Number of Dimensions" section, uses literally np.cumsum(pca.explained_variance_ratio_) and np.argmax(cumsum >= 0.95) + 1 — the same approach with a slightly different criterion.
* Scikit learn PCA tutorial: https://scikit-learn.org/stable/modules/decomposition.html#pca
* The 90% / 95% variance threshold is a standard heuristic discussed in Jolliffe & Cadima (2016).


------------------- t-SNE on PCA-reduced data ------------------------

python
tsne = TSNE(n_components=2, perplexity=40, max_iter=1000, init="pca", ...)
X_tsne = tsne.fit_transform(X_pca)

* Scikit-learn manifold guide: https://scikit-learn.org/stable/modules/manifold.html#t-sne
* The recommendation to apply t-SNE after PCA dimensionality reduction is given in the scikit learn user guide itself and in the original Van der Maaten & Hinton (2008) paper.
* init="pca" for reproducibility is a recommendation from Kobak & Berens (2019), "The art of using t-SNE for single-cell transcriptomics", Nature Communications widely cited for t-SNE best practices.

-------- K-Means optimal k selection (elbow + silhouette) -------------

python
for k in k_range:
    km = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE)
    labels = km.fit_predict(X_pca)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_pca, labels))

* The elbow method pattern: taught in Géron (2022), Chapter 9, "Finding the optimal number of clusters" section, with essentially this exact loop.
* Silhouette analysis: Rousseeuw (1987) is the original paper; the scikit-learn implementation is documented at https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html The example page contains a very similar loop allowing adaptation.
* n_init=20 (or 30) to mitigate local minima: scikit-learn KMeans documentation explicitly recommends this mitigation.

---------- Silhouette diagram (the horizontal bar layout)----------

python
y_lower = 10
for c in range(k):
    c_vals = np.sort(sil_vals[labels == c])
    ...
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, c_vals, ...)

* This is directly adapted from the scikit-learn official example: https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
* This particular visualization pattern is the most recognizable and directly borrowed structure in the script and it is essentially the canonical silhouette diagram code. 


----------------- Heatmap of cluster means -----------------------

python
means_norm = means.apply(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9))
sns.heatmap(means_norm, cmap="YlOrRd", ...)

* Seaborn heatmap documentation and examples: https://seaborn.pydata.org/generated/seaborn.heatmap.html
* Row wise min/max normalization for cluster comparison: general data-viz pattern, I cannot citate one single canonical source it has been random and experimental.
* Main pipeline structure (modular functions + main())

* The overall architecture separating load_data(), preprocess(), engineer_features(), reduce_dimensions(), etc., with a main() driver guarded by if __name__ == "__main__": — is standard Python project structure referenced in:

- PEP 8 style guide: https://peps.python.org/pep-0008/
- "The Hitchhiker's Guide to Python": https://docs.python-guide.org/writing/structure/
- Géron (2022) uses similar modular structure in his Chapter 2 end-to-end example.

"""

# ------------------------- IMPORTS --------------------------
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.impute import SimpleImputer

# ---------------------- CONFIGURATION -----------------------
RANDOM_STATE = 42
DATA_FILE    = "mental-heath-in-tech-2016_20161114.csv"
OUTPUT_DIR   = "."

np.random.seed(RANDOM_STATE)
plt.rcParams.update({"figure.dpi": 120, "font.size": 10})
PALETTE = sns.color_palette("tab10")

# Actual long-form column names in the 2016 OSMI dataset
AGE_COL    = "What is your age?"
GENDER_COL = "What is your gender?"

# Friendly short labels for plotting (long question -> short label)
SHORT_LABELS = {
    "Are you self-employed?": "self_employed",
    "How many employees does your company or organization have?": "company_size",
    "Is your employer primarily a tech company/organization?": "tech_company",
    "Does your employer provide mental health benefits as part of healthcare coverage?": "mh_benefits",
    "Do you know the options for mental health care available under your employer-provided coverage?": "knows_care_options",
    "Has your employer ever formally discussed mental health (for example, as part of a wellness campaign or other official communication)?": "wellness_program",
    "Does your employer offer resources to learn more about mental health concerns and options for seeking help?": "seek_help_resources",
    "Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources provided by your employer?": "anonymity",
    "If a mental health issue prompted you to request a medical leave from work, asking for that leave would be:": "leave_difficulty",
    "Do you think that discussing a mental health disorder with your employer would have negative consequences?": "mh_disclosure_consequence",
    "Do you think that discussing a physical health issue with your employer would have negative consequences?": "phys_disclosure_consequence",
    "Would you feel comfortable discussing a mental health disorder with your coworkers?": "comfort_coworkers",
    "Would you feel comfortable discussing a mental health disorder with your direct supervisor(s)?": "comfort_supervisor",
    "Do you feel that your employer takes mental health as seriously as physical health?": "mh_vs_phys",
    "Have you heard of or observed negative consequences for co-workers who have been open about mental health issues in your workplace?": "obs_consequences",
    "Do you have previous employers?": "had_previous_emp",
    "Would you be willing to bring up a physical health issue with a potential employer in an interview?": "phys_interview",
    "Would you bring up a mental health issue with a potential employer in an interview?": "mh_interview",
    "Do you feel that being identified as a person with a mental health issue would hurt your career?": "career_hurt",
    "Do you think that team members/co-workers would view you more negatively if they knew you suffered from a mental health issue?": "negative_view",
    "How willing would you be to share with friends and family that you have a mental illness?": "share_family",
    "Have you observed or experienced an unsupportive or badly handled response to a mental health issue in your current or previous workplace?": "bad_response_obs",
    "Do you have a family history of mental illness?": "family_history",
    "Have you had a mental health disorder in the past?": "had_disorder_past",
    "Do you currently have a mental health disorder?": "current_disorder",
    "Have you been diagnosed with a mental health condition by a medical professional?": "diagnosed",
    "Have you ever sought treatment for a mental health issue from a mental health professional?": "sought_treatment",
    "If you have a mental health issue, do you feel that it interferes with your work when being treated effectively?": "interferes_treated",
    "If you have a mental health issue, do you feel that it interferes with your work when NOT being treated effectively?": "interferes_untreated",
    "Which of the following best describes your work position?": "work_position",
    "Do you work remotely?": "remote_work",
}


# ----------------------------------------------------------------
# 1. DATA LOADING & EXPLORATION
# ----------------------------------------------------------------
def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    print(f"\n{'='*60}\n1. INITIAL DATA EXPLORATION\n{'='*60}")
    print(f"  Shape          : {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"  Duplicate rows : {df.duplicated().sum()}")
    print(f"\n  Dtypes:\n{df.dtypes.value_counts().to_string()}")
    missing = df.isnull().sum()
    print(f"\n  Top-10 missing columns:")
    for col, n in missing.sort_values(ascending=False).head(10).items():
        print(f"    {n:4d}  {col[:80]}")
    return df


# ----------------------------------------------------------------
# 2. CLEANING & PREPROCESSING
# ----------------------------------------------------------------
def clean_age(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    median_age = s[(s >= 18) & (s <= 75)].median()
    s = s.where((s >= 18) & (s <= 75), other=np.nan)
    return s.fillna(median_age)


def standardize_gender(series: pd.Series) -> pd.Series:
    s = series.fillna("").astype(str).str.strip().str.lower()

    def _map(v):
        if not isinstance(v, str) or v in ("nan", "", "none"):
            return "Other"
        if v.startswith("male") or v in ("m", "man", "cis male", "cis man",
                                          "male (cis)", "dude", "mail",
                                          "sex is male", "m|"):
            return "Male"
        if v.startswith("female") or v.startswith("woman") or v in (
                "f", "fem", "cis female", "cis woman", "female (cis)",
                "female assigned at birth"):
            return "Female"
        return "Other"

    return s.map(_map)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n{'='*60}\n2. DATA CLEANING & PREPROCESSING\n{'='*60}")
    d = df.copy()

    if AGE_COL in d.columns:
        d[AGE_COL] = clean_age(d[AGE_COL])
        print(f"  Age – cleaned; range [{d[AGE_COL].min():.0f}, {d[AGE_COL].max():.0f}]")

    if GENDER_COL in d.columns:
        d[GENDER_COL] = standardize_gender(d[GENDER_COL])
        print(f"\n  Gender distribution:")
        for v, n in d[GENDER_COL].value_counts().items():
            print(f"    {v:8s} {n}")

    # Drop free-text / geographic columns
    drop_cols_free_text = [
        "Why or why not?", "Why or why not?.1",
        "If yes, what condition(s) have you been diagnosed with?",
        "If maybe, what condition(s) do you believe you have?",
        "If so, what condition(s) were you diagnosed with?",
        "What country do you live in?",
        "What US state or territory do you live in?",
        "What country do you work in?",
        "What US state or territory do you work in?",
    ]
    n0 = d.shape[1]
    d = d.drop(columns=[c for c in drop_cols_free_text if c in d.columns])
    print(f"\n  Dropped {n0 - d.shape[1]} free-text / geographic columns.")

    # Drop columns with >60% missing
    threshold = 0.60
    drop_high_miss = d.columns[d.isnull().mean() > threshold].tolist()
    d = d.drop(columns=drop_high_miss)
    print(f"  Dropped {len(drop_high_miss)} columns with >{threshold*100:.0f}% missing.")

    # Rename to short labels
    rename_map = {k: v for k, v in SHORT_LABELS.items() if k in d.columns}
    rename_map[AGE_COL] = "age"
    rename_map[GENDER_COL] = "gender"
    d = d.rename(columns=rename_map)

    print(f"\n  Remaining shape: {d.shape[0]} rows × {d.shape[1]} columns")
    return d


# ----------------------------------------------------------------
# 3. FEATURE ENGINEERING
# ----------------------------------------------------------------
def engineer_features(df: pd.DataFrame):
    print(f"\n{'='*60}\n3. FEATURE ENGINEERING\n{'='*60}")
    d = df.copy()
    numeric_cols     = d.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = d.select_dtypes(exclude=[np.number]).columns.tolist()
    print(f"  Numeric cols    : {len(numeric_cols)}  → {numeric_cols}")
    print(f"  Categorical cols: {len(categorical_cols)}")

    if numeric_cols:
        d[numeric_cols] = SimpleImputer(strategy="median").fit_transform(d[numeric_cols])
    for col in categorical_cols:
        d[col] = LabelEncoder().fit_transform(d[col].astype(str))
    if categorical_cols:
        d[categorical_cols] = SimpleImputer(strategy="most_frequent").fit_transform(d[categorical_cols])

    X = StandardScaler().fit_transform(d)
    print(f"\n  Final feature matrix: {X.shape}")
    return X, d


# ----------------------------------------------------------------
# 4. DIMENSIONALITY REDUCTION
# ----------------------------------------------------------------
def reduce_dimensions(X: np.ndarray):
    print(f"\n{'='*60}\n4. DIMENSIONALITY REDUCTION\n{'='*60}")

    pca_full = PCA(random_state=RANDOM_STATE).fit(X)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n_comp = int(np.searchsorted(cumvar, 0.90) + 1)
    print(f"  PCA: {n_comp} components explain ≥ 90% of variance")

    X_pca  = PCA(n_components=n_comp, random_state=RANDOM_STATE).fit_transform(X)
    X_pca2 = PCA(n_components=2,      random_state=RANDOM_STATE).fit_transform(X)

    print("  Running t-SNE …")
    try:
        tsne = TSNE(n_components=2, perplexity=40, max_iter=1000,
                    learning_rate="auto", init="pca", random_state=RANDOM_STATE)
    except TypeError:
        tsne = TSNE(n_components=2, perplexity=40, n_iter=1000,
                    learning_rate="auto", init="pca", random_state=RANDOM_STATE)
    X_tsne = tsne.fit_transform(X_pca)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    n_show = min(20, len(pca_full.explained_variance_ratio_))
    axes[0].bar(range(1, n_show + 1), pca_full.explained_variance_ratio_[:n_show] * 100,
                color=PALETTE[0], alpha=0.8, label="Individual")
    axes[0].plot(range(1, n_show + 1), cumvar[:n_show] * 100, "o-",
                 color=PALETTE[1], lw=2, label="Cumulative")
    axes[0].axhline(90, ls="--", color="grey", lw=1)
    axes[0].set_xlabel("Principal Component")
    axes[0].set_ylabel("Explained Variance (%)")
    axes[0].set_title("PCA – Scree Plot"); axes[0].legend()

    axes[1].scatter(X_pca2[:, 0], X_pca2[:, 1], s=8, alpha=0.4, color=PALETTE[0])
    axes[1].set_title("2-D PCA Projection (all data)")
    axes[1].set_xlabel("PC 1"); axes[1].set_ylabel("PC 2")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/01_pca_scree.png", bbox_inches="tight"); plt.close()
    print("  Saved → 01_pca_scree.png")

    return X_pca, X_pca2, X_tsne


# ----------------------------------------------------------------
# 5. K-MEANS CLUSTERING
# ----------------------------------------------------------------
def find_optimal_k(X_pca, k_range=range(2, 10)):
    print(f"\n{'='*60}\n5. K-MEANS – OPTIMAL k SEARCH\n{'='*60}")
    inertias, silhouettes = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE)
        labels = km.fit_predict(X_pca)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_pca, labels))
        print(f"  k={k:2d}  inertia={km.inertia_:10,.0f}  silhouette={silhouettes[-1]:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(list(k_range), inertias, "o-", color=PALETTE[0], lw=2)
    axes[0].set_xlabel("k"); axes[0].set_ylabel("Inertia (within-cluster SSE)")
    axes[0].set_title("Elbow Method"); axes[0].set_xticks(list(k_range))
    axes[1].plot(list(k_range), silhouettes, "o-", color=PALETTE[1], lw=2)
    axes[1].set_xlabel("k"); axes[1].set_ylabel("Avg Silhouette Score")
    axes[1].set_title("Silhouette Analysis"); axes[1].set_xticks(list(k_range))
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/02_kmeans_elbow_silhouette.png", bbox_inches="tight"); plt.close()
    print("  Saved → 02_kmeans_elbow_silhouette.png")

    best_k = list(k_range)[int(np.argmax(silhouettes))]
    print(f"\n  ★ Optimal k (highest silhouette): {best_k}")
    return best_k


def fit_final_kmeans(X_pca, k):
    km = KMeans(n_clusters=k, n_init=30, random_state=RANDOM_STATE)
    labels = km.fit_predict(X_pca)
    print(f"\n  Final K-Means (k={k}): silhouette={silhouette_score(X_pca, labels):.4f}")
    return labels


# ----------------------------------------------------------------
# 6. VISUALISATIONS
# ----------------------------------------------------------------
def plot_clusters_2d(X_2d, labels, title, filename, k):
    fig, ax = plt.subplots(figsize=(9, 6))
    for c in range(k):
        mask = labels == c
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], s=10, alpha=0.55,
                   color=PALETTE[c], label=f"Cluster {c+1}")
    ax.set_title(title); ax.legend(markerscale=2.5, loc="best")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{filename}", bbox_inches="tight"); plt.close()
    print(f"  Saved → {filename}")


def plot_silhouette_diagram(X_pca, labels, k):
    sil_vals = silhouette_samples(X_pca, labels)
    fig, ax = plt.subplots(figsize=(9, 5))
    y_lower = 10
    for c in range(k):
        c_vals = np.sort(sil_vals[labels == c])
        y_upper = y_lower + c_vals.shape[0]
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, c_vals,
                         facecolor=PALETTE[c], alpha=0.75, label=f"Cluster {c+1}")
        y_lower = y_upper + 10
    avg = sil_vals.mean()
    ax.axvline(x=avg, color="red", linestyle="--", lw=1.5, label=f"Avg = {avg:.3f}")
    ax.set_xlabel("Silhouette coefficient"); ax.set_ylabel("Cluster")
    ax.set_title("Silhouette Plot by Cluster"); ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/05_silhouette_diagram.png", bbox_inches="tight"); plt.close()
    print("  Saved → 05_silhouette_diagram.png")


def plot_cluster_profiles(df_encoded, labels):
    df_plot = df_encoded.copy()
    df_plot["Cluster"] = labels + 1
    variances = df_plot.drop(columns=["Cluster"]).var().sort_values(ascending=False)
    top_cols = variances.head(12).index.tolist()

    fig, axes = plt.subplots(3, 4, figsize=(16, 11))
    axes = axes.flatten()
    for idx, col in enumerate(top_cols):
        grp = df_plot.groupby("Cluster")[col].mean()
        grp.plot(kind="bar", ax=axes[idx], color=PALETTE[:len(grp)],
                 edgecolor="black", linewidth=0.5)
        axes[idx].set_title(col, fontsize=10)
        axes[idx].set_xlabel("Cluster"); axes[idx].set_ylabel("Mean (encoded)")
        axes[idx].tick_params(axis="x", rotation=0)
    plt.suptitle("Cluster Profiles – Top-12 Highest-Variance Features", y=1.00, fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/06_cluster_profiles.png", bbox_inches="tight"); plt.close()
    print("  Saved → 06_cluster_profiles.png")


def plot_heatmap(df_encoded, labels):
    df_h = df_encoded.copy()
    df_h["Cluster"] = labels + 1
    means = df_h.groupby("Cluster").mean()
    means_norm = means.apply(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9))

    plt.figure(figsize=(max(14, len(means.columns) * 0.45), 5))
    sns.heatmap(means_norm, cmap="YlOrRd", linewidths=0.4,
                yticklabels=[f"Cluster {c}" for c in means.index],
                cbar_kws={"label": "Normalised mean"})
    plt.title("Cluster × Feature Heatmap (row-normalised means)")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/07_cluster_heatmap.png", bbox_inches="tight"); plt.close()
    print("  Saved → 07_cluster_heatmap.png")


# ----------------------------------------------------------------
# 7. DESCRIPTIVE STATS PER CLUSTER
# ----------------------------------------------------------------
def cluster_statistics(df_encoded, labels, k):
    print(f"\n{'='*60}\n7. CLUSTER DESCRIPTIVE STATISTICS\n{'='*60}")
    df_s = df_encoded.copy()
    df_s["Cluster"] = labels + 1
    summary = df_s.groupby("Cluster").mean().round(2)
    sizes   = df_s["Cluster"].value_counts().sort_index()

    print("\n  Cluster sizes:")
    for c, n in sizes.items():
        print(f"    Cluster {c}: {n:4d}  ({100*n/len(df_s):.1f}%)")

    print("\n  Cluster means (all features):")
    print(summary.T.to_string())

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar([f"Cluster {c}" for c in sizes.index], sizes.values,
                  color=[PALETTE[c - 1] for c in sizes.index], edgecolor="black")
    ax.bar_label(bars, fmt="%d", padding=3)
    ax.set_ylabel("Number of Respondents"); ax.set_title("Respondents per Cluster")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/08_cluster_sizes.png", bbox_inches="tight"); plt.close()
    print("\n  Saved → 08_cluster_sizes.png")
    return summary, sizes


# ----------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------
def main():
    print("\n" + "="*60)
    print("  MENTAL HEALTH IN TECHNOLOGY RELATED JOBS – UNSUPERVISED ML PIPELINE")
    print("  DLBDSMLUSL01  |  IU International University")
    print("="*60)

    df_raw = load_data(DATA_FILE)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    if AGE_COL in df_raw.columns:
        clean_age(df_raw[AGE_COL]).hist(bins=30, ax=axes[0],
                                        color=PALETTE[0], edgecolor="black")
        axes[0].set_title("Age Distribution (cleaned)"); axes[0].set_xlabel("Age")
    miss = df_raw.isnull().mean().sort_values(ascending=False).head(15)
    miss.index = [c[:50] + "…" if len(c) > 50 else c for c in miss.index]
    miss.plot(kind="barh", ax=axes[1], color=PALETTE[1])
    axes[1].set_title("Top-15 Missing Value Fractions")
    axes[1].set_xlabel("Fraction missing")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/00_exploration.png", bbox_inches="tight"); plt.close()
    print("\n  Saved → 00_exploration.png")

    df_clean = preprocess(df_raw)
    X, df_encoded = engineer_features(df_clean)
    X_pca, X_pca2, X_tsne = reduce_dimensions(X)
    best_k = find_optimal_k(X_pca)
    labels = fit_final_kmeans(X_pca, best_k)

    print(f"\n{'='*60}\n6. VISUALISATIONS\n{'='*60}")
    plot_clusters_2d(X_pca2, labels,
                     f"K-Means Clusters (k={best_k}) – PCA 2-D",
                     "03_clusters_pca2d.png", best_k)
    plot_clusters_2d(X_tsne, labels,
                     f"K-Means Clusters (k={best_k}) – t-SNE",
                     "04_clusters_tsne.png", best_k)
    plot_silhouette_diagram(X_pca, labels, best_k)
    plot_cluster_profiles(df_encoded, labels)
    plot_heatmap(df_encoded, labels)

    cluster_statistics(df_encoded, labels, best_k)

    print(f"\n{'='*60}\n  PIPELINE COMPLETE  –  All outputs saved.\n{'='*60}\n")


if __name__ == "__main__":
    main()
