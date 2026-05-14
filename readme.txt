-------------------
DOCUMENT START
-------------------

# Mental Health in Technology Related Jobs – Unsupervised Learning Pipeline

*Course:* DLBDSMLUSL01 – Machine Learning: Unsupervised Learning and Feature Engineering
**Task:** Case Study 1.1 – Mental Health in Technology related Jobs
***Institution:*** IU International University of Applied Sciences

-------------------

## Overview

This project applies an unsupervised machine learning pipeline (PCA + t-SNE + K-Means clustering) to the OSMI Mental Health in Tech Survey 2016 dataset. It segments survey respondents into interpretable clusters and produces visualizations to support HR decisionmaking.

-------------------

### Requirements

- * Python 3.9 or newer** (student tested on 3.14)
- ** The following Python packages:
  - 'numpy'
  - 'pandas'
  - 'matplotlib'
  - 'seaborn'
  - 'scikit-learn'

----------------------------------

#### Setup Instructions

* 1. Verify Python is installed

Open a CMD terminal (Command Prompt on Windows, Terminal on Mac/Linux) and run the following command:

python --version

You should see (Python 3.1.x) or higher. If you see a lower version of Python e.g. (Python 2.x) or "command not found", install Python from https://www.python.org/downloads/ 
IMPORTANT! Tick "Add Python to PATH" during installation.


** 2. Place the files in one/same folder together

NOTICE! Please make sure these three files are in the same folder:

Your folder structure content should look like below: 

your_project_folder/
|__ mental_health_analysis.py
|__ mental-heath-in-tech-2016_20161114.csv
|__ README.md

> IMPORTANT! **Please Notice:** The CSV filename from the source contains a typo ("heath" instead of "health"). This is the original Kaggle filename do not rename it, or update the `DATA_FILE` constant near the top of the script. If you modify it this script will fail to provide the proper desired /scoped output as the reviewer will download the file from its source, not your computer.

The dataset can be downloaded from: https://www.kaggle.com/osmi/mental-health-in-tech-2016 link provided in the 1.1 Task 1: Mental Health in Technology related Jobs task description.

> ATTENTION! You will need to create an account on kaggle.com prior to being able to download the .csv file. Before downloading make sure you have selected all 63 columns. The default on site selected preview has only 10 columns selected.

*** 3. Navigate to the folder in your CMD terminal

cd path/to/your_project_folder

**** 4. (Optional but recommended) Create a virtual environment (I highly recommend based on my experience with this case)

This isolates the project dependencies from your system Python

Run on CMD:

**Windows:** 

1. python -m venv venv

2. venv\Scripts\activate


**Mac/Linux:**

Run on terminal:

1. python -m venv venv

2. source venv/bin/activate


You will/should be able to see (venv) appear at the start of your terminal prompt when active.


***** 5. Install required packages

This is not y/n process. Once you run the command the installation of the packages begin.

Run:

pip install numpy pandas matplotlib seaborn scikit-learn


This downloads and installs all required libraries. The process takes around 5-10 minutes (it might be faster depending on your computer and connectivity circumstances).

----------------------------------

##### Running the Script

From within the project folder that you have already navigated to in CMD run the command;

Run:

python mental_health_analysis.py


> NOTICE! The pipeline will execute in roughly 1-2 minutes (time is relative). The t-SNE step is the slowest part please wait for it to complete.

###### Saving the console output to a file (Windows)

If you want to save the printed statistics to a text file, run these two commands in order:

Run: 

1. set PYTHONIOENCODING=utf-8

2. python mental_health_analysis.py > output.txt 2>&1


> NOTICE! The `PYTHONIOENCODING=utf-8` setting is required because the script uses Unicode characters (such as arrows and box-drawing characters) that the default Windows code page (cp1252) cannot encode when redirecting to a file.

####### Saving the console output to a file (Mac/Linux)

Run: 

1. python mental_health_analysis.py | tee output.txt

> NOTICE! The 'tee' command shows the output on screen and will save it to "output.txt"

----------------------------------

## Outputs

After the script completes successfully, the following files are created in the same folder:

| File                            | Contents                                            |
|---------------------------------|-----------------------------------------------------|
| 00_exploration.png              | Age distribution + Top-15 missing-value fractions   |
| 01_pca_scree.png                | PCA explained variance + 2D PCA projection          |
| 02_kmeans_elbow_silhouette.png  | Elbow and silhouette curves for k=2 to 9            |
| 03_clusters_pca2d.png           | K-Means clusters on 2D PCA projection               |
| 04_clusters_tsne.png            | K-Means clusters on t-SNE projection                |
| 05_silhouette_diagram.png       | Per respondent silhouette plot by cluster           |
| 06_cluster_profiles.png         | Bar charts of top-12 features by cluster            |
| 07_cluster_heatmap.png          | Normalized heatmap of all features by cluster       |
| 08_cluster_sizes.png            | Number of respondents per cluster                   |

The console output (in.txt format) contains:
- Initial data exploration (shape, dtypes, missing value summary)
- Cleaning summary (age range, gender distribution, columns dropped)
- Dimensionality reduction summary (number of PCA components retained)
- K-Means optimization results (inertia and silhouette score for each k)
- Per-cluster descriptive statistics (size, percentage, mean feature values)

----------------------------------

######## Expected Results

With the default random seed ( RANDOM_STATE = 42 ) and the OSMI 2016 dataset, the pipeline produces the following:

- * 27 principal components* retain ≥ 90 % of the total variance.
- ** Optimal number of clusters: k = 3** (silhouette score ≈ 0.30).
- *** Cluster sizes:*** Cluster 1 ≈ 1,015 (70.8 %), Cluster 2 ≈ 169 (11.8 %), Cluster 3 ≈ 249 (17.4 %).

----------------------------------

######### Possible Troubleshooting Experience

* ModuleNotFoundError: No module named 'pandas'*
The required packages are not installed. Run `pip install numpy pandas matplotlib seaborn scikit-learn` again, making sure the virtual environment is activated if you created one.

** FileNotFoundError: mental-heath-in-tech-2016_20161114.csv **
The CSV file is not in the same folder as the script, or the filename is different. Check the exact filename it includes the typo "heath" (not "health").

*** UnicodeEncodeError: 'charmap' codec can't encode character...***
Windows-only issue when redirecting output to a file. Run; ( set PYTHONIOENCODING=utf-8 ) before running the script (see "Saving the console output to a file" above).

**** Script hangs at "Running t-SNE..."****
This is normal. t-SNE takes a minute or two depending on the machine. Please be patient.

***** Plots look different from expected results*****
This may occur if your scikit-learn version is significantly different. The script was tested on scikit-learn 1.4 through 1.8. Re-run; ( pip install --upgrade scikit-learn ) if needed.


****** Note from personal experience (Windows10 + Python 3.14):

When I first tried to save the console output to a file using python mental_health_analysis.py > output.txt, the file only contained the first portion of the output (Section 1) and stopped abruptly. Running the script normally without redirection worked perfectly all images were generated and the full output appeared on screen.

The cause was a Unicode encoding issue: the script prints characters such as arrows (→) and box-drawing characters (═) which display fine in the terminal but cannot be encoded by Windows' default cp1252 code page when redirecting to a file. The script crashed silently on the first such character during redirection.

The fix that worked for me on Windows10 Command Prompt was to set the encoding explicitly before running the script:

Run:
1. set PYTHONIOENCODING=utf-8

2. python mental_health_analysis.py > output.txt 2>&1

The 2>&1 part also captures any error messages into the same file, which is useful for debugging. After applying this fix, the full output (all seven sections) was saved correctly.

******* Cosmetic issue can be observed on image (06_cluster_profiles): four of the subplot titles are too long and overlap with the title of the adjacent subplot. You can see this on three titles in particular:

Top row, far right: "Was your anonymity protected..." overlaps into "mh_benefits"
Bottom row, third panel: "Were you aware of the options..." overlaps into "have you sought treatment..." (or similar)
Bottom row, far right: title is also cut off / overlapping

This happens because some of the previous employer questions kept their full long form column names they were not in the SHORT_LABELS dictionary built in the script. Those questions are real and meaningful for the cluster interpretation. They are what defines Cluster 2.

----------------------------------

########## Final File Structure

your_project_folder/
|__ mental_health_analysis.py                # Main pipeline script
|__ mental-heath-in-tech-2016_20161114.csv   # Input dataset
|__ README.md                                # This file
|__ (generated outputs after running)
    |__ 00_exploration.png
    |__ 01_pca_scree.png
    |__ 02_kmeans_elbow_silhouette.png
    |__ 03_clusters_pca2d.png
    |__ 04_clusters_tsne.png
    |__ 05_silhouette_diagram.png
    |__ 06_cluster_profiles.png
    |__ 07_cluster_heatmap.png
    |__ 08_cluster_sizes.png


########### License & Attribution

- * Code:* Student referenced implementation for academic coursework ref. DLBDSMLUSL01 from different resources.

- ** Dataset:** OSMI Mental Health in Tech Survey 2016. Made available by Open Sourcing Mental Illness Ltd. on Kaggle (https://www.kaggle.com/osmi/mental-health-in-tech-2016).

- *** Course material reference:*** Sayed-Mouchaweh, M. & Müller-Kett, C. (2025). *Machine Learning – Unsupervised Learning and Feature Engineering*. Course Book DLBDSMLUSL01. IU International University of Applied Sciences.

References for the pipeline implementation draws on standard patterns/coding from:

- **** Géron, A. (2022). Hands-on machine learning with Scikit-Learn, Keras, and TensorFlow **** (3rd ed.). O'Reilly Media.
- ***** Scikit-learn user guide *****: https://scikit-learn.org/stable/user_guide.html
- ****** Course book DLBDSMLUSL01 ******, IU International University of Applied Sciences.
- ... The script header of (python mental_health_analysis.py) docstring contains the full implementation and algorithmic reference list in formal academic citation and section by section attribution format.

-------------------
DOCUMENT END
-------------------
