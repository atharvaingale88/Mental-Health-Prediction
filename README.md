# Mental Health in Tech: Predictive Analysis of Treatment-Seeking Behavior

## 1. Project Overview

This initiative focuses on developing a machine learning model to predict mental health treatment-seeking behavior among individuals within the technology industry. The project's core objective is to leverage a comprehensive survey dataset to discern and understand the multifaceted factors that influence an individual's decision to seek professional mental health support. The insights gleaned from this analysis are intended to contribute to a data-driven understanding of mental well-being in the tech sector and to inform the development of targeted, effective interventions.<!----><!----><!----><!----><!---->

The significance of this project stems from the universal prevalence of mental health challenges, which are often compounded by the demanding environment characteristic of the tech industry. By meticulously analyzing this specific dataset, the project aims to transcend mere statistical prevalence, seeking to uncover actionable understandings regarding attitudes, perceived barriers, and facilitating factors related to mental health treatment. Such an understanding is paramount for crafting support systems that are precisely tailored to the unique needs of tech professionals, thereby moving beyond generic solutions toward more impactful, individualized approaches.<!----><!----><!----><!----><!---->

This analytical undertaking is designed to transition from observational data to actionable strategies. The project's utility extends beyond a simple description of the mental health landscape in tech. Instead, it provides a predictive framework capable of identifying individuals who are likely to seek treatment and, crucially, illuminating the specific factors that either encourage or deter this vital step. This shifts the project's inherent value from purely academic comprehension to the practical development of intervention strategies for human resources departments and organizational leadership.


## 2. Dataset Description

The primary dataset underpinning this analysis, `survey.csv`, is derived from surveys conducted by Open Sourcing Mental Health (OSMI), a non-profit organization dedicated to advancing mental health awareness and support within the tech community. This dataset captures responses predominantly from the 2014 survey, offering a valuable snapshot of perceptions and experiences regarding mental health across the global tech industry.<!----><!----><!----><!----><!---->

The `survey.csv` dataset comprises 1259 individual records and 27 distinct features, providing a rich foundation for exploring mental health dynamics in the workplace. A detailed breakdown of the columns is as follows:<!----><!----><!----><!----><!---->

- `Timestamp`: Records the date and time when the survey was submitted (DateTime).

- `Age`: Represents the numerical age of the respondent (Numeric).

- `Gender`: Captures the self-reported gender of the respondent (Categorical).

- `Country`: Indicates the respondent's country of residence (Categorical).

- `state`: Specifies the US state of residence, primarily applicable to US-based respondents (Categorical).

- `self_employed`: A binary indicator of whether the respondent is self-employed (Categorical: Yes/No/NA).

- `family_history`: Denotes whether the respondent has a family history of mental illness (Categorical: Yes/No).

- `treatment`: This is the **Target Variable**, indicating whether the respondent has sought professional mental health treatment (Categorical: Yes/No).

- `work_interfere`: Describes the frequency with which mental health issues affect the respondent's work (Categorical: Often, Rarely, Sometimes, Never, NA).

- `no_employees`: Categorizes the number of employees within the respondent's company (Categorical: 1-5, 6-25, 26-100, 100-500, 500-1000, More than 1000).

- `remote_work`: Indicates if the respondent works remotely (Categorical: Yes/No).

- `tech_company`: Specifies if the employer is a technology-focused company (Categorical: Yes/No).

- `benefits`: Reveals whether the employer provides mental health benefits (Categorical: Yes/No/Don't know/Not sure).

- `care_options`: Indicates if the employer discusses mental health care options with employees (Categorical: Yes/No/Not sure).

- `wellness_program`: Denotes if the employer offers a wellness program (Categorical: Yes/No/Don't know).

- `seek_help`: Shows if the employer provides resources for seeking mental health assistance (Categorical: Yes/No/Don't know).

- `anonymity`: Addresses whether anonymity is protected when seeking mental health help through the employer (Categorical: Yes/No/Don't know).

- `leave`: Assesses the perceived ease of taking medical leave for mental health reasons (Categorical: Very easy, Somewhat easy, Somewhat difficult, Very difficult, Don't know).

- `mental_health_consequence`: Reflects perceived negative repercussions for discussing mental health with an employer (Categorical: Yes/No/Maybe).

- `phys_health_consequence`: Reflects perceived negative repercussions for discussing physical health with an employer (Categorical: Yes/No/Maybe).

- `coworkers`: Measures the comfort level discussing mental health with colleagues (Categorical: Yes/No/Some of them).

- `supervisor`: Measures the comfort level discussing mental health with a supervisor (Categorical: Yes/No/Some of them).

- `mental_health_interview`: Indicates willingness to raise mental health issues during a potential employer interview (Categorical: Yes/No/Maybe).

- `phys_health_interview`: Indicates willingness to raise physical health issues during a potential employer interview (Categorical: Yes/No/Maybe).

- `mental_vs_physical`: Assesses the perceived equality of mental and physical health seriousness by the employer (Categorical: Yes/No/Don't know).

- `obs_consequence`: Indicates awareness of negative consequences for coworkers who discussed mental health issues (Categorical: Yes/No).

- `comments`: Contains open-ended textual feedback from respondents (Text).

The `results.csv` file, which represents the output of this project's predictive model, contains two columns: `Index` (corresponding to the original row index from `survey.csv`) and `Treatment` (the predicted binary outcome for the `treatment` variable, where 0 typically denotes 'No' and 1 indicates 'Yes').<!----><!----><!----><!----><!---->

Initial inspection of the dataset reveals several data quality considerations that necessitate rigorous preprocessing. Various columns, such as `state` (for non-US respondents), `self_employed`, `work_interfere`, and `comments`, frequently contain missing values, marked as `NA`. These require careful handling, such as imputation or strategic removal, to prevent bias in the model. A notable challenge arises from inconsistent categorical entries, particularly within the<!----><!----><!----><!----><!---->

`Gender` column, which exhibits a wide array of non-standardized responses (e.g., "M", "Female", "male", "Trans-female", "Cis Female", "something kinda male?", "Male-ish", "maile", "f", "Woman", "male leaning androgynous", "Femake", "fluid", "Enby", "Nah", "All", "p"). This necessitates a robust standardization process to group similar responses into coherent categories. Furthermore, the<!----><!----><!----><!----><!---->

`Age` column presents extreme outliers, including negative values (`-29`, `-1`) and improbably large numbers (`99999999999`, `329`). Such anomalies require careful outlier detection and treatment, likely through removal or replacement with a more plausible value (e.g., the median of valid ages), to prevent distortion of statistical measures and model training.<!----><!----><!----><!----><!---->

The successful and reliable operation of the machine learning model, as well as the validity of any derived conclusions, will be highly dependent on the thoroughness and intelligence applied during the data preprocessing and feature engineering phases. This underscores a critical understanding: data science, in many real-world applications, often demands more effort in data preparation and cleansing than in the actual model construction. This foundational work is paramount for ensuring the integrity of the analytical outcomes.

The project's central focus is on predicting `treatment` seeking behavior. However, the survey also captures information regarding `family_history`, perceived `mental_health_consequence`, and includes qualitative `comments`. These comments frequently reveal that respondents may experience mental health issues but choose not to seek treatment due to various factors such as stigma, financial constraints, or a perceived lack of efficacy in available support. This distinction is critical: the model is not merely predicting the<!----><!----><!----><!----><!---->

_presence_ of a mental health condition, but rather the _behavioral decision_ to seek professional help. This reframes the problem from a purely medical perspective to one that encompasses broader social and organizational psychological dimensions, highlighting the complex interplay of individual perceptions and environmental factors.


## 3. Technical Stack

The technical infrastructure supporting this project is built upon a comprehensive suite of Python libraries and frameworks, as meticulously detailed in the `requirements.txt` file. This robust collection facilitates every phase of the data science lifecycle, from data ingestion and cleaning to advanced model development, rigorous evaluation, and potential deployment.<!----><!----><!----><!----><!---->

The core of the predictive modeling effort is powered by `scikit-learn`, which is utilized for implementing traditional machine learning algorithms, specifically a Decision Tree Classifier. For scenarios demanding more complex or scalable solutions, the inclusion of<!----><!----><!----><!----><!---->

`tensorflow`, `keras`, and `jax` provides powerful deep learning capabilities. `mlxtend` further augments the toolkit by offering specialized utilities for model evaluation and feature engineering, thereby enhancing the predictive power of the models.

For efficient data handling, `numpy` and `pandas` serve as foundational libraries, enabling robust data loading, manipulation, and extensive cleaning, which is particularly crucial given the raw data's initial characteristics.

Data visualization is comprehensively addressed through a diverse set of libraries, including `matplotlib`, `seaborn and plotly`. These tools facilitate thorough exploratory data analysis (EDA) and effective communication of findings, allowing for the creation of both static statistical plots and interactive dashboards.

A notable aspect of the technical stack is the inclusion of `Flask`.`Flask` provides a lightweight framework for developing web applications. This suggests an intention to deploy the project's findings in an accessible, interactive format, thereby broadening the practical utility and reach of the research.

**4. Analysis and Modeling Approach**

The central analytical objective of this project is to construct a robust machine learning model capable of predicting whether an individual in the tech industry has sought professional treatment for a mental health issue. This objective is framed as a binary classification problem, with the `treatment` column from the `survey.csv` dataset serving as the designated target variable.<!----><!----><!----><!----><!---->

Prior to the development of the predictive model, a thorough Exploratory Data Analysis (EDA) phase was conducted. This critical step involved systematically examining the distributions of individual variables, identifying and addressing outliers (e.g., in the `Age` column), meticulously handling missing values (e.g., `NA` entries across various fields), and standardizing inconsistent categorical data (e.g., the highly varied entries in the `Gender` column). EDA also encompassed the exploration of preliminary relationships and correlations between features, providing foundational understandings that guided subsequent feature engineering and the selection of the appropriate model.<!----><!----><!----><!----><!---->

The core predictive model employed in this project is a **Decision Tree Classifier**, implemented using the `scikit-learn` library. This choice is particularly well-suited for classification tasks due to its interpretability, which is invaluable for understanding the specific factors influencing treatment-seeking behavior. To enhance the model's predictive power, significant feature engineering was performed. This involved transforming raw survey responses into more informative and impactful features. Key examples include the direct utilization of cleaned `Age`, `self_employedand` and `family_history` data.

<!----><!----><!----><!----><!---->

Given the identified data quality issues, comprehensive preprocessing steps were fundamental to constructing a robust and reliable predictive model. These steps included:

- **Standardization of Categorical Variables:** Consolidating the numerous varied entries in the `Gender` column into a consistent and manageable set of categories.

- **Handling Missing Data:** Employing strategic approaches such as mode imputation for categorical features or median imputation for numerical features, or the judicious removal of records or columns with excessive missingness, to ensure data completeness without introducing bias.

- **Outlier Treatment:** Addressing anomalous values in the `Age` column, such as negative or implausibly large numbers, through methods like capping or removal, to prevent them from skewing the model's learning process.<!----><!----><!----><!----><!---->

The emphasis on feature engineering, particularly the creation of interaction terms, signifies an analytical depth that extends beyond merely applying an algorithm to raw data. This approach reflects a hypothesis-driven exploration designed to uncover non-obvious relationships within the dataset. For instance, the combined influence of a respondent's age and the size of their workplace (`no_employees`) might collectively exert a different or stronger influence on treatment-seeking behavior than either factor considered in isolation. This demonstrates a sophisticated understanding of both data manipulation and the underlying domain.


## 5. Results and Key Findings

The `results.csv` file constitutes the primary output of the machine learning model, containing the predicted binary outcomes for the `treatment` variable for each respondent originally present in the `survey.csv` dataset. Each row in<!----><!----><!----><!----><!---->

`results.csv` uniquely identifies a respondent by their `Index`, with the `Treatment` column indicating the model's prediction (0 for 'No' treatment sought, 1 for 'Yes' treatment sought). This file directly addresses the project's core predictive question.

The performance of the Decision Tree Classifier is rigorously assessed using standard classification metrics to evaluate its effectiveness and reliability. These metrics typically include:

- **Accuracy:** The overall proportion of correctly predicted instances, encompassing both individuals who sought treatment and those who did not.

- **Confusion Matrix:** A tabular summary detailing the performance of the classification model, explicitly showing the counts of true positives, true negatives, false positives, and false negatives. This provides a granular breakdown of correct and incorrect predictions for each class.
