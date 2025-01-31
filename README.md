This project aims to analyze admission data from the Medical Information Mart for Intensive Care (MIMIC) database. The dataset is read from a CSV file, converted to appropraite data types and visualized to provide insights into the number of admissions per subject and the number of patients per admission number.

### 1. **Numerical Data**
   - **Definition**: Numerical data consists of values that are measurable and can be ordered. They can be further classified as:
     - **Continuous**: Can take any value within a range (e.g., height, weight, temperature).
     - **Discrete**: Can only take specific values, often counts (e.g., number of students, number of pets).
   - **EDA Techniques**:
     - **Descriptive Statistics**: Mean, median, mode, variance, standard deviation, quantiles.
     - **Visualizations**: Histograms, box plots, density plots, scatter plots (for bivariate analysis), and pair plots (for multivariate analysis).
     - **Outlier Detection**: Box plot visualization, Z-score method, and IQR (Interquartile Range) method.
     - **Distribution Analysis**: To understand if the data follows a specific distribution (e.g., normal distribution).

### 2. **Categorical Data**
   - **Definition**: Categorical data represents discrete categories or labels that can either be nominal (without order) or ordinal (with order).
     - **Nominal**: Categories without a meaningful order (e.g., gender, color).
     - **Ordinal**: Categories with a meaningful order (e.g., rating levels such as "poor", "good", "excellent").
   - **EDA Techniques**:
     - **Frequency Counts**: Count of each category or mode.
     - **Visualizations**: Bar plots, pie charts, and count plots.
     - **Cross-tabulation**: For bivariate analysis, to see relationships between two categorical variables.
     - **Encoding Techniques**: Transform categorical data into numerical forms, such as one-hot encoding, to facilitate further analysis.

### 3. **Date and Time Data (Temporal Data)**
   - **Definition**: Temporal data includes data points measured over time, like dates or timestamps, which can capture trends, seasonality, or patterns over intervals.
   - **EDA Techniques**:
     - **Time Series Plots**: Line plots to observe trends over time.
     - **Seasonal Analysis**: Breaking data into seasonal components (e.g., monthly or weekly trends).
     - **Trend Analysis**: Identifying upward or downward trends over time.
     - **Lag Plots**: To understand if there is any correlation between current and previous values (useful for time series data).
     - **Rolling Statistics**: Calculate moving averages or other rolling statistics to smooth out short-term fluctuations and highlight trends.

### 4. **Text Data (Unstructured Data)**
   - **Definition**: Text data is qualitative data that can be transformed into structured formats (e.g., word counts, n-grams) for analysis. 
   - **EDA Techniques**:
     - **Text Preprocessing**: Tokenization, stop word removal, stemming, and lemmatization.
     - **Word Clouds**: To visualize the most frequent words.
     - **Frequency Distribution**: Count the frequency of words or phrases.
     - **N-gram Analysis**: To find common phrases of two, three, or more words.
     - **Sentiment Analysis**: Gauge the positive or negative sentiment in the text.
     - **Topic Modeling**: Techniques like LDA (Latent Dirichlet Allocation) to identify topics within text data.

### 5. **Mixed Data Types**
   - **Definition**: Many datasets have a mix of numerical, categorical, and date/time data.
   - **EDA Techniques**:
     - **Correlation Analysis**: Check correlations between numerical variables or with encoded categorical variables.
     - **Multi-Variable Plots**: Visualizations like heatmaps, pair plots, and grouped bar plots to study interactions across types.
     - **Data Transformation and Encoding**: Encode categorical data and transform date/time data to add relevant information, like day of the week or time interval, for richer insights.

