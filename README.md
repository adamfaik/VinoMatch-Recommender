# VinoMatch 🍷

**VinoMatch is an intelligent wine steward that uses Natural Language Processing to provide personalized wine recommendations.** This application allows users to discover new wines by either selecting a wine they already enjoy or by describing their preferences in natural language.

**Live demo:** https://vinomatch.streamlit.app/

---

### Project goals and business impact

The primary goal of this project is to move beyond simple rating-based recommendations and create a more intuitive and human-like experience for discovering wine. For a business, such as an online wine retailer, the impact is twofold:

1.  **Enhanced user experience:** By allowing users to search in their own words, we reduce the "paradox of choice" and make the discovery process more engaging and less intimidating.
2.  **Increased sales and engagement:** Personalized, explainable recommendations build user trust and can lead to higher conversion rates and customer loyalty.

---

### Key features

* **Hybrid recommendation engine:** Combines TF-IDF for textual similarity with numerical features like price, points, and a custom "value score" for highly accurate recommendations.
* **Natural language search:** Users can type descriptions like "a fruity, full-bodied red" to get relevant suggestions.
* **Conversational summaries:** Integrates the Gemini API to provide friendly, "wine steward" style summaries explaining *why* the wines are a good match.
* **Interactive filtering:** Results can be filtered by variety, price range, points, and country.
* **Explainability:** Each recommendation includes the top shared keywords that drove the similarity score, providing transparency into the model's reasoning.

---

### The dataset: wine reviews from Kaggle

This project uses the [Wine Reviews dataset](https://www.kaggle.com/datasets/zynicide/wine-reviews), which contains over 130,000 reviews from wine experts. Here's a breakdown of the key features used:

* **`description`**: This is a tasting note written by a professional wine taster. It's a rich, descriptive text that details the wine's aromas, flavors, and texture, forming the heart of our NLP project.
* **`points`**: A score given by a wine critic, typically on a 100-point scale, to quickly gauge the overall quality of a wine.
* **`price`**: The retail price of a bottle of the wine in USD.
* **`title`**: The full, official name of the wine, usually including the winery, vintage, and specific name.
* **`variety`**: The type of grape used to make the wine (e.g., Chardonnay, Pinot Noir). This is the single biggest factor influencing a wine's flavor.
* **`country`, `province`, `region_1`**: These fields describe the wine's origin. The place where grapes are grown—its **terroir**—has a massive impact on a wine's character.

---

### Technical overview

This project follows a comprehensive data science workflow:

1.  **Exploratory Data Analysis (EDA):** Analyzed the Wine Reviews dataset to uncover patterns in pricing, scoring, and the language of wine descriptions. (Found in `[01_EDA_and_Data_Understanding.ipynb](https://github.com/adamfaik/VinoMatch-Recommender/blob/main/01_EDA_and_Data_Understanding.ipynb)`)
2.  **Preprocessing and feature engineering:** Cleaned the data, handled missing values, and engineered new features like `value_score` and `readability_score`. (Found in `[02_Preprocessing_and_Feature_Engineering.ipynb](https://github.com/adamfaik/VinoMatch-Recommender/blob/main/02_Preprocessing_and_Feature_Engineering.ipynb)`)
3.  **Model experimentation:** Systematically tested four different NLP and hybrid models (TF-IDF, Sentence-Transformers, custom Word2Vec, and a weighted hybrid model) to find the best performer. (Found in `[03_Model_Experimentation_and_Selection.ipynb](https://github.com/adamfaik/VinoMatch-Recommender/blob/main/03_Model_Experimentation_and_Selection.ipynb)`)
4.  **Deployment:** The final hybrid model was integrated into an interactive web application using Streamlit and deployed on Streamlit Community Cloud. (Found in `[app.py](https://github.com/adamfaik/VinoMatch-Recommender/blob/main/app.py)`)

---

### How to run locally

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/adamfaik/VinoMatch-Recommender.git](https://github.com/adamfaik/VinoMatch-Recommender.git)
    cd VinoMatch-Recommender
    ```

2.  **Set up the environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

3.  **Add your API Key:**
    * Create a folder: `.streamlit`
    * Inside it, create a file: `secrets.toml`
    * Add your Gemini API key to the file: `GEMINI_API_KEY = "YOUR_API_KEY_HERE"`

4.  **Run the app:**
    ```bash
    streamlit run app.py
    
