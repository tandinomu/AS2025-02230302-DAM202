# Practical 1
# Text Preprocessing Practical Report


---

## Text Preprocessing Overview

Text preprocessing is a fundamental step in Natural Language Processing (NLP) that transforms raw, unstructured text data into a clean, standardized format suitable for machine learning algorithms. Raw text often contains inconsistencies, noise, and irrelevant information such as special characters, punctuation, emojis, HTML tags, and irregular spacing that can negatively impact model performance.

The preprocessing process involves several sequential steps including normalization (converting to lowercase), cleaning (removing unwanted characters), tokenization (splitting into words), lemmatization (reducing words to base forms), and stop word removal (filtering common words). The ultimate goal is to extract meaningful features from text and convert them into numerical representations that machine learning algorithms can process effectively.

This practical focuses on building a robust, reusable text preprocessing pipeline applicable to various downstream NLP tasks such as sentiment analysis, spam detection, text classification, information retrieval, and topic modeling.

---

## 1. Dataset Creation

A sample dataset was created containing 8 sentences with various text challenges including mixed cases, punctuation marks, emojis, numbers, currency symbols, contractions, citations, and repeated words. The list was converted into a pandas DataFrame and display settings were configured to show full content without truncation.

---

## 2. Text Normalization

Text normalization was performed by converting all text to lowercase. This ensures consistency in the dataset so that words like "Lemon" and "lemon" are treated as the same token. A copy of the original dataframe was created and the normalized results were stored in a new column called 'clean_sentence'.

**Example transformation:**
- Original: "When life gives you lemons, make lemonade! 🙂"
- After normalization: "when life gives you lemons, make lemonade! 🙂"

---

## 3. Text Cleaning

Unwanted elements were removed from the text using regular expressions. First, specific citations like "[Wikipedia]" and "[AllRecipes]" were removed. Then a comprehensive regex pattern was applied that removed:

- URLs (HTTP/HTTPS and www)
- HTML tags
- Email addresses
- Social media mentions (@username)
- Hashtags (#hashtag)
- Non-alphanumeric characters (punctuation, emojis, special symbols)

Whitespace was also normalized by replacing multiple spaces with single spaces and removing leading/trailing spaces.

**Example transformation:**
- Before cleaning: "when life gives you lemons, make lemonade! 🙂"
- After cleaning: "when life gives you lemons make lemonade"

---

## 4. Advanced Text Processing with spaCy

spaCy's English language model (en_core_web_sm) was installed and loaded to perform advanced linguistic processing.

### 4.1 Tokenization

spaCy was used to split the cleaned text into individual tokens (words). The tokenizer intelligently handled the text and broke it down into separate word units.

**Example:**
- Input: "when life gives you lemons make lemonade"
- Tokens: ['when', 'life', 'gives', 'you', 'lemons', 'make', 'lemonade']

### 4.2 Lemmatization

Lemmatization was applied to reduce words to their base or dictionary form. This process uses linguistic knowledge to find the correct root form of words, considering their part of speech.

**Example transformations:**
- "gives" → "give"
- "lemons" → "lemon"
- "running" → "run"

### 4.3 Stop Words Removal

Stop words were removed using spaCy's built-in list of 326 common English stop words. These are high-frequency words like "the", "and", "is", "you", "when" that carry little semantic meaning. Removing them helps focus on more meaningful content.

**Example transformation:**
- Before: "when life gives you lemons make lemonade"
- After: "life give lemon lemonade"

---

## 5. Creating Reusable Functions

A reusable function called `token_lemma_stopw(text)` was created that combines lemmatization and stop word removal. This function takes a text input, processes it through spaCy, extracts lemmatized forms while filtering out stop words, and returns the cleaned text.

Two additional functions were developed:
- `lower_replace(series)`: Handles normalization and regex-based cleaning
- `nlp_pipeline(series)`: Orchestrates the complete preprocessing workflow

The complete pipeline function combines all preprocessing steps into a single workflow, making it easy to apply consistent transformations to any dataset. The processed data was saved using pickle format for efficient future loading.

---

## 6. Word Representation (Vectorization)

### 6.1 Count Vectorization (Bag of Words)

Count Vectorization was implemented to transform the preprocessed text into numerical matrix representation. Each row represents a document and each column represents a unique word in the corpus. The cell values indicate how many times each word appears in each document.

Filtering was applied using `min_df=2` to include only words that appear in at least 2 documents, which helps reduce noise and focus on more meaningful terms. Term frequencies were calculated to analyze which words appeared most frequently across the entire dataset.

### 6.2 TF-IDF Vectorization

TF-IDF (Term Frequency-Inverse Document Frequency) vectorization was implemented, which weighs words based on both their frequency in a document and their rarity across the corpus. This method gives higher weights to distinctive words and lower weights to common words.

The TF-IDF formula calculates:
- Term Frequency: How often a word appears in a specific document
- Inverse Document Frequency: How rare the word is across all documents

TF-IDF was applied with `min_df=2` filtering and bigram analysis was experimented with using `ngram_range=(1,2)`. This captured meaningful two-word phrases like "arnold palmer" and "ice tea" that carry more context than individual words.

---

## 7. Results and Observations

The complete preprocessing pipeline successfully transformed raw text into clean, structured data:

**Complete transformation example:**
- **Original:** "When life gives you lemons, make lemonade! 🙂"
- **Final output:** "life give lemon lemonade"

The pipeline achieved:
- Removal of all emojis, punctuation, and special characters
- Case normalization for consistency
- Lemmatization of inflected word forms
- Elimination of 326 common stop words
- Significant vocabulary reduction while preserving semantic meaning

**Vectorization results:**
- Count Vectorization captured word frequency patterns in a sparse matrix format
- TF-IDF identified distinctive terms with scores closer to 1.0 indicating more informative words
- Bigram analysis revealed meaningful phrases that unigrams alone would miss

---

## 8. Key Learnings

Key concepts learned through this practical include:

1. **Sequential Processing**: Text preprocessing follows a logical sequence from normalization to vectorization
2. **Tool Selection**: Different tools serve different purposes - pandas for basic operations, spaCy for linguistic processing, and scikit-learn for vectorization
3. **Vocabulary Reduction**: Preprocessing significantly reduces vocabulary size through normalization, lemmatization, and stop word removal
4. **Modular Design**: Creating reusable functions ensures consistency and maintainability
5. **Feature Engineering**: Different vectorization methods (Count vs TF-IDF) produce different feature representations suitable for different tasks

The preprocessing pipeline built is applicable to various NLP tasks including text classification, sentiment analysis, spam detection, information retrieval, and topic modeling. The choice of preprocessing steps depends on the specific use case and the nature of the text data.

---

## 9. Conclusion

A comprehensive text preprocessing pipeline was successfully implemented that transforms raw text into clean numerical representations suitable for machine learning algorithms. The pipeline includes normalization, cleaning, tokenization, lemmatization, stop word removal, and vectorization. All preprocessing tasks were implemented correctly, producing a clean dataset ready for modeling contexts. The modular design ensures the pipeline can be reused across different datasets and applications.