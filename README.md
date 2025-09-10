# Word2Vec Practical: 

In this practical, is aimed to learning how to train the **Word2Vec models from scratch** using Python and the **Gensim** library. This focuses on creating custom word embeddings using *Alice's Adventures in Wonderland* as the training dataset.


## 1. Setup and Data Preparation

The practical begins by setting up the working directory and importing the **Alice in Wonderland** text file.  
The code reads the text data line by line and stores it in a list structure, with each line becoming one item in the dataset.


## 2. Data Quality Assessment

A **comprehensive data quality assessment function** analyzes the text dataset by calculating:

- **Total documents:** 3,598  
- **Vocabulary size:** 4,950 unique words  
- **Average sentence length:** 7.4 words  
- **Vocabulary diversity:** 0.1870  

This assessment helps determine if the dataset is suitable for effective Word2Vec training.


## 3. Text Preprocessing Pipeline

It implements an **advanced text preprocessing system** using **NLTK**.  
The `AdvancedTextPreprocessor` class provides customizable options for cleaning text, including:

- Converting text to lowercase  
- Removing punctuation and numbers  
- Filtering words by length  
- Tokenizing into sentences and words  
- Optional removal of stopwords and lemmatization  

After preprocessing, the *Alice in Wonderland* text is converted into:

- **2,941 clean sentences**  
- **2,519 unique words in vocabulary**



## 4. Parameter Recommendation System

A parameter recommendation function suggests **optimal Word2Vec settings** based on corpus characteristics.  

For the Alice dataset, it recommends:

- **Vector size:** 50 dimensions  
- **Window size:** 6 words  
- **Minimum word count:** 1  
- **Algorithm:** Skip-gram (`sg=1`)  
- **Training epochs:** 15  
- **Negative sampling:** 10 samples  


## 5. Model Training

The **Word2Vec model** is trained using **Gensim** with optimized parameters for the Alice dataset:

- **Vector size:** 150 dimensions  
- **Window size:** 8 words  
- **Minimum count:** 2 occurrences  
- **Algorithm:** Skip-gram  
- **Training epochs:** 50  
- **Negative samples:** 15  

The training process includes **real-time progress monitoring**, showing each epoch’s completion time.  

- **Final vocabulary size:** 1,396 words  
- **Training time:** 2.42 seconds  


## 6. Model Evaluation

It implements a comprehensive evaluation system (`Word2VecEvaluator` class) to test performance using:

- **Word similarity tasks** (model vs. human judgment)  
- **Analogy solving**  
- **Odd-one-out identification**  
- **Vocabulary coverage analysis**

### Evaluation Results:
- **Word similarity:** Spearman correlation = 0.4000 (4 valid word pairs)  
- **Analogy accuracy:** 33.33% (1 correct out of 3 valid analogies)  



## 7. Practical Results

The trained model demonstrates its learning with the following examples:

- **Most similar words to "alice":**  
  - uncomfortable (0.5413)  
  - she (0.5239)  
  - grave (0.4982)  
  - thoughtfully (0.4899)  
  - cautiously (0.4886)  

- **Similarity between "alice" and "rabbit":** 0.2926  



## Conclusion

The practical successfully demonstrates the **complete pipeline** from raw text to trained word embeddings with **hands-on experience** in creating **domain-specific Word2Vec models**.
