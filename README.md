# Computational Intelligence Project - Genetic Algorithm for Text Inscriptions

## Project Overview

This project implements a **Genetic Algorithm (GA)** to optimize the reconstruction of a **damaged ancient Greek inscription**. The goal is to use GA to find the best possible reconstruction of the inscription by adding missing words that improve the similarity to existing inscriptions. The algorithm uses **TF-IDF vectorization** to encode text and **cosine similarity** to measure the similarity between the damaged inscription and the reference texts.

The project combines natural language processing (NLP) techniques with evolutionary algorithms to address the problem of text reconstruction.

## Key Features

- **Text Extraction and Vectorization**: The ancient Greek texts are extracted from a dataset and encoded using **TF-IDF vectorization**.
- **Cosine Similarity**: Cosine similarity is used to evaluate the similarity between the damaged inscription and the reference texts.
- **Genetic Algorithm**: A genetic algorithm is implemented to evolve solutions that improve the damaged inscription by adding words that increase its similarity to existing texts.
- **Parameter Tuning**: The genetic algorithm is configured with adjustable parameters such as population size, mutation rate, and crossover probability.
- **Performance Tracking**: The performance of the GA is tracked through the best fitness score over generations, and the results are visualized.

## Dataset

The dataset used in this project consists of ancient Greek inscriptions. You can find the dataset [here](Dataset/iphi2802.csv). The data is filtered for inscriptions from **Greater Syria and the East (region_main_id = 1693)**.

## Project Structure

The project is organized into the following main components:

1. **Text Extraction and Vectorization**: The text data is read, filtered, and transformed into numerical vectors using TF-IDF.
2. **Genetic Algorithm**:
   - The GA is used to evolve potential solutions by adding missing words to the damaged inscription.
   - The fitness of each solution is calculated based on cosine similarity with the most similar texts.
3. **Results Analysis and Visualization**: The best solutions are visualized through fitness score graphs, and the optimal reconstructed inscription is displayed.
