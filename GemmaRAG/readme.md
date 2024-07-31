# GemmaAI Assistant

This repository contains the code for the `GemmaAI Assistant`, a helpful AI assistant designed to answer questions based on a provided knowledge base. The assistant leverages advanced natural language processing (NLP) models and embeddings to generate accurate and contextually relevant responses.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Classes and Functions](#classes-and-functions)
- [License](#license)

## Overview

The `GemmaAI Assistant` is built using state-of-the-art transformer models and embeddings. It uses the `transformers` library for the language model and `sentence-transformers` for embeddings. The assistant can learn from a knowledge base, index the knowledge for fast retrieval, and generate contextually relevant answers to user queries.

## Features

- Utilises advanced transformer models for text generation.
- Employs sentence embeddings for efficient context retrieval.
- Cleans and processes text data from Wikipedia.
- Provides a modular and extensible codebase.

## Installation

Ensure you have the libraries given in requirements.txt installed.

You can install these libraries using pip:

pip install -r requirements.txt

This will install all the necessary libraries listed in the `requirements.txt` file.

## Requirements

The memory requirements will vary depending on the language model (LM) used. For the configuration provided in Gemma, you will need between 4 GB and 20 GB of graphics card memory to run the process.

## Usage

1. **Initialise the model and assistant:**

# Initialise the model
modelName = "/path/to/your/model"
gemmaAIAssistant = AIAssistant(gemmaModel=GemmaHF(modelName))

# Learn the knowledge base
gemmaAIAssistant.learnKnowledgeBase(knowledgeBase=extractedWikiTexts)

# Save the embeddings for later use
gemmaAIAssistant.saveEmbeddings()

2. **Query the assistant:**

# Ask a question
gemmaAIAssistant.query("What is artificial intelligence?")

## Classes and Functions

### `GemmaHF`

A class to initialise and manage the transformer model.

- `__init__(self, modelName, maxSeqLength=2048)`
- `generate(self, prompt, maxNewTokens=2048, temperature=0.0)`

### `AIAssistant`

A class to manage the AI assistant, including learning and querying the knowledge base.

- `__init__(self, gemmaModel)`
- `learnKnowledgeBase(self, knowledgeBase)`
- `indexEmbeddings(self)`
- `query(self, query)`
- `saveEmbeddings(self, filename="embeddings.npy")`
- `loadEmbeddings(self, filename="embeddings.npy")`

### Utility Functions

- `getDevice()`
- `initModel(modelName, device, maxSeqLength)`
- `getEmbedding(text, embeddingModel)`
- `cleanText(txt, EOSTOKEN)`
- `generateAnswer(question, data, searcher, embeddingModel, model, maxNewTokens=2048)`
- `map2Embeddings(data, embeddingModel)`
- `cleanString(input_string)`
- `getWikiPagesSingle(wiki, categoryName)`
- `getWikiPagesAll(categories)`

## License

This project is licensed under the Apache 2.0 License. See the [LICENSE](https://www.apache.org/licenses/LICENSE-2.0) file for details.