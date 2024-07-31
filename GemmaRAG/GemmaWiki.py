# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# SPDX-FileCopyrightText: [2024] University of Manchester

# SPDX-License-Identifier: apache-2.0

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
warnings.filterwarnings("ignore")

# +
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
import scann
import wikipediaapi

import torch

import transformers
from transformers import (AutoModelForCausalLM, 
                          AutoTokenizer, 
                          BitsAndBytesConfig,
                         )
from sentence_transformers import SentenceTransformer
import bitsandbytes as bnb


# -

def getDevice():
    print(f"PyTorch version: {torch.__version__}")

    device = None
    # Check if MPS device is available on MacOS
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        # If MPS is not available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f'using {device}')

    return device


def initModel(modelName, device, maxSeqLength):

    # Define the configuration for quantisation
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=getattr(torch, "float16"),
    )

    # Load the pre-trained model with quantisation configuration
    model = AutoModelForCausalLM.from_pretrained(
        modelName,
        device_map=device,
        quantization_config=bnb_config,
    )

    # Load the tokenizer with specified device and sequence length
    tokenizer = AutoTokenizer.from_pretrained(
        modelName,
        device_map=device,
        max_seq_length=maxSeqLength
    )
    
    # Return the initialized model and tokenizer
    return model, tokenizer


class GemmaHF():
    def __init__(self, modelName, maxSeqLength=2048):
        self.modelName = modelName
        if maxSeqLength:
            # Set the maximum sequence length for the model
            # This value is set to 2048 by default due to memory and computational constraints
            # Increase this value if you have sufficient resources and need to process longer sequences
            # The maximum value this variable can have is 8192.
            self.maxSeqLength = maxSeqLength
        
        # Initialize the model and tokenizer
        print("Initialising model:")
        self.device = getDevice()
        self.model, self.tokenizer = initModel(self.modelName, self.device, self.maxSeqLength)
    
    def generate(self, prompt, maxNewTokens=2048, temperature=0.0):    
        # Encode the prompt and convert to PyTorch tensor
        inputIDs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.device)

        # Calculate the available number of tokens for generation
        # Seems like this does not improve upon performance
        #inputLength = inputIDs.shape[1]
        #maxTokens = self.maxSeqLength - inputLength

        # Generate text based on the input prompt
        outputs = self.model.generate(**inputIDs, 
                                      max_new_tokens=maxNewTokens, 
                                      do_sample=False, 
                                      temperature=temperature
                                     )

        # Decode the generated output into text
        results = [self.tokenizer.decode(output) for output in outputs]

        # Return the list of generated text results
        return results


def getEmbedding(text, embeddingModel):  
    # Encode the text to obtain embeddings using the provided embedding model
    embedding = embeddingModel.encode(text, show_progress_bar=True)
    
    # Convert the embeddings to a list of floats and return
    return embedding.tolist()


def cleanText(txt, EOSTOKEN):
    txt = (txt
        .replace(EOSTOKEN, "")  # Replace the end-of-sentence token with an empty string
        .replace("**", "")  # Replace double asterisks with an empty string
        .replace("<pad>", "")  # Replace "<pad>" with an empty string
        .replace("<start>", "")  # Replace "<start>" with an empty string
        .replace("<end>", "")  # Replace "<end>" with an empty string
        .replace("<unk>", "")  # Replace "<unk>" with an empty string
        .replace("\n", " ")  # Replace newline characters with a space
        .replace("  ", " ")  # Replace double spaces with single spaces
        .replace(".,", ".")  # Remove comma after period
        .replace(",.", ".")  # Remove period after comma
    ).strip()  # Strip leading and trailing spaces from the text
    
    return txt


def generateAnswer(question, data, searcher, embeddingModel, model, maxNewTokens=2048):    
    # Embed the input question using the provided embedding model
    embededQuestion = np.array(getEmbedding(question, embeddingModel)).reshape(1, -1)
    
    # Find similar contexts in the dataset based on the embedded question
    neighbors, distances = searcher.search_batched(embededQuestion)
    
    # Extract context from the dataset based on the indices of similar contexts
    context = " ".join([data[pos] for pos in np.ravel(neighbors)])
    
    # Get the end-of-sentence token from the tokenizer
    EOSTOKEN = model.tokenizer.eos_token
    
    # Give a role to Gemma.
    role = 'A helpful assistant'
    
    # Generate a prompt for summarizing the context
    prompt = f"""
             Summarise this context: "{context}" in order to answer the question "{question}" as {role}
             SUMMARY:
             """.strip() + EOSTOKEN
    
    # Generate a summary based on the prompt
    results = model.generate(prompt, maxNewTokens)
    
    # Clean the generated summary
    summary = cleanText(results[0].split("SUMMARY:")[-1], EOSTOKEN)

    # Generate a prompt for providing an answer
    prompt = f"""
             Here is the context: {summary}
             Provide an answer as {role} to the question: {question}.
             ANSWER:
             """.strip() + EOSTOKEN

    # Generate an answer based on the prompt
    results = model.generate(prompt, maxNewTokens)
    
    # Clean the generated answer
    answer = cleanText(results[0].split("ANSWER:")[-1], EOSTOKEN)

    # Return the cleaned answer
    return answer


def map2Embeddings(data, embeddingModel):
    # Initialise an empty list to store embeddings
    embeddings = []

    # Iterate over each text in the input data list
    texts = len(data)
    print(f"Mapping {texts} pieces of information")
    for i in tqdm(range(texts)):
        # Get embeddings for the current text using the provided embedding model
        embeddings.append(getEmbedding(data[i], embeddingModel))
    
    # Return the list of embeddings
    return embeddings


# +

class AIAssistant():    
    def __init__(self, gemmaModel):
        # Initialise attributes
        self.embeddingsName = "thenlper/gte-large"
        self.knowledgeBase = []
        
        # Initialize Gemma model (it can be transformer-based or any other)
        self.gemmaModel = gemmaModel
        
        # Load the embedding model
        self.embeddingModel = SentenceTransformer(self.embeddingsName)
        
    def learnKnowledgeBase(self, knowledgeBase):
        # Storing the knowledge base
        self.knowledgeBase = knowledgeBase
        
        # Load and index the knowledge base
        embeddings = map2Embeddings(self.knowledgeBase, self.embeddingModel)
        self.embeddings = np.array(embeddings).astype(np.float32)
        
        # Instantiate the searcher for similarity search
        self.indexEmbeddings()
        
    def indexEmbeddings(self):
        ## Testing values:
        '''
        # Set the parameter values to test
        num_neighbors_values = [5, 10, 20, 50]
        num_leaves_values = [self.embeddings.shape[0] // 4, self.embeddings.shape[0] // 2, self.embeddings.shape[0] // 8, 500, 1000, 2000]
        num_leaves_to_search_values = [50, 100, 200, 500]
        training_sample_size_values = [self.embeddings.shape[0], self.embeddings.shape[0] // 2, min(self.embeddings.shape[0], 100000)]
        score_ah_configs = [
            (2, 0.2),
            (4, 0.4),
            (8, 0.6)
        ]
        reorder_values = [50, 100, 200]
        '''
        '''
        This makes the code stuck. So just search for best inputs.
        self.searcher = (scann.scann_ops_pybind.builder(db=self.embeddings, num_neighbors=10, distance_measure="dot_product")
                 .tree(num_leaves=min(self.embeddings.shape[0] // 2, 1000), 
                       num_leaves_to_search=100, 
                       training_sample_size=self.embeddings.shape[0])
                 .score_ah(2, anisotropic_quantization_threshold=0.2)
                 .reorder(100)
                 .build()
           )
        '''
        self.searcher = (scann.scann_ops_pybind.builder(db=self.embeddings, num_neighbors=5, distance_measure="dot_product")
                 .tree(num_leaves=min(self.embeddings.shape[0] // 8, 500),
                       num_leaves_to_search=50,
                       training_sample_size=min(self.embeddings.shape[0], 50000))
                 .score_ah(2, anisotropic_quantization_threshold=0.2)
                 .reorder(50)
                 .build()
                )
        
    def query(self, query):
        # Generate and print an answer to the query
        answer = generateAnswer(query, 
                                self.knowledgeBase, 
                                self.searcher, 
                                self.embeddingModel, 
                                self.gemmaModel)
        print(answer)
        
    # Use numpy save to .npy
    def saveEmbeddings(self, filename="embeddings.npy"):
        np.save(filename, self.embeddings)
    # Use numpy fast load, needs .npy     
    def loadEmbeddings(self, filename="embeddings.npy"):
        self.embeddings = np.load(filename)
        # Re-instantiate the searcher
        self.index_embeddings()


# -

def cleanString(input_string):
    # Remove extra spaces by splitting the string by spaces and joining back together
    output = ' '.join(input_string.split())
    
    # Remove consecutive carriage return characters until there are no more consecutive occurrences
    output = re.sub(r'\r+', '\r', output)
    
    # Maybe remove some Brackets? Hard to say can try.
    
    # Remove weird characters using regex
    output = re.sub(r'[^a-zA-Z0-9.,;:!?()\\[\\]{}\\-+=/\\s]', '', output)
    
    output = output.strip()
    
    # Return the cleaned string
    return output


def getWikiPagesSingle(wiki, categoryName):
    # Get the Wikipedia page corresponding to the provided category name
    category = wiki.page("Category:" + categoryName)
    
    # Initialize an empty list to store page titles
    pages = []
    
    # Check if the category exists
    if category.exists():
        # Iterate through each article in the category and append its title to the list
        for article in category.categorymembers.values():
            pages.append(article.title)
    
    # Return the list of page titles
    return pages


def getWikiPagesAll(categories):
    # Create a Wikipedia object
    wiki = wikipediaapi.Wikipedia('A helpful assisstant', 'en')
    
    # Initialise lists to store explored categories and Wikipedia pages
    exploredCategories = []
    wikipediaPages = []

    # Iterate through each category
    print("Wikipedia categories:")
    for categoryName in categories:
        print(f"{categoryName}:")
        
        # Get the Wikipedia page corresponding to the category
        category = wiki.page("Category:" + categoryName)
        
        # Extract Wikipedia pages from the category and extend the list
        wikipediaPages.extend(getWikiPagesSingle(wiki, categoryName))
        
        # Add the explored category to the list
        exploredCategories.append(categoryName)

    # Extract subcategories and remove duplicate categories
    categoriesToExplore = [item.replace("Category:", "") for item in wikipediaPages if "Category:" in item]
    wikipediaPages = list(set([item for item in wikipediaPages if "Category:" not in item]))
    
    '''
    
    Possibly need to make this recursive to read more sections? Hard to say but it would make sense to make it recursive. AI is very general no?
    
    '''
    extractedTexts = []
    
    # Iterate through each Wikipedia page
    for title in tqdm(wikipediaPages):
        # Get the Wikipedia page
        page = wiki.page(title)
        
        if len(page.summary) > len(page.title):
            extractedTexts.append(page.title + " : " + cleanString(page.summary))
            
        # Iterate through the sections in the page
        for section in page.sections:
            # Append the page title and section text to the extracted texts list
            if len(section.text) > len(page.title):
                extractedTexts.append(page.title + " : " + cleanString(section.text))
                    
    # Return the extracted texts
    return extractedTexts


categories = ["Machine learning"]
extractedWikiTexts = getWikiPagesAll(categories)
print("Found", len(extractedWikiTexts), "Wikipedia pages")

wikipediaData = pd.DataFrame(extractedWikiTexts, columns=["wikipedia_text"])
wikipediaData.to_csv("wikipediaData.csv", index=False)
wikipediaData.head()

# +
# Initialise model
modelName = "/kaggle/input/gemma/transformers/2b-it/2"

# Create an instance of AIAssistant with specified parameters
gemmaAIAssistant = AIAssistant(gemmaModel=GemmaHF(modelName))

# Map the intended knowledge base to embeddings and index it
gemmaAIAssistant.learnKnowledgeBase(knowledgeBase=extractedWikiTexts)

# Save the embeddings to disk (for later use)
gemmaAIAssistant.saveEmbeddings()
# -

gemmaAIAssistant.query("What is artificial intelligence?")


