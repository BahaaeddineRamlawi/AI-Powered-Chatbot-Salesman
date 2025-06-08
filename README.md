# AI-Powered-Chatbot-Salesman

This project aims to revolutionize customer engagement in e-commerce by developing an intelligent, AI-powered salesman chatbot. By leveraging Retrieval-Augmented Generation (RAG), LangChain, Weaviate, and Gradio, this chatbot dynamically retrieves product information, generates personalized responses, and provides tailored recommendations to users.

## Project Overview

The goal of this project is to build a chatbot that:

- Retrieves up-to-date product information dynamically from a structured knowledge base.
- Generates personalized and persuasive responses using generative AI models.
- Provides real-time product recommendations and offers based on user queries and preferences.
- Ensures seamless interaction through a user-friendly interface using Gradio.

## Key Technologies

- Retrieval-Augmented Generation (RAG): Combines traditional search-based retrieval with generative AI to provide accurate, context-aware responses.
- LangChain: Facilitates seamless integration of language models for processing user queries and generating responses.
- Weaviate: A vector database that stores product data and enables semantic search for accurate and fast information retrieval.
- Gradio: A Python library for building user-friendly interfaces, allowing easy interaction with the chatbot.

## Project Features

- Dynamic Product Retrieval: The chatbot fetches up-to-date product details from a knowledge base and provides relevant information on demand.
- Generative AI Responses: The chatbot leverages advanced AI models to generate persuasive, engaging responses that encourage user interaction.
- Personalized Recommendations: Based on user input, the chatbot recommends products, offers, and deals, making the shopping experience more personalized.
- Hybrid Search: Combines semantic vector search (via Weaviate) with metadata filtering to provide the most relevant products.
- Gradio Interface: The user interacts with the chatbot through a clean and intuitive interface created with Gradio.

## Project Workflow

- Data Collection: Product data is scraped from Rifai.com using Selenium. The data is stored in CSV or JSON format for testing and validation.
- Embedding Generation: Product descriptions are processed with Sentence Transformers to create vector embeddings for efficient semantic search.
- Weaviate Integration: The product metadata and embeddings are stored in Weaviate for dynamic and fast retrieval during chatbot interactions.
- Chatbot Development: The chatbot is developed using LangChain for handling language processing and Weaviate for information retrieval. It generates personalized responses based on user queries.
- User Interface: A Gradio interface is implemented to ensure seamless interaction with the chatbot. Users can ask queries and get dynamic, context-aware product recommendations.

## Installation Instructions

1. Clone the Repository

```bash
git clone https://github.com/BahaaeddineRamlawi/AI-Powered-Chatbot-Salesman
```

2. Set up a Virtual Environment

Create and activate a virtual environment:

```bash
python -m venv myenv
source myenv/bin/activate  # For Linux/macOS
myenv\Scripts\activate  # For Windows
```

3. Install Required Packages

Install the necessary dependencies using:

```bash
pip install -r requirements.txt
```

4. Weaviate Setup

Ensure that Weaviate is up and running. You can run it locally:

```bash
docker-compose up
```

## Usage

1. Embedding Generation & Data Insertion:

- Run vector_database_creation.py to generate embeddings for the products and insert them into Weaviate. The script will clean the data, generate embeddings using SentenceTransformer, and insert the data into the Weaviate vector database.

```bash
python vector_database_creation.py
```

2. Hybrid Search (via Gradio Interface):

- Once the data is in Weaviate, use query_interface.py to launch the Gradio interface for searching products using both semantic and metadata search. The interface runs locally and can be accessed via the URL: `http://127.0.0.1:7860`.

```bash
python query_interface.py
```

3. Fetching All Products from Weaviate:

- To fetch and view all product data stored in Weaviate, use get_all_items.py. This will display the product ID, title, and price for all products in the "Product" collection.

```bash
python get_all_items.py
```

## Code Breakdown

- `vector_database_creation.py`: Handles data preparation, embedding generation using SentenceTransformer, and inserting product data (along with embeddings) into Weaviate.

- `query_interface.py`: Implements the Gradio interface for users to input search queries, performs a hybrid search using both text-based queries and semantic vector search, and returns results with product details. The interface runs locally and can be accessed via the URL: `http://127.0.0.1:7860`.

- `get_all_items.py`: Fetches all product data from Weaviate and logs or prints the product information.

## Logging

Logs are stored in the ./logs directory. These logs capture important events, including:

- Embedding generation
- Data insertion into Weaviate
- Search operations
- Errors and warnings

## Conclusion

This project demonstrates the power of combining modern AI tools like LangChain, Weaviate, and Sentence Transformers with a user-friendly interface provided by Gradio. It is designed to enhance the e-commerce shopping experience through intelligent, personalized recommendations and efficient product search.
