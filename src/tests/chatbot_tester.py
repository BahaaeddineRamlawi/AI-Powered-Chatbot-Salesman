import csv
from src.llm import ChatbotHandler
from src.data_retriever import WeaviateHandler
import asyncio

from src.utils import logging, config

OUTPUT_CSV = config["test_data_file"]["output_path"]
INPUT_CSV = config["test_data_file"]["input_path"]


def test_chatbot_responses():
    # Initialize Weaviate handler and chatbot
    weaviate_handler = WeaviateHandler()  # You must define/init this according to your project
    chatbot = ChatbotHandler(weaviate_handler=weaviate_handler)

    # Load test queries from CSV
    with open(INPUT_CSV, mode='r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        queries = [row['Test Query'] for row in reader]

    # Open output CSV
    with open(OUTPUT_CSV, mode='w', newline='', encoding='utf-8') as outfile:
        fieldnames = ['query', 'response', 'response_intent', 'retrieval_time', 'response_time']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for query in queries:
            logging.info(f"Processing: {query}")
            # Use a fake history for test
            history = []
            response = ""

            try:
                generator = chatbot.stream_response(query=query, history=history, user_id=1)
                for chunk in generator:
                    response = chunk  # collect latest (final) chunk
            except Exception as e:
                response = f"Error: {str(e)}"
                logging.error(f"Error processing query '{query}': {e}")

            writer.writerow({'query': query, 'response': response, 'retrieval_time': chatbot.retrieval_time, 'response_time': chatbot.response_time})

    chatbot.shutdown()
    print(f"\n✅ Testing complete. Responses saved to {OUTPUT_CSV}")
    logging.info(f"✅ Testing complete. Responses saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    test_chatbot_responses()
