
from langchain_mistralai import ChatMistralAI  # Mistral AI

import gradio as gr
import logging
from src.utils import config

class GradioChatbotApp:
    def __init__(self, search_engine):
        self.search_engine = search_engine
        self.chatbot = gr.ChatInterface(
            self.stream_response,
            textbox=gr.Textbox(placeholder="Send a message...", container=False, autoscroll=True, scale=7),
        )
        self.llm = ChatMistralAI(api_key=config["mistral"]["api_key"], temperature=0.5, model=config["mistral"]["model"])
        
    
    # Function to handle user input
    def stream_response(self, message, history):

        # retrieved knowledge
        knowledge = "\n\n".join(self.search_engine.hybrid_search(message))

        # Construct the RAG prompt
        rag_prompt = f"""
        You are an salesman assistent which answers questions based on knowledge.
        While answering, you don't use your internal knowledge, 
        but solely the information in the "The knowledge" section.
        You don't mention anything to the user about the povided knowledge.

        The question: {message}
        Conversation history: {history}
        
        The knowledge: {knowledge}
        """

        # Stream response
        partial_message = ""
        for response in self.llm.stream(rag_prompt):
            partial_message += response.content
            yield partial_message
        
        
    def launch(self):
        try:
            self.chatbot.launch()
        finally:
            self.search_engine.close()