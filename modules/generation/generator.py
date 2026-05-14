from dotenv import load_dotenv
from core.base_llm import BaseLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

class Generator:
    def __init__(self, llm: BaseLLM):
        self.llm = llm.get_model()
        self.template = """Answer the question based only on the following context:
        {context}

        Question: {question}
        """
        self.prompt = ChatPromptTemplate.from_template(self.template)

    def get_chain(self, retriever):
        """Creates a simple RAG chain."""
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        return chain
