from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from langchain_community.llms import LlamaCpp


# Define the Pydantic model for structured output
# class StartupInfo(BaseModel):
#     startup_id: int = Field(description="The unique identifier of the startup.")
#     startup_name: str = Field(description="The name of the startup.")
#     startup_url: str = Field(description="The URL of the startup's website.")
#     startup_industry: str = Field(description="The industry to which the startup belongs.")
#     startup_technology: str = Field(description="The technology used by the startup.")
#     startup_overview: str = Field(description="A brief overview of the startup.")
#     startup_description: str = Field(description="A detailed description of the startup's solutions.")
#     startup_usecases: str = Field(description="Use cases of the startup's solutions.")
#     startup_solutions: str = Field(description="The solutions offered by the startup.")
#     startup_analyst_rating: str = Field(description="The rating provided by analysts.")
#     startup_customers: str = Field(description="Customers or clients of the startup.")
#     startup_partners: str = Field(description="Partnerships of the startup.")
#     startup_gsi: str = Field(description="Global system integrator relationships.")
#     startup_country: str = Field(description="The country where the startup is based.")
#     startup_company_stage: str = Field(description="The stage of the startup company.")
#     startup_founders_info: str = Field(description="Information about the founders of the startup.")
#     startup_emails: str = Field(description="Contact emails of the startup.")


def setup_and_run_summarization():
    # Load pdf files
    loader = TextLoader("/home/lenovo/Desktop/summarization/Forma Vision.txt")
    data = loader.load()

    # Text splitting
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=400)
    text_chunks = text_splitter.split_documents(data)

    print("created text chuncks")

    # Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create Embeddings for each of the Text Chunk
    vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)
    print("converted into vector embeddings")

    # Import Model
    llm = LlamaCpp(
        streaming=True,
        model_path="/home/lenovo/Desktop/summarization/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        temperature=0.75,
        top_p=1,
        verbose=True,
        n_ctx=4096
    )

    retriever = vector_store.as_retriever()

    print("retriever is done")

    # Define examples for few-shot learning
    # examples = [
    #     {
    #         "input": "Summarize and give the necessary fields for Arviem",
    #         "output": {
    #             "startup_id": 32,
    #             "startup_name": "Arviem",
    #             "startup_url": "https://arviem.com/",
    #             "startup_industry": "Retail",
    #             "startup_technology": "IoT",
    #             "startup_overview": "Arviem provides integrated hardware & software solutions for real-time cargo monitoring.",
    #             "startup_description": "Arviem provides integrated hardware & software solutions for realtime cargo monitoring. The solution consists of a container plugin device with a customizable sensor suite along with its corresponding cloud-based management platform. This device monitors container conditions in real-time and the software uses the device/sensor data in combination with other external data sources to provide actionable insights. The solution enables users to monitor temperature & humidity fluctuations, real-time location, container intrusions & door openings, and detect shocks. The solution also provides real-time ETAs, route/shipment performance reports, track & trace functionalities, and carbon footprint reports.",
    #             "startup_usecases": "Arviem provides IoT-enabled real-time cargo monitoring and tracking to uncover inefficiencies in the flow of goods, finances, and information in supply chains. Their multimodal in-transit supply chain visibility allows customers to understand their extended supply chain better. Arviem's actionable insights help clients develop cost-saving strategies, optimize their supply chain, assess performance, and identify bottlenecks. They offer analytics dashboards to improve strategic decision-making and daily operations. Arviem guarantees a minimum of 150% ROI on their cargo monitoring and supply chain visibility services. They monitor cargo in real-time, collecting reliable data using automated locating and sensing technology installed on multimodal containers and cargo. Their devices provide real-time, carrier-independent data on the location and condition of cargo throughout the journey, including metrics such as temperature, humidity, shock, door opening, light intrusion, and GPS location. Arviem offers cargo monitoring as a fully managed service, including access to their data analytics platform and device logistics and maintenance. Their services enable significant working capital optimization and provide end-to-end transparency in the supply chain, allowing for reduced safety stocks, efficient allocation of work-in-process inventories, and access to supply chain finance services for goods in transit. Arviem serves beneficial cargo owners, exporters, importers, and manufacturers across industries such as tobacco, food and beverage, industrial manufacturing, chemicals, and cosmetics. Their solution has won awards and recognition in various industry forums.",
    #             "startup_solutions": "Arviem offers supply chain visibility and optimization through real-time cargo monitoring.",
    #             "startup_analyst_rating": "Gartner Peer Insights -4",
    #             "startup_customers": "Airtel, DBS, TransUnion, Snap Inc., Yahoo",
    #             "startup_partners": "",
    #             "startup_gsi": "",
    #             "startup_country": "Switzerland",
    #             "startup_company_stage": "Series B",
    #             "startup_founders_info": "Stefan Reidy (Founder & CEO), Pieter van de Mheen (Founder)",
    #             "startup_emails": "pmdehaan@arviem.com, reidy@arviem.com"
    #         }
    #     }
    # ]

    # Define the example prompt template
    # example_prompt = ChatPromptTemplate.from_messages(
    #     [
    #         ("human", "{input}"),
    #         ("ai", "{output}"),
    #     ]
    # )

    # Create the few-shot prompt template
    # few_shot_prompt = FewShotChatMessagePromptTemplate(
    #     example_prompt=example_prompt,
    #     examples=examples,
    # )

    # Define the custom RAG prompt template
    custom_rag_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """
            You are a Summarizer bot Expert tasked with summarizing information about startups.
            Your role is to provide concise summaries based on the {context} provided about a specific startup.
            Your response should summarize the details of the startup in a clear and concise manner, adhering to the prescribed format.
            You can consider the below examples to understand the format of your response"""),
            # few_shot_prompt,
            ("human", "{input}"),
        ]
    )
    print ("set into custom_rag_prompt")
    # Define the output parser
    # parser = PydanticOutputParser(pydantic_object=StartupInfo)

    # Define a function to format retrieved documents
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Define the RAG pipeline
    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
        | StrOutputParser()
    )

    # Invoke the RAG chain with a specific query
    return rag_chain.invoke("Give a summarized information on Neo Technology Of 10 words")

# Call the function
summary_result = setup_and_run_summarization()
print(summary_result)
