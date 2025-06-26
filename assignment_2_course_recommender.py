from typing import List, Tuple
import chromadb
import pandas as pd
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI


OPENAI_API_VERSION = '2024-12-01-preview'
AZURE_DEPLOYMENT_NAME = "myllm"
EMBEDDING_MODEL_NAME = "text-embedding-3-small"


embeddings = AzureOpenAIEmbeddings(
    model=EMBEDDING_MODEL_NAME,
    openai_api_version=OPENAI_API_VERSION
)


client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name="courses",
    metadata={"hnsw:space": "cosine"}
)


if collection.count() == 0:
    df = pd.read_csv("course_catalog.csv")

    # Generate embeddings in batches
    batch_size = 50
    embeddings_list = []
    for i in range(0, len(df), batch_size):
        batch = df[i:i+batch_size]
        embeddings_list.extend(embeddings.embed_documents(batch['description'].tolist()))

    metadata = [{"title": title} for title in df['title']]

    collection.add(
        embeddings=embeddings_list,
        documents=df['description'].tolist(),
        metadatas=metadata,
        ids=df['course_id'].tolist()
    )
    print(f"Inserted {len(df)} courses")


def get_completed_descriptions(completed_ids: List[str]) -> List[str]:
    if not completed_ids:
        return []
    results = collection.get(
        ids=completed_ids,
        include=["documents"]
    )
    return results.get('documents', [])


def setup_rag_chain():
    llm = AzureChatOpenAI(
        azure_deployment=AZURE_DEPLOYMENT_NAME,
        openai_api_version=OPENAI_API_VERSION,
        temperature=0.1
    )

    # Initialize LangChain Chroma vector store
    vectorstore = Chroma(
        client=client,
        collection_name="courses",
        embedding_function=embeddings
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    # Define RAG prompt for profile enhancement
    prompt_template = """
    You are an academic advisor specializing in course recommendations. 
    Refine and enhance the student's profile based on their stated interests 
    and relevant course context. Focus on:
    - Clarifying ambiguous terms
    - Identifying key learning objectives
    - Suggesting related specialization areas
    - Outputting a concise, optimized profile description

    Student's Stated Interests: {profile}

    Relevant Course Context:
    {context}

    Enhanced Profile:
    """
    prompt = PromptTemplate.from_template(prompt_template)

    rag_chain = (
        {"context": retriever, "profile": RunnablePassthrough()}
        | prompt
        | llm
    )
    return rag_chain


def recommend_courses(profile: str, completed_ids: List[str], rag_chain) -> List[Tuple[str, float]]:
    # Enhance profile using RAG
    enhanced_profile = rag_chain.invoke(profile).content
    print(f"\nEnhanced profile: {enhanced_profile}")

    completed_descriptions = get_completed_descriptions(completed_ids)
    query_text = enhanced_profile + " " + " ".join(completed_descriptions)

    # Generate embedding using Azure
    query_embedding = embeddings.embed_query(query_text)

    # Validate completed IDs
    valid_completed_ids = [cid.strip() for cid in completed_ids if cid.strip()]
    print(f"Filtering out: {valid_completed_ids}")

    # Query ChromaDB with exclusion
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5 + len(valid_completed_ids),  # Get extra results
        where={"id": {"$nin": valid_completed_ids}}
    )

    top_ids = results['ids'][0]
    top_distances = results['distances'][0]
    
    # Combine and filter results
    recommendations = list(zip(top_ids, top_distances))
    filtered_recommendations = [
        (course_id, score) 
        for course_id, score in recommendations 
        if course_id not in valid_completed_ids
    ]
    
    # Debug output
    print(f"Raw recommendations: {recommendations}")
    print(f"Filtered recommendations: {filtered_recommendations[:5]}")
    
    return filtered_recommendations[:5]


def cli_interface():
    # Initialize RAG chain for profile enhancement
    rag_chain = setup_rag_chain()

    completed_input = input("Enter completed course IDs (comma-separated): ").strip()
    completed = [c.strip() for c in completed_input.split(',')] if completed_input else []
    raw_interests = input("Describe your interests: ").strip()

    recommendations = recommend_courses(raw_interests, completed, rag_chain)

    print("\nTop Recommendations:")
    for course_id, score in recommendations:
        course_info = collection.get(ids=[course_id], include=["metadatas"])
        title = course_info['metadatas'][0]['title'] if course_info['metadatas'] else "Unknown"
        similarity = 1 - score  # Convert distance to similarity
        print(f"{course_id}: {title} (Relevance: {similarity:.4f})")


if __name__ == "__main__":
    cli_interface()
