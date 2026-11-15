
from langchain_openai import ChatOpenAI
import os

# Retrieve the top-k relevant chunks from the vector store
def retrieve(query: str, k: int = 2,vectorstore=None) -> list[str]:
    results = vectorstore.similarity_search(query, k=k)

    # Return each chunk with its content and metadata
    return [
        f"Content:\n{r.page_content}\n\nMetadata:\n{r.metadata}"
        for r in results
    ]

# Define the generation function that prompts the LLM with retrieved context
def generate(query, retrieved_chunks,model) -> str:

    prompt = f"""
    You are an AI assistant that helps AuroraStay Hotels front-desk and guest service staff
    answer guest questions using the hotel group's internal policy documents
    (booking terms and cancellation, room and property information, refunds and additional
    charges, and guest conduct, accessibility, safety, and resolution guidelines).

    User Query:
    {query}

    Retrieved Policy Information:
    {retrieved_chunks}

    Rules:
    1. Use only information explicitly supported by the retrieved policy information.
    2. Do not use outside knowledge, assumptions, or information not present in the retrieved policy information.
    3. Include all relevant information needed to answer the query completely. If the question has several parts
       (for example a fee AND a refund rule), answer every part.
    4. Do not omit important conditions, exceptions, approvals, or limits (for example "subject to availability",
       "manager approval", "applicable tax", "the reservation confirmation controls", pet number limits or stay caps).
    5. Focus only on information directly relevant to the user's query.
    6. Do not include unrelated information from the retrieved policy information.
    7. Answer the user's specific question directly in the first sentence, then give the supporting conditions.
    8. Keep the answer concise, clear, and easy for hotel staff to relay to a guest.
    9. Do not invent fees, amounts, times, property codes, amenities, refund outcomes, or exceptions. Quote amounts
       and times exactly as written and only for the property named in the question. Never transfer a fee from one
       property to another.
    10. If the information is insufficient (for example the property or amenity is not covered), say:
        "The available policy documents do not contain enough information to answer this question. Please check the
        live property profile or contact the property team."
        If the retrieved information conflicts, clearly mention the conflict.
    11. Where the documents say so, remind staff to verify the live property profile or reservation confirmation
        before quoting the guest, and do not promise refunds or exceptions that require approval.

    Answer:
    """

    return model.invoke(prompt)

# Define the full RAG pipeline combining retrieval and generation
def rag(
    query: str,
    k: int = 2,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 512,
    vectorstore=None
):
    # Create the generator model
    model = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        openai_api_base=os.getenv('OPENAI_API_BASE')
    )

    # Retrieve relevant chunks
    retrieved_chunks = retrieve(query=query, k=k,vectorstore=vectorstore)

    # Generate answer using retrieved chunks
    answer = generate(
        query=query,
        retrieved_chunks=retrieved_chunks,
        model=model
    )

    return answer.content, retrieved_chunks
