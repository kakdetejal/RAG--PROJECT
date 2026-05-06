from rag_chain import generate_answer

def main():

    """
    Simple command line interface for the RAG system.
    """

    while True:
        query = input("Ask a question: ")

        ## Exit condition
        if query.lower() in ['exit', 'quit']:
            break

        ## Generate the response using RAG pipeline
        answer = generate_answer(query)

        print("Answer:", answer)
        print()


if __name__ == "__main__":
    main()