def save_vector_store(documents, save_path: str = 'faiss_db'):
    from langchain.embeddings import HuggingFaceEmbeddings
    # from langchain.indexes import VectorstoreIndexCreator
    from langchain.vectorstores import FAISS

    embeddings = HuggingFaceEmbeddings()

    # index = VectorstoreIndexCreator(
    #             vectorstore_cls=FAISS,
    #             embedding=embeddings,
    #         ).from_documents(documents=documents)

    db = FAISS.from_documents(documents=documents, embedding=embeddings)
    db.save_local(save_path)


def load_vector_store(path: str):
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import FAISS
    from langchain.indexes.vectorstore import VectorStoreIndexWrapper
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    embeddings = HuggingFaceEmbeddings()
    db = FAISS.load_local(path, embeddings)
    return VectorStoreIndexWrapper(vectorstore=db)
