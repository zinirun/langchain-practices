def get_documents_with_splitter(web_path: str):
    from langchain.document_loaders import WebBaseLoader
    from langchain.text_splitter import CharacterTextSplitter

    loader = WebBaseLoader(web_path=web_path)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    return text_splitter.split_documents(documents)
