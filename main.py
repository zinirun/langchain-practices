def load_api_key_from_env():
    from dotenv import load_dotenv
    load_dotenv()


if __name__ == '__main__':
    load_api_key_from_env()

    # msg = run_llm_davinci(message='Who are you?')
    # print('davinci: ' + msg)
    #
    # msg = run_llm_gpt35(message='What is gpt?')
    # print('gpt3.5: ' + msg)
    #
    # msg = run_llm_translate(text="I'm Groot.")
    # print('translated: ' + msg)
    #
    # msg = run_llm_with_agent(message='OpenAI의 창업자는 누구?', tool_names=['wikipedia', 'serpapi', 'llm-math'])
    # print('with-agent: ' + msg)
    #
    # mem_chain = get_llm_chain_with_memory(verbose=True)
    # msg = mem_chain.predict(input="My name is John Doe")
    # print('with-memory: ' + msg)
    # msg = mem_chain.predict(input="What is my name?")
    # print('with-memory: ' + msg)
    # print(mem_chain.memory)
    #
    # docs = get_documents_with_splitter("https://ko.wikipedia.org/wiki/%EC%98%A4%ED%94%88AI")
    # save_vector_store(documents=docs[1:4], save_path='faiss_openai')
    #
    # index = load_vector_store('faiss_openai')
    # index.query("OpenAI의 창업자는 누구?")