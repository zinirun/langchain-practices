from langchain import BasePromptTemplate
from langchain.base_language import BaseLanguageModel


def run_llm_davinci(message: str):
    from langchain.llms import OpenAI
    llm = OpenAI(model_name='text-davinci-003', temperature=0.9)
    return llm(message)


def run_llm_gpt35(message: str, system_message='You are expert robot to help human'):
    from langchain.chat_models import ChatOpenAI
    from langchain.schema import (
        AIMessage,
        HumanMessage,
        SystemMessage
    )

    chat = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.9)

    system_msg = SystemMessage(content=system_message)
    human_msg = HumanMessage(content=message)

    result = chat([system_msg, human_msg])
    return result.content


def run_llm_chain(llm: BaseLanguageModel, prompt: BasePromptTemplate, args):
    from langchain.chains import LLMChain
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(args)


def run_llm_translate(text: str, input_language: str = 'English', output_language: str = 'Korean'):
    from langchain.chains import LLMChain
    from langchain.chat_models import ChatOpenAI
    from langchain.prompts.chat import (
        ChatPromptTemplate,
        SystemMessagePromptTemplate,
        HumanMessagePromptTemplate
    )

    chat = ChatOpenAI(temperature=0)

    system_msg_prompt = SystemMessagePromptTemplate.from_template('You are a helpful assistant that translates {input_language} to {output_language}')
    human_msg_prompt = HumanMessagePromptTemplate.from_template('{text}')

    chat_prompt = ChatPromptTemplate.from_messages([system_msg_prompt, human_msg_prompt])

    chain = LLMChain(llm=chat, prompt=chat_prompt)
    return chain.run(input_language=input_language, output_language=output_language, text=text)


def run_llm_with_agent(message: str, tool_names=None, verbose: bool = True):
    if tool_names is None:
        tool_names = ['wikipedia', 'llm-math']
    from langchain.agents import (load_tools, initialize_agent, AgentType)
    from langchain.chat_models import ChatOpenAI

    chat = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.9)
    tools = load_tools(tool_names, llm=chat)

    agent = initialize_agent(tools=tools, llm=chat, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=verbose)
    return agent.run(message)


def get_llm_chain_with_memory(verbose: bool = True):
    from langchain import ConversationChain
    from langchain.chat_models import ChatOpenAI

    chat = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.9)
    chain = ConversationChain(llm=chat, verbose=verbose)
    return chain


def run_llm_summarization(documents, verbose: bool = True):
    from langchain.chains.summarize import load_summarize_chain
    from langchain.chat_models import ChatOpenAI

    chat = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.9)
    chain = load_summarize_chain(llm=chat, chain_type="map_reduce", verbose=verbose)
    return chain.run(documents)