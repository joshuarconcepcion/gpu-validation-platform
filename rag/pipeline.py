"""LCEL RAG chain: retrieve similar historical GPU validation failures, format
them as grounding context, and ask Claude (via langchain_anthropic.ChatAnthropic)
for a diagnosis grounded in that context -- with per-session follow-up memory
and token-streaming support.
"""

from langchain_anthropic import ChatAnthropic
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory

from rag.retriever import has_relevant_matches

MODEL = "claude-opus-4-8"

NO_MATCH_FALLBACK = ( # message shown when no relevant context found
    "No sufficiently similar historical failures were found in the validation "
    "history for this query. The following is a general diagnosis based on the "
    "failure description alone, not grounded in past runs -- treat it as a "
    "starting point rather than a confirmed root cause.\n\n"
)

SYSTEM_PROMPT = ( # Claude instructions w/ context slot
    "You are a GPU validation diagnostic assistant for an RTX 3090 Ti "
    "validation platform. Base your diagnosis only on the historical failure "
    "context provided below -- do not invent past runs or metrics that are "
    "not present in the context. If the context does not contain anything "
    "relevant to the current failure, say so explicitly rather than "
    "guessing at a root cause.\n\n"
    "## Historical failure context\n{context}"
)

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])

# Per-session message history, so engineers can ask follow-up questions about
# a failure within a session. RunnableWithMessageHistory + an in-memory
# ChatMessageHistory is the current LCEL-native replacement for the older
# langchain.memory.ConversationBufferMemory class. State lives only in
# process memory and is lost on restart.
_session_histories: dict[str, BaseChatMessageHistory] = {}


def _get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Return (creating if needed) the message history for a session_id."""
    if session_id not in _session_histories: # returns message history if exists, 
        _session_histories[session_id] = InMemoryChatMessageHistory()
    return _session_histories[session_id]


def _format_context(documents: list[Document]) -> str:
    """Render retrieved failure Documents as plain-text grounding context for the prompt."""
    if not documents:
        return "(no similar historical failures found)"
    return "\n\n".join(
        f"- {doc.page_content} [run_id={doc.metadata.get('run_id')}, "
        f"severity={doc.metadata.get('severity')}]"
        for doc in documents
    )


def build_chain(retriever, model: str = MODEL) -> RunnableWithMessageHistory:
    """Assemble the LCEL (LangChain Expression Language) RAG chain: retrieve -> format context -> prompt -> Claude -> text."""
    llm = ChatAnthropic(model=model, max_tokens=1024) # creates claude model instance

    chain = (
        RunnablePassthrough.assign( # passes input through but adds additional key to dict
            context=RunnableLambda(lambda x: x["question"]) | retriever | RunnableLambda(_format_context)
            # RunnableLambda wraps python function so it works as a runnable in lanchang; extracts question string
            # question string goes into retriever, retriever searches ChromaDB for similar historical failures, returns list of LangChain documents
            # list of documents go through _format_context to convert into plain text string
            # string assigned to value of context
        ) 
        | PROMPT # prompt now has question and context values
        | llm # prompt goes to claude, response returned as AIMessage
        | StrOutputParser() # converts AIMessage to plain string
    )

    return RunnableWithMessageHistory( # wrapper adds conversation history to chain
        chain,
        _get_session_history, # passes function to be called internally by RunnableWithMessageHistory
        input_messages_key="question", # tells wrapper which key in input dict contains human question
        history_messages_key="history", # tells wrapper which placeholder in prompt to inject history into
    )


def query(question: str, retriever, session_id: str = "default") -> str:
    """Run the RAG chain once and return the full diagnostic response text.

    Prepends NO_MATCH_FALLBACK when no stored failure is similar enough to the
    query (rag.retriever.has_relevant_matches) -- the chain still runs and
    Claude still answers, just without grounding, so a cold vector store
    doesn't block diagnosis entirely.
    """
    chain = build_chain(retriever)
    grounded = has_relevant_matches(retriever.vectorstore, question)
    response = chain.invoke(
        {"question": question},
        config={"configurable": {"session_id": session_id}},
    )
    return response if grounded else NO_MATCH_FALLBACK + response


def stream_query(question: str, retriever, session_id: str = "default"):
    """Run the RAG chain and yield the diagnostic response token by token.

    Mirrors query()'s fallback behavior: if nothing relevant was found, the
    NO_MATCH_FALLBACK notice is yielded as the first chunk, before the
    (ungrounded) streamed answer.
    """
    chain = build_chain(retriever)
    if not has_relevant_matches(retriever.vectorstore, question):
        yield NO_MATCH_FALLBACK
    yield from chain.stream(
        {"question": question},
        config={"configurable": {"session_id": session_id}},
    )
