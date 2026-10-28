"""AI Twin chat: grounded answers from the owner's ready global documents.

Rerun safety: creating a conversation and sending a message happen only from a button
callback or after ``st.chat_input`` returns a prompt. Ordinary reruns only GET history.
``pending_client_message_id`` makes a Streamlit rerun retry the same turn instead of
inserting a duplicate user message.
"""

from __future__ import annotations

import uuid

import streamlit as st

from components.state import flash, get_api_client, render_flashes
from services.api_client import ApiError, ChatMessage

client = get_api_client()
state = st.session_state
state.setdefault("conversation_id", None)
state.setdefault("pending_client_message_id", None)
state.setdefault("pending_content", None)
state.setdefault("last_workflow_debug", None)


def _on_new_conversation() -> None:
    try:
        conversation = client.create_conversation()
    except ApiError as exc:
        flash("error", f"Could not start a conversation: {exc.message}")
        return
    state.conversation_id = conversation.id
    state.conversation_picker = conversation.id
    state.pending_client_message_id = None
    state.pending_content = None


def _on_select_conversation() -> None:
    state.conversation_id = state.get("conversation_picker")
    state.pending_client_message_id = None
    state.pending_content = None


def _ensure_conversation() -> str:
    if state.conversation_id:
        return str(state.conversation_id)
    conversation = client.create_conversation()
    state.conversation_id = conversation.id
    state.conversation_picker = conversation.id
    return conversation.id


def _send_pending() -> None:
    content = state.pending_content
    client_message_id = state.pending_client_message_id
    if not content or not client_message_id:
        return
    try:
        conversation_id = _ensure_conversation()
        turn = client.send_message(conversation_id, content, client_message_id=client_message_id)
        state.last_workflow_debug = getattr(turn, "operational_metadata", None)
    except ApiError as exc:
        if exc.status_code == 502:
            flash(
                "error",
                f"{exc.message} Your question was saved; send it again to retry the answer.",
            )
        else:
            flash("error", f"Could not send the message: {exc.message}")
    finally:
        state.pending_client_message_id = None
        state.pending_content = None


def _render_message(message: ChatMessage) -> None:
    with st.chat_message(message.role):
        st.markdown(message.content)
        if message.role == "assistant" and message.citations:
            with st.expander(f"Sources ({len(message.citations)})"):
                for citation in message.citations:
                    label = citation.citation_id or "source"
                    title = citation.document_title or citation.original_filename
                    st.markdown(f"**[{label}]** {title} ({citation.page_label()})")
                    if citation.original_filename and citation.original_filename != title:
                        st.caption(citation.original_filename)
                    if citation.excerpt:
                        st.caption(citation.excerpt)


st.title("AI Twin")
st.caption(
    "Answers are grounded in **your ready global documents** only. Private documents stay "
    "in Documents / Playground and are never searched. Other owners are never searched."
)
render_flashes()

with st.sidebar:
    st.subheader("Conversations")
    st.button("New conversation", on_click=_on_new_conversation, width="stretch")
    try:
        page = client.list_conversations(limit=50)
    except ApiError as exc:
        st.error(f"Could not load conversations: {exc.message}")
        page = None
    if page is not None and page.items:
        options = [c.id for c in page.items]
        titles = {c.id: c.title for c in page.items}
        if state.conversation_id not in options:
            state.conversation_id = options[0]
        st.radio(
            "Conversation",
            options=options,
            index=options.index(state.conversation_id),
            format_func=lambda i: titles.get(i, "Untitled"),
            key="conversation_picker",
            on_change=_on_select_conversation,
            label_visibility="collapsed",
        )
    elif page is not None:
        st.caption("No conversations yet.")
    debug = state.get("last_workflow_debug")
    if debug:
        if debug.get("used_persistent_context"):
            st.caption("Using context from earlier conversations (summary / memories).")
        with st.expander("Workflow debug (last turn)"):
            st.json(debug)
    if state.conversation_id:
        with st.expander("Conversation summary"):
            try:
                summary = client.get_conversation_summary(str(state.conversation_id))
            except ApiError as exc:
                st.caption(f"Summary unavailable: {exc.message}")
            else:
                text = summary.get("summary_text")
                if text:
                    st.markdown(text)
                    st.caption(f"Version {summary.get('version', 0)}")
                else:
                    st.caption("No summary yet — summaries update after enough messages.")

messages: list[ChatMessage] = []
if state.conversation_id:
    try:
        messages = client.get_messages(state.conversation_id)
    except ApiError as exc:
        if exc.status_code == 404:
            state.conversation_id = None
            st.info("That conversation is no longer available.")
        else:
            st.error(f"Could not load messages: {exc.message}")
            st.stop()

if not messages and not state.pending_content:
    st.info(
        "Ask a question about your uploaded **global** documents. If nothing relevant is "
        "ready yet, the Twin will say so instead of guessing."
    )

for message in messages:
    _render_message(message)

if state.pending_content:
    with st.chat_message("user"):
        st.markdown(state.pending_content)

prompt = st.chat_input("Ask about your global documents")
if prompt and not state.pending_client_message_id:
    state.pending_content = prompt
    state.pending_client_message_id = str(uuid.uuid4())
    st.rerun()

if state.pending_content and state.pending_client_message_id:
    with st.spinner("Retrieving sources and generating an answer..."):
        _send_pending()
    st.rerun()
