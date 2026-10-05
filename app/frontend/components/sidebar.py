"""
Sidebar UI: new-chat button, PDF upload, and the list of past conversations
(with pin / delete controls).
"""
import os
import tempfile

import streamlit as st

from app.backend import delete_thread, pin_thread, process_pdf


def inject_sidebar_css():
    """
    - Widens the sidebar's default width so controls have room without the user
      having to manually drag it wider first (still resizable).
    - Makes each thread-title button compact, left-aligned, and single-line with
      "..." truncation for long titles - closer to Claude/ChatGPT's sidebar tiles
      instead of Streamlit's default centered, wrapping button text.
    - Gives each thread row's "⋮" menu trigger a reliable minimum width so it always
      has room for its label plus Streamlit's own popover chevron.

    IMPORTANT: every rule below is scoped with an attribute selector on
    `class*="st-key-<prefix>"` (Streamlit turns a widget's `key=` into a
    `st-key-<key>` CSS class - see render_thread_list()), NOT by structural
    selectors like `:nth-child` on `[data-testid="stHorizontalBlock"]`. The sidebar
    has OTHER multi-column widgets in it (e.g. the PDF uploader's internal layout),
    and a structural rule meant for "the 2nd column of a thread row" also matches
    "the 2nd column of anything else in the sidebar", corrupting unrelated UI. Key
    scoping only ever touches the exact elements we render here.
    """
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {
            min-width: 340px;
            max-width: 480px;
        }

        /* Thread-title buttons (key="open_<thread_id>"): compact, left-aligned,
           single-line with ellipsis truncation for long titles. */
        [class*="st-key-open_"] button {
            justify-content: flex-start !important;
            padding-top: 0.35rem !important;
            padding-bottom: 0.35rem !important;
        }
        [class*="st-key-open_"] button p {
            text-align: left !important;
            white-space: nowrap !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
        }

        /* Per-thread "⋮" menu trigger (key="menu_<thread_id>"): just wide enough
           for its label + Streamlit's popover chevron, so it's never clipped down
           to a bare arrow. */
        [class*="st-key-menu_"] button {
            min-width: 2.75rem !important;
            padding-left: 0.5rem !important;
            padding-right: 0.5rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_new_chat_button(reset_chat):
    if st.sidebar.button("New Chat"):
        reset_chat()


def render_pdf_uploader():
    st.sidebar.header("Upload a PDF (optional)")
    uploaded_pdf = st.sidebar.file_uploader("Chat with a document", type=["pdf"])

    if uploaded_pdf is None:
        return

    # Unique per thread + file, so we don't reprocess on every Streamlit rerun
    file_key = f"{st.session_state['thread_id']}_{uploaded_pdf.name}_{uploaded_pdf.size}"

    if st.session_state.get("processed_pdf_key") == file_key:
        st.sidebar.info(f"'{uploaded_pdf.name}' is already indexed for this chat.")
        return

    with st.sidebar:
        with st.spinner("Reading and indexing PDF..."):
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_pdf.getbuffer())
                    tmp_path = tmp.name

                num_chunks = process_pdf(tmp_path, str(st.session_state["thread_id"]))
                st.session_state["processed_pdf_key"] = file_key
                st.success(f"'{uploaded_pdf.name}' indexed ({num_chunks} chunks). Ask away!")
            except Exception as e:
                st.error(f"Couldn't process this PDF: {e}")
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)


def render_thread_list(threads, switch_to_thread, reset_chat):
    """threads: list of {thread_id, title, pinned, updated_at}, already sorted
    pinned-first / most-recent by retrieve_all_threads()."""
    st.sidebar.header("My Conversations")

    for thread in threads:
        thread_id = thread["thread_id"]
        title = thread["title"] or "New Chat"
        is_current = str(thread_id) == str(st.session_state["thread_id"])
        label = f"{'📌 ' if thread['pinned'] else ''}{'▶ ' if is_current else ''}{title}"

        # Two columns: the title (flexes/truncates) and a single "⋮" menu, rather than
        # three tiny ones for title/pin/delete each.
        row = st.sidebar.columns([6, 1])
        if row[0].button(label, key=f"open_{thread_id}", use_container_width=True):
            switch_to_thread(thread_id)
            st.rerun()

        with row[1].popover("⋮", key=f"menu_{thread_id}", use_container_width=True):
            pin_label = "📌 Unpin" if thread["pinned"] else "📌 Pin"
            if st.button(pin_label, key=f"pin_{thread_id}", use_container_width=True):
                pin_thread(thread_id, not thread["pinned"])
                st.rerun()
            if st.button("🗑️ Delete", key=f"del_{thread_id}", use_container_width=True):
                delete_thread(thread_id)
                if is_current:
                    reset_chat()
                st.rerun()