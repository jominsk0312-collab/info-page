{\rtf1\ansi\ansicpg949\cocoartf2865
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import streamlit as st\
import pandas as pd\
import time\
import uuid\
from google import genai\
from google.genai import types\
from google.genai.errors import APIError\
\
# --- \uc0\u49345 \u49688  \u48143  \u49884 \u49828 \u53596  \u49444 \u51221  ---\
APP_TITLE = "\uc0\u52828 \u51208 \u54620  \u51613 \u44428 \u49324  AI \u44256 \u44061  \u51025 \u45824  \u52311 \u48391 "\
# \uc0\u49464 \u49496  \u44256 \u50976  ID \u49373 \u49457  (\u54168 \u51060 \u51648  \u49352 \u47196 \u44256 \u52840  \u49884  \u50976 \u51648 )\
if 'session_id' not in st.session_state:\
    st.session_state.session_id = str(uuid.uuid4())\
\
# --- Gemini API \uc0\u53364 \u46972 \u51060 \u50616 \u53944  \u52488 \u44592 \u54868  ---\
def initialize_gemini_client():\
    """API \uc0\u53412 \u47484  \u49444 \u51221 \u54616 \u44256  Gemini \u53364 \u46972 \u51060 \u50616 \u53944 \u47484  \u52488 \u44592 \u54868 \u54633 \u45768 \u45796 ."""\
    \
    # === \uc0\u49324 \u50857 \u51088  \u50836 \u52397 \u50640  \u46384 \u46972  \u51076 \u49884 \u47196  \u49341 \u51077 \u46108  \u53412  (\u48372 \u50504  \u51452 \u51032 : \u49892 \u51228  \u48176 \u54252  \u49884  \u51060  \u51460  \u49325 \u51228  \u44428 \u51109 ) ===\
    TEMP_API_KEY = "AIzaSyCxoz_ay1Q2BunY2jFhlkN_COxiCAOHe20"\
    # =================================================================================\
\
    # 1. st.secrets\uc0\u50640 \u49436  \u53412 \u47484  \u49884 \u46020 \
    api_key = st.secrets.get('GEMINI_API_KEY')\
\
    # 2. st.secrets\uc0\u50640  \u53412 \u44032  \u50630 \u51004 \u47732 , \u49324 \u50857 \u51088 \u50640 \u44172  \u51077 \u47141  UI \u51228 \u44277 \
    if not api_key:\
        with st.sidebar:\
            st.warning("st.secrets\uc0\u50640  'GEMINI_API_KEY'\u44032  \u49444 \u51221 \u46104 \u50612  \u51080 \u51648  \u50506 \u49845 \u45768 \u45796 .")\
            \
            # \uc0\u49324 \u50857 \u51088  \u51077 \u47141  \u54596 \u46300 \u50640  \u51076 \u49884  \u53412 \u47484  \u44592 \u48376 \u44050 \u51004 \u47196  \u49444 \u51221 \
            api_key = st.text_input(\
                "Gemini API Key\uc0\u47484  \u51077 \u47141 \u54616 \u49464 \u50836 :", \
                type="password",\
                value=TEMP_API_KEY # \uc0\u49324 \u50857 \u51088  \u51228 \u44277  \u53412 \u47196  \u54532 \u47532 \u49483 \
            )\
\
    if api_key:\
        try:\
            # \uc0\u53364 \u46972 \u51060 \u50616 \u53944  \u52488 \u44592 \u54868 \
            client = genai.Client(api_key=api_key)\
            st.session_state.client = client\
            return True\
        except Exception as e:\
            # API \uc0\u53412 \u44032  \u51096 \u47803 \u46108  \u44221 \u50864  \u46321  \u50724 \u47448  \u52376 \u47532 \
            st.sidebar.error(f"Gemini \uc0\u53364 \u46972 \u51060 \u50616 \u53944  \u52488 \u44592 \u54868  \u49892 \u54056 : \{e\}")\
            return False\
    \
    st.session_state.client = None\
    return False\
\
# --- \uc0\u49884 \u49828 \u53596  \u54532 \u47212 \u54532 \u53944  \u51221 \u51032  ---\
SYSTEM_PROMPT = """\
\uc0\u45817 \u49888 \u51008  \u51613 \u44428 \u49324  \u50612 \u54540 \u47532 \u52992 \u51060 \u49496  \u51060 \u50857  \u44256 \u44061 \u51032  \u48520 \u54200  \u49324 \u54637 \u51012  \u51217 \u49688 \u54616 \u45716  \u52828 \u51208 \u54620  AI \u44256 \u44061  \u51025 \u45824  \u45812 \u45817 \u51088 \u51077 \u45768 \u45796 .\
\uc0\u44508 \u52825 \u51012  \u50628 \u44201 \u54616 \u44172  \u46384 \u47476 \u49464 \u50836 :\
1. \uc0\u49324 \u50857 \u51088 \u45716  \u51613 \u44428 \u49324  \u50612 \u54540  \u51060 \u50857  \u44284 \u51221 \u50640 \u49436  \u44202 \u51008  \u48520 \u54200 /\u48520 \u47564 \u51012  \u50616 \u44553 \u54633 \u45768 \u45796 . \u51221 \u51473 \u54616 \u44256  \u44618 \u51060  \u44277 \u44048 \u54616 \u45716  \u46384 \u46907 \u54620  \u47568 \u53804 \u47196  \u51025 \u45813 \u54616 \u49464 \u50836 .\
2. \uc0\u49324 \u50857 \u51088 \u51032  \u48520 \u54200  \u49324 \u54637 \u51012  \u44396 \u52404 \u51201 \u51004 \u47196  \u51221 \u47532 \u54616 \u50668 (\u47924 \u50631 \u51060 /\u50616 \u51228 /\u50612 \u46356 \u49436 /\u50612 \u46523 \u44172 ) \u49688 \u51665 \u54616 \u44256 , "\u51060  \u45236 \u50857 \u51012  \u44256 \u44061  \u51025 \u45824  \u45812 \u45817 \u51088 \u50640 \u44172  \u51204 \u45804 \u54616 \u50668  \u49888 \u49549 \u54616 \u44172  \u44160 \u53664 \u54616 \u46020 \u47197  \u54616 \u44192 \u45796 "\u45716  \u52712 \u51648 \u47196  \u50504 \u45236 \u54616 \u49464 \u50836 .\
3. \uc0\u47560 \u51648 \u47561  \u51025 \u45813  \u49884 , \u45812 \u45817 \u51088  \u54869 \u51064  \u54980  \u54924 \u49888 \u51012  \u50948 \u54644  '\u51060 \u47700 \u51068  \u51452 \u49548 '\u47484  \u50836 \u52397 \u54616 \u49464 \u50836 .\
4. \uc0\u47564 \u51068  \u49324 \u50857 \u51088 \u44032  \u50672 \u46973 \u52376  \u51228 \u44277 \u51012  \u50896 \u52824  \u50506 \u44144 \u45208  \u44144 \u51208 \u54616 \u47732 , \u45796 \u51020  \u47928 \u44396 \u47564  \u49324 \u50857 \u54616 \u50668  \u51221 \u51473 \u55176  \u50504 \u45236 \u54633 \u45768 \u45796 : "\u51396 \u49569 \u54616 \u51648 \u47564 , \u50672 \u46973 \u52376  \u51221 \u48372 \u47484  \u48155 \u51648  \u47803 \u54616 \u50668  \u45812 \u45817 \u51088 \u51032  \u44160 \u53664  \u45236 \u50857 \u51012  \u48155 \u51004 \u49892  \u49688  \u50630 \u50612 \u50836 . \u45824 \u49888 , \u47928 \u51032  \u45236 \u50857 \u51012  \u48148 \u53461 \u51004 \u47196  \u49436 \u48708 \u49828  \u44060 \u49440 \u50640  \u52572 \u49440 \u51012  \u45796 \u54616 \u44192 \u49845 \u45768 \u45796 . \u51060 \u50857 \u50640  \u48520 \u54200 \u51012  \u46300 \u47140  \u45796 \u49884  \u54620 \u48264  \u49324 \u44284 \u46300 \u47549 \u45768 \u45796 ."\
5. \uc0\u47784 \u46304  \u45824 \u54868 \u50640 \u49436  \u44256 \u44061 \u51032  \u44048 \u51221 \u51012  \u54644 \u52824 \u51648  \u50506 \u46020 \u47197  \u49464 \u49900 \u54616 \u44172  \u51452 \u51032 \u54616 \u49901 \u49884 \u50724 .\
"""\
\
# --- \uc0\u49464 \u49496  \u49345 \u53468  \u52488 \u44592 \u54868  ---\
def initialize_session_state():\
    """Streamlit \uc0\u49464 \u49496  \u49345 \u53468 \u47484  \u52488 \u44592 \u54868 \u54633 \u45768 \u45796 ."""\
    # \uc0\u47784 \u45944  \u47785 \u47197  (gemini-exp \u51228 \u50808 )\
    AVAILABLE_MODELS = [\
        "gemini-2.0-flash", \
        "gemini-2.0-pro",\
        "gemini-2.5-flash",\
        "gemini-2.5-pro",\
    ]\
    \
    if 'model_name' not in st.session_state:\
        st.session_state.model_name = "gemini-2.0-flash"\
    if 'available_models' not in st.session_state:\
        st.session_state.available_models = AVAILABLE_MODELS\
    if 'messages' not in st.session_state:\
        st.session_state.messages = []\
    if 'log_history' not in st.session_state:\
        st.session_state.log_history = []\
    if 'log_enabled' not in st.session_state:\
        st.session_state.log_enabled = True\
    if 'client' not in st.session_state:\
        st.session_state.client = None\
\
def log_message(role, content):\
    """\uc0\u45824 \u54868  \u47700 \u49884 \u51648 \u47484  \u49464 \u49496  \u49345 \u53468 \u50752  \u47196 \u44613  \u55176 \u49828 \u53664 \u47532 \u50640  \u52628 \u44032 \u54633 \u45768 \u45796 ."""\
    message = \{'role': role, 'content': content\}\
    st.session_state.messages.append(message)\
    \
    # CSV \uc0\u47196 \u44613  \u45936 \u51060 \u53552  \u54252 \u47607 \
    if st.session_state.log_enabled:\
        log_data = \{\
            'session_id': st.session_state.session_id,\
            'timestamp': pd.to_datetime('now', utc=True).isoformat(),\
            'role': role,\
            'content': content,\
            'model': st.session_state.model_name\
        \}\
        st.session_state.log_history.append(log_data)\
\
# --- API \uc0\u54840 \u52636  \u48143  \u51116 \u49884 \u46020  \u47196 \u51649  ---\
def generate_response_with_retry(user_prompt):\
    """\
    API \uc0\u54840 \u52636 \u51012  \u49688 \u54665 \u54616 \u44256  429 \u50724 \u47448  \u48156 \u49373  \u49884  \u51116 \u49884 \u46020  \u47196 \u51649 \u51012  \u54252 \u54632 \u54633 \u45768 \u45796 .\
    \uc0\u45824 \u54868  \u55176 \u49828 \u53664 \u47532 \u47484  6\u53556 (3\u49933 )\u51004 \u47196  \u51228 \u54620 \u54616 \u50668  \u52968 \u53581 \u49828 \u53944 \u47484  \u50976 \u51648 \u54616 \u44256  \u51116 \u49884 \u51089 \u54633 \u45768 \u45796 .\
    """\
    \
    # \uc0\u51204 \u52404  \u45824 \u54868  \u44592 \u47197 \u50640 \u49436  LLM\u50640  \u51204 \u45804 \u54624  \u52968 \u53581 \u49828 \u53944  \u44396 \u49457 \
    history_for_llm = []\
    for msg in st.session_state.messages:\
        # Gemini API\uc0\u45716  'user'\u50752  'model' \u50669 \u54624 \u51012  \u49324 \u50857 \u54633 \u45768 \u45796 .\
        role_map = \{'user': 'user', 'assistant': 'model'\}\
        if msg['role'] in role_map:\
            content_part = types.Part.from_text(msg['content'])\
            history_for_llm.append(types.Content(\
                role=role_map[msg['role']],\
                parts=[content_part]\
            ))\
\
    # \uc0\u52968 \u53581 \u49828 \u53944  \u50952 \u46020 \u50864  \u44288 \u47532 \u47484  \u50948 \u54644  LLM\u50640  \u51204 \u45804 \u54624  history\u47484  \u52572 \u44540  6\u44060  \u47700 \u49884 \u51648 \u47196  \u51228 \u54620 \
    # (\uc0\u49324 \u50857 \u51088  \u47700 \u49884 \u51648  3\u44060 , \u47784 \u45944  \u47700 \u49884 \u51648  3\u44060  \u49933 \u51012  \u50976 \u51648 )\
    recent_history = history_for_llm[-6:]\
    \
    # \uc0\u54788 \u51116  \u49324 \u50857 \u51088  \u54532 \u47212 \u54532 \u53944  \u52628 \u44032 \
    user_content = types.Content(role='user', parts=[types.Part.from_text(user_prompt)])\
    \
    # LLM\uc0\u50640  \u51204 \u45804 \u54624  \u52572 \u51333  \u52968 \u53584 \u52768 : \u52572 \u44540  \u44592 \u47197  + \u54788 \u51116  \u49324 \u50857 \u51088  \u51077 \u47141 \
    contents_for_llm = recent_history + [user_content]\
    \
    max_retries = 3\
    for attempt in range(max_retries):\
        try:\
            with st.spinner("AI \uc0\u52311 \u48391 \u51060  \u45813 \u48320 \u51012  \u49373 \u49457  \u51473 \u51077 \u45768 \u45796 ..."):\
                config = types.GenerateContentConfig(\
                    system_instruction=SYSTEM_PROMPT\
                )\
                \
                response = st.session_state.client.models.generate_content(\
                    model=st.session_state.model_name,\
                    contents=contents_for_llm,\
                    config=config\
                )\
            \
            return response.text\
\
        except APIError as e:\
            if e.response.status_code == 429 and attempt < max_retries - 1:\
                # 429 \uc0\u50724 \u47448  (Rate Limit Exceeded) \u52376 \u47532 \
                wait_time = 2 ** attempt  # \uc0\u51648 \u49688  \u48177 \u50724 \u54532 : 1, 2, 4\u52488 \
                st.warning(f"\uc0\u50836 \u52397  \u51228 \u54620 (429) \u48156 \u49373 ! \{wait_time\}\u52488  \u54980  \u51116 \u49884 \u46020 \u54633 \u45768 \u45796 . (\u49884 \u46020  \{attempt + 1\}/\{max_retries\})")\
                time.sleep(wait_time)\
            else:\
                st.error(f"API \uc0\u54840 \u52636  \u51473  \u50724 \u47448  \u48156 \u49373  (Code: \{e.response.status_code\}): \{e\}")\
                st.info("\uc0\u45824 \u54868 \u47484  \u52488 \u44592 \u54868 \u54616 \u44144 \u45208  \u47784 \u45944 \u51012  \u48320 \u44221 \u54644  \u48372 \u49464 \u50836 .")\
                return "\uc0\u51396 \u49569 \u54633 \u45768 \u45796 . API \u54840 \u52636  \u51473  \u50724 \u47448 \u44032  \u48156 \u49373 \u54664 \u49845 \u45768 \u45796 ."\
        except Exception as e:\
            st.error(f"\uc0\u50696 \u49345 \u52824  \u47803 \u54620  \u50724 \u47448  \u48156 \u49373 : \{e\}")\
            return "\uc0\u51396 \u49569 \u54633 \u45768 \u45796 . \u50696 \u49345 \u52824  \u47803 \u54620  \u50724 \u47448 \u44032  \u48156 \u49373 \u54664 \u49845 \u45768 \u45796 ."\
    \
    st.error("\uc0\u52572 \u45824  \u51116 \u49884 \u46020  \u54943 \u49688 \u47484  \u52488 \u44284 \u54664 \u49845 \u45768 \u45796 . \u51104 \u49884  \u54980  \u45796 \u49884  \u49884 \u46020 \u54644  \u51452 \u49464 \u50836 .")\
    return "\uc0\u51396 \u49569 \u54633 \u45768 \u45796 . \u49436 \u48708 \u49828  \u51217 \u49549 \u51060  \u50896 \u54876 \u54616 \u51648  \u50506 \u49845 \u45768 \u45796 ."\
\
# --- UI \uc0\u54632 \u49688  ---\
def handle_reset():\
    """\uc0\u45824 \u54868  \u49464 \u49496 \u51012  \u52488 \u44592 \u54868 \u54633 \u45768 \u45796 ."""\
    st.session_state.messages = []\
    st.session_state.log_history = []\
    st.session_state.session_id = str(uuid.uuid4())\
    st.rerun()\
\
def get_csv_download_link():\
    """\uc0\u47196 \u44613 \u46108  \u45936 \u51060 \u53552 \u47484  CSV \u54028 \u51068 \u47196  \u45796 \u50868 \u47196 \u46300 \u54624  \u49688  \u51080 \u45716  \u47553 \u53356 \u47484  \u49373 \u49457 \u54633 \u45768 \u45796 ."""\
    if not st.session_state.log_history:\
        return None\
\
    df = pd.DataFrame(st.session_state.log_history)\
    # \uc0\u54620 \u44544  \u44648 \u51664  \u48169 \u51648 \u47484  \u50948 \u54644  'utf-8-sig' \u51064 \u53076 \u46377  \u49324 \u50857 \
    csv = df.to_csv(index=False, encoding='utf-8-sig')\
    \
    return csv\
\
# --- \uc0\u47700 \u51064  \u50545  \u47196 \u51649  ---\
def main():\
    st.set_page_config(page_title=APP_TITLE, layout="wide")\
    initialize_session_state()\
    \
    # --- Sidebar ---\
    with st.sidebar:\
        st.title(APP_TITLE)\
        \
        # \uc0\u47784 \u45944  \u49440 \u53469 \
        selected_model = st.selectbox(\
            "\uc0\u49324 \u50857 \u54624  Gemini \u47784 \u45944 \u51012  \u49440 \u53469 \u54616 \u49464 \u50836 :",\
            options=st.session_state.available_models,\
            index=st.session_state.available_models.index(st.session_state.model_name)\
        )\
        if selected_model != st.session_state.model_name:\
            st.session_state.model_name = selected_model\
            st.toast(f"\uc0\u47784 \u45944 \u51060  \{selected_model\}\u47196  \u48320 \u44221 \u46104 \u50632 \u49845 \u45768 \u45796 . \u45824 \u54868 \u47484  \u52488 \u44592 \u54868 \u54644 \u51452 \u49464 \u50836 .")\
\
        # API \uc0\u53364 \u46972 \u51060 \u50616 \u53944  \u52488 \u44592 \u54868  \u48143  \u50672 \u44208  \u49345 \u53468  \u54364 \u49884 \
        api_connected = initialize_gemini_client()\
        if not api_connected:\
            return # API \uc0\u53412  \u50630 \u51004 \u47732  \u50545  \u49892 \u54665  \u51473 \u45800 \
\
        st.success("Gemini API \uc0\u50672 \u44208 \u46120 ")\
\
        st.markdown("---")\
        st.subheader("\uc0\u45824 \u54868  \u48143  \u47196 \u44613  \u50741 \u49496 ")\
        \
        # \uc0\u47196 \u44613  \u50741 \u49496 \
        st.session_state.log_enabled = st.checkbox(\
            "CSV \uc0\u51088 \u46041  \u47196 \u44613  \u54876 \u49457 \u54868 ",\
            value=st.session_state.log_enabled,\
            help="\uc0\u54876 \u49457 \u54868  \u49884 , \u47784 \u46304  \u45824 \u54868  \u53556 \u51060  \u45796 \u50868 \u47196 \u46300  \u44032 \u45733 \u54620  \u47196 \u44536 \u50640  \u44592 \u47197 \u46121 \u45768 \u45796 ."\
        )\
\
        # \uc0\u45824 \u54868  \u52488 \u44592 \u54868  \u48260 \u53948 \
        st.button("\uc0\u55357 \u56580  \u45824 \u54868  \u52488 \u44592 \u54868  \u48143  \u49352  \u49464 \u49496  \u49884 \u51089 ", on_click=handle_reset, type="primary")\
\
        # \uc0\u47196 \u44536  \u45796 \u50868 \u47196 \u46300  \u48260 \u53948 \
        csv_data = get_csv_download_link()\
        if csv_data:\
            st.download_button(\
                label="\uc0\u11015 \u65039  \u45824 \u54868  \u47196 \u44536  (CSV) \u45796 \u50868 \u47196 \u46300 ",\
                data=csv_data,\
                file_name=f"customer_chat_log_\{st.session_state.session_id\}.csv",\
                mime="text/csv"\
            )\
        \
        st.markdown("---")\
        st.info(f"""\
        **\uc0\u49464 \u49496  \u51221 \u48372 **\
        - \uc0\u47784 \u45944 : `\{st.session_state.model_name\}`\
        - \uc0\u49464 \u49496  ID: `\{st.session_state.session_id[:8]\}...`\
        - \uc0\u47196 \u44613 \u46108  \u45824 \u54868  \u53556  \u49688 : \{len(st.session_state.log_history)\}\
        """)\
\
    # --- Main Chat Interface ---\
    st.title(f"\uc0\u10024  \{APP_TITLE\}")\
    st.markdown(f"**AI \uc0\u45812 \u45817 \u51088 :** \u50504 \u45397 \u54616 \u49464 \u50836 , \u51200 \u45716  \u51613 \u44428 \u49324  AI \u44256 \u44061  \u51025 \u45824  \u45812 \u45817 \u51088 \u51077 \u45768 \u45796 . \u51200 \u55148  \u50612 \u54540  \u51060 \u50857  \u51473  \u44202 \u51004 \u49888  \u48520 \u54200  \u49324 \u54637 \u51012  \u50508 \u47140 \u51452 \u49884 \u47732 , \u44277 \u44048 \u44284  \u52293 \u51076 \u44048 \u51012  \u44032 \u51648 \u44256  \u54644 \u44208 \u51012  \u50948 \u54644  \u45432 \u47141 \u54616 \u44192 \u49845 \u45768 \u45796 .")\
    st.markdown("---")\
\
    # \uc0\u45824 \u54868  \u55176 \u49828 \u53664 \u47532  \u54364 \u49884 \
    for message in st.session_state.messages:\
        with st.chat_message(message["role"]):\
            st.markdown(message["content"])\
\
    # \uc0\u49324 \u50857 \u51088  \u51077 \u47141  \u52376 \u47532 \
    if user_prompt := st.chat_input("\uc0\u48520 \u54200 \u54616 \u49888  \u45236 \u50857 \u51012  \u47568 \u50432 \u54644  \u51452 \u49464 \u50836 ."):\
        # 1. \uc0\u49324 \u50857 \u51088  \u47700 \u49884 \u51648 \u47484  \u44592 \u47197  \u48143  \u54364 \u49884 \
        log_message("user", user_prompt)\
        with st.chat_message("user"):\
            st.markdown(user_prompt)\
\
        # 2. AI \uc0\u51025 \u45813  \u49373 \u49457  \u48143  \u52376 \u47532 \
        assistant_response = generate_response_with_retry(user_prompt)\
\
        # 3. AI \uc0\u51025 \u45813  \u44592 \u47197  \u48143  \u54364 \u49884 \
        log_message("assistant", assistant_response)\
        with st.chat_message("assistant"):\
            st.markdown(assistant_response)\
\
if __name__ == "__main__":\
    main()\
}