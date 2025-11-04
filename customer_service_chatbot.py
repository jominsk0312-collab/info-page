{\rtf1\ansi\ansicpg949\cocoartf2865
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import streamlit as st\
import time\
import pandas as pd\
import io\
import os\
from google import genai\
from google.genai.errors import ResourceExhaustedError, APIError\
\
# --- \uc0\u49345 \u49688  \u48143  \u49444 \u51221  ---\
\
# \uc0\u50545  \u51228 \u47785  \u49444 \u51221 \
st.set_page_config(page_title="Gemini \uc0\u44256 \u44061  \u51025 \u45824  \u52311 \u48391 ", layout="wide")\
st.title("\uc0\u55357 \u57057 \u65039  \u51613 \u44428 \u49324  \u50545  \u44256 \u44061  \u48520 \u54200  \u49324 \u54637  \u51217 \u49688  \u52311 \u48391 ")\
\
# \uc0\u49324 \u50857  \u44032 \u45733 \u54620  \u47784 \u45944  \u47785 \u47197  (gemini-2.0-flash\u47484  \u44592 \u48376 \u51004 \u47196  \u51228 \u44277 )\
AVAILABLE_MODELS = ["gemini-2.0-flash", "gemini-2.0-pro"]\
\
# \uc0\u49884 \u49828 \u53596  \u54532 \u47212 \u54532 \u53944  \u51221 \u51032 \
SYSTEM_INSTRUCTION = """\
\uc0\u45817 \u49888 \u51008  \u51613 \u44428 \u49324  \u50612 \u54540 \u47532 \u52992 \u51060 \u49496  \u51060 \u50857  \u44256 \u44061 \u51032  \u48520 \u54200  \u49324 \u54637 \u51012  \u51217 \u49688 \u54616 \u44256  \u54644 \u44208 \u51012  \u46037 \u45716  '\u44256 \u44061  \u51025 \u45824  \u52311 \u48391 '\u51077 \u45768 \u45796 . \u45796 \u51020  \u51648 \u52840 \u51012  \u50628 \u44201 \u54616 \u44172  \u46384 \u47476 \u49464 \u50836 :\
\
1.  **\uc0\u53668  \u50532  \u47588 \u45320 :** \u49324 \u50857 \u51088 \u45716  \u51613 \u44428 \u49324  \u50612 \u54540  \u51060 \u50857  \u44284 \u51221 \u50640 \u49436  \u44202 \u51008  \u48520 \u54200 /\u48520 \u47564 \u51012  \u50616 \u44553 \u54616 \u44256  \u51080 \u49845 \u45768 \u45796 . \u44256 \u44061 \u51032  \u44048 \u51221 \u50640  \u51221 \u51473 \u54616 \u44256  \u44277 \u44048  \u50612 \u47536  \u47568 \u53804 (\u52828 \u51208 \u54616 \u44256  \u46384 \u46907 \u54620  \u50612 \u53804 )\u47196  \u51025 \u45813 \u54616 \u47728 , \u48520 \u54200 \u51012  \u44202 \u44172  \u54644 \u46300 \u47536  \u51216 \u50640  \u45824 \u54644  \u44618 \u51060  \u49324 \u44284 \u54633 \u45768 \u45796 .\
2.  **\uc0\u51221 \u48372  \u49688 \u51665 :** \u49324 \u50857 \u51088 \u51032  \u48520 \u54200  \u49324 \u54637 \u51012  \u44396 \u52404 \u51201 \u51004 \u47196  \u51221 \u47532 \u54616 \u50668  (\u47924 \u50631 \u51012  \u48520 \u54200 \u54644 \u54616 \u49512 \u45716 \u51648 /\u50616 \u51228  \u48156 \u49373 \u54664 \u45716 \u51648 /\u50612 \u54540 \u51032  \u50612 \u46500  \u48512 \u48516 \u50640 \u49436  \u48156 \u49373 \u54664 \u45716 \u51648 /\u50612 \u46523 \u44172  \u49884 \u46020 \u54616 \u49512 \u45716 \u51648 ) \u49688 \u51665 \u54616 \u44256 , \u51060 \u47484  \u44256 \u44061  \u51025 \u45824  \u45812 \u45817 \u51088 \u50640 \u44172  \u51204 \u45804 \u54620 \u45796 \u45716  \u52712 \u51648 \u47196  \u47749 \u54869 \u55176  \u50504 \u45236 \u54633 \u45768 \u45796 .\
3.  **\uc0\u50672 \u46973 \u52376  \u50836 \u52397 :** \u47560 \u51648 \u47561 \u50640 \u45716  \u45812 \u45817 \u51088  \u54869 \u51064  \u54980  \u54924 \u49888 \u51012  \u50948 \u54644  \u48152 \u46300 \u49884  \u51060 \u47700 \u51068  \u51452 \u49548 \u47484  \u50836 \u52397 \u54616 \u49464 \u50836 .\
    - \uc0\u47564 \u51068  \u49324 \u50857 \u51088 \u44032  \u50672 \u46973  \u51228 \u44277 \u51012  \u50896 \u52824  \u50506 \u44144 \u45208  \u44144 \u48512 \u54616 \u47732 : "\u51396 \u49569 \u54616 \u51648 \u47564 , \u44256 \u44061 \u45784 \u51032  \u49548 \u51473 \u54620  \u50672 \u46973 \u52376  \u51221 \u48372 \u47484  \u48155 \u51648  \u47803 \u54616 \u50668  \u45812 \u45817 \u51088 \u51032  \u44160 \u53664  \u45236 \u50857 \u51012  \u44256 \u44061 \u45784 \u44760  \u51649 \u51217  \u54924 \u49888 \u54644  \u46300 \u47540  \u49688  \u50630 \u50612 \u50836 . \u45824 \u49888 , \u47928 \u51032  \u45236 \u50857 \u51012  \u45236 \u48512 \u51201 \u51004 \u47196  \u52384 \u51200 \u55176  \u44160 \u53664 \u54616 \u50668  \u49436 \u48708 \u49828  \u44060 \u49440 \u50640  \u54876 \u50857 \u54616 \u44192 \u49845 \u45768 \u45796 ."\u46972 \u44256  \u51221 \u51473 \u55176  \u50504 \u45236 \u54616 \u47728  \u45824 \u54868 \u47484  \u51333 \u47308 \u54633 \u45768 \u45796 .\
"""\
\
# --- \uc0\u49345 \u53468  \u52488 \u44592 \u54868  \u54632 \u49688  ---\
\
def initialize_session_state():\
    """Streamlit \uc0\u49464 \u49496  \u49345 \u53468  \u48320 \u49688 \u47484  \u52488 \u44592 \u54868 \u54633 \u45768 \u45796 ."""\
    if 'messages' not in st.session_state:\
        st.session_state['messages'] = []\
    if 'model_name' not in st.session_state:\
        st.session_state['model_name'] = AVAILABLE_MODELS[0]\
    if 'gemini_client' not in st.session_state:\
        st.session_state['gemini_client'] = None\
    if 'chat_session' not in st.session_state:\
        st.session_state['chat_session'] = None\
    if 'log_history' not in st.session_state:\
        st.session_state['log_history'] = []\
\
def reset_conversation():\
    """\uc0\u45824 \u54868  \u45236 \u50857 \u51012  \u52488 \u44592 \u54868 \u54616 \u44256  Gemini Chat \u49464 \u49496 \u51012  \u51116 \u49444 \u51221 \u54633 \u45768 \u45796 ."""\
    st.session_state['messages'] = []\
    st.session_state['log_history'] = []\
    # Chat \uc0\u49464 \u49496 \u47564  \u51116 \u49444 \u51221  (\u53364 \u46972 \u51060 \u50616 \u53944 \u50752  \u47784 \u45944 \u51008  \u50976 \u51648 )\
    if st.session_state['gemini_client']:\
        st.session_state['chat_session'] = create_chat_session(st.session_state['gemini_client'], st.session_state['model_name'])\
    st.toast("\uc0\u45824 \u54868 \u44032  \u52488 \u44592 \u54868 \u46104 \u50632 \u49845 \u45768 \u45796 . \u49352 \u47196 \u50868  \u47928 \u51032 \u47484  \u49884 \u51089 \u54644  \u51452 \u49464 \u50836 .", icon="\u55357 \u56492 ")\
\
# --- Gemini API \uc0\u44288 \u47144  \u54632 \u49688  ---\
\
@st.cache_resource(show_spinner=False)\
def get_gemini_client(api_key):\
    """API \uc0\u53412 \u47196  Gemini \u53364 \u46972 \u51060 \u50616 \u53944 \u47484  \u49373 \u49457 \u54616 \u44256  \u52880 \u49884 \u54633 \u45768 \u45796 ."""\
    if not api_key:\
        return None\
    try:\
        # \uc0\u53364 \u46972 \u51060 \u50616 \u53944  \u49373 \u49457  \u49884  key\u47484  \u51649 \u51217  \u51204 \u45804 \
        client = genai.Client(api_key=api_key)\
        # \uc0\u53364 \u46972 \u51060 \u50616 \u53944 \u44032  \u50976 \u54952 \u54620 \u51648  \u44036 \u45800 \u55176  \u54869 \u51064 \
        client.models.get(model=AVAILABLE_MODELS[0])\
        return client\
    except Exception as e:\
        st.error(f"API \uc0\u53412 \u44032  \u50976 \u54952 \u54616 \u51648  \u50506 \u44144 \u45208  \u53364 \u46972 \u51060 \u50616 \u53944  \u52488 \u44592 \u54868 \u50640  \u49892 \u54056 \u54664 \u49845 \u45768 \u45796 : \{e\}")\
        return None\
\
def create_chat_session(client, model_name):\
    """\uc0\u51648 \u51221 \u46108  \u47784 \u45944 \u44284  \u49884 \u49828 \u53596  \u54532 \u47212 \u54532 \u53944 \u47196  \u49352 \u47196 \u50868  Chat \u49464 \u49496 \u51012  \u49373 \u49457 \u54633 \u45768 \u45796 ."""\
    if not client:\
        return None\
    \
    # System Instruction\uc0\u51012  \u49324 \u50857 \u54616 \u50668  Chat \u49464 \u49496  \u49884 \u51089 \
    config = genai.types.GenerateContentConfig(\
        system_instruction=SYSTEM_INSTRUCTION\
    )\
    \
    chat_session = client.chats.create(\
        model=model_name,\
        config=config,\
    )\
    return chat_session\
\
def call_gemini_with_retry(chat_session, prompt, max_retries=5):\
    """\
    429 ResourceExhaustedError\uc0\u47484  \u54252 \u54632 \u54620  API \u50724 \u47448  \u48156 \u49373  \u49884  \u51116 \u49884 \u46020  \u47196 \u51649 \u51012  \u51201 \u50857 \u54616 \u50668  Gemini API\u47484  \u54840 \u52636 \u54633 \u45768 \u45796 .\
    """\
    for attempt in range(max_retries):\
        try:\
            # Gemini Chat API \uc0\u54840 \u52636 \
            response = chat_session.send_message(prompt)\
            return response\
        except ResourceExhaustedError:\
            if attempt < max_retries - 1:\
                wait_time = 2 ** attempt  # \uc0\u51648 \u49688  \u48177 \u50724 \u54532  (1\u52488 , 2\u52488 , 4\u52488 , ...)\
                st.warning(f"API \uc0\u50836 \u52397  \u54620 \u46020  \u52488 \u44284 (429). \{wait_time\}\u52488  \u54980  \u51116 \u49884 \u46020 \u54633 \u45768 \u45796 . (\u49884 \u46020  \{attempt + 1\}/\{max_retries\})")\
                time.sleep(wait_time)\
            else:\
                st.error("\uc0\u51648 \u49549 \u51201 \u51064  \u50836 \u52397  \u54620 \u46020  \u52488 \u44284 (429). \u51104 \u49884  \u54980  \u45796 \u49884  \u49884 \u46020 \u54644  \u51452 \u49464 \u50836 .")\
                return None\
        except APIError as e:\
            st.error(f"API \uc0\u50724 \u47448 \u44032  \u48156 \u49373 \u54664 \u49845 \u45768 \u45796 : \{e\}")\
            return None\
        except Exception as e:\
            st.error(f"\uc0\u50508  \u49688  \u50630 \u45716  \u50724 \u47448 \u44032  \u48156 \u49373 \u54664 \u49845 \u45768 \u45796 : \{e\}")\
            return None\
    return None\
\
# --- UI \uc0\u48143  \u47196 \u51649  \u49892 \u54665  ---\
\
def main():\
    initialize_session_state()\
\
    # --- \uc0\u49324 \u51060 \u46300 \u48148 : \u49444 \u51221  \u48143  \u44592 \u45733  ---\
    with st.sidebar:\
        st.header("\uc0\u9881 \u65039  \u49444 \u51221  \u48143  \u46020 \u44396 ")\
\
        # 1. API \uc0\u53412  \u51077 \u47141  (st.secrets \u46608 \u45716  \u51076 \u49884  UI)\
        api_key = st.secrets.get("GEMINI_API_KEY")\
        if not api_key:\
            st.warning("`st.secrets['GEMINI_API_KEY']`\uc0\u44032  \u49444 \u51221 \u46104 \u51648  \u50506 \u50520 \u49845 \u45768 \u45796 . \u50500 \u47000 \u50640  API \u53412 \u47484  \u51077 \u47141 \u54644  \u51452 \u49464 \u50836 .", icon="\u9888 \u65039 ")\
            api_key = st.text_input("Gemini API Key", type="password")\
            \
        # \uc0\u53364 \u46972 \u51060 \u50616 \u53944  \u52488 \u44592 \u54868 \
        if api_key:\
            st.session_state['gemini_client'] = get_gemini_client(api_key)\
        \
        if st.session_state['gemini_client']:\
            st.success("Gemini API \uc0\u53364 \u46972 \u51060 \u50616 \u53944  \u51456 \u48708  \u50756 \u47308 !")\
        else:\
            st.error("API \uc0\u53412 \u47484  \u54869 \u51064 \u54616 \u44256  \u51077 \u47141 \u54644  \u51452 \u49464 \u50836 .")\
\
        # 2. \uc0\u47784 \u45944  \u49440 \u53469 \
        model_selection = st.selectbox(\
            "\uc0\u47784 \u45944  \u49440 \u53469 :", \
            AVAILABLE_MODELS, \
            index=AVAILABLE_MODELS.index(st.session_state['model_name']),\
            key='model_selector'\
        )\
        # \uc0\u47784 \u45944  \u48320 \u44221  \u44048 \u51648  \u48143  \u49464 \u49496  \u51116 \u49444 \u51221 \
        if model_selection != st.session_state['model_name']:\
            st.session_state['model_name'] = model_selection\
            reset_conversation()\
\
        # Chat \uc0\u49464 \u49496  \u52488 \u44592 \u54868  (\u53364 \u46972 \u51060 \u50616 \u53944 \u44032  \u51456 \u48708 \u46104 \u50632 \u44256 , \u49464 \u49496 \u51060  \u50630 \u44144 \u45208  \u47784 \u45944 \u51060  \u48320 \u44221 \u46104 \u50632 \u51012  \u46412 )\
        if st.session_state['gemini_client'] and st.session_state['chat_session'] is None:\
            st.session_state['chat_session'] = create_chat_session(\
                st.session_state['gemini_client'], \
                st.session_state['model_name']\
            )\
\
        st.markdown("---")\
        \
        # 3. \uc0\u45824 \u54868  \u52488 \u44592 \u54868  \u48260 \u53948 \
        st.button("\uc0\u55357 \u56580  \u45824 \u54868  \u52488 \u44592 \u54868 ", on_click=reset_conversation)\
\
        # 4. CSV \uc0\u51088 \u46041  \u44592 \u47197  \u50741 \u49496  (\u52404 \u53356 \u48149 \u49828 )\
        record_to_csv = st.checkbox("\uc0\u9989  \u45824 \u54868  \u45236 \u50857  CSV \u51088 \u46041  \u44592 \u47197 ", value=True)\
\
        # 5. \uc0\u47196 \u44536  \u45796 \u50868 \u47196 \u46300 \
        if st.session_state['log_history']:\
            # \uc0\u47196 \u44536  \u45936 \u51060 \u53552 \u47484  DataFrame\u51004 \u47196  \u48320 \u54872 \
            df = pd.DataFrame(st.session_state['log_history'], columns=['Role', 'Content', 'Timestamp'])\
            csv_data = df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')\
            \
            st.download_button(\
                label="\uc0\u11015 \u65039  \u45824 \u54868  \u47196 \u44536  \u45796 \u50868 \u47196 \u46300  (CSV)",\
                data=csv_data,\
                file_name=f"customer_service_log_\{time.strftime('%Y%m%d_%H%M%S')\}.csv",\
                mime="text/csv",\
                key='download_button'\
            )\
            st.caption(f"\uc0\u52509  \{len(st.session_state['log_history'])//2\} \u53556  \u44592 \u47197 \u46120 .")\
\
        st.markdown("---")\
        st.caption(f"\uc0\u49324 \u50857  \u47784 \u45944 : **\{st.session_state['model_name']\}**")\
        if st.session_state['chat_session']:\
             st.caption(f"\uc0\u49464 \u49496  ID: **\{id(st.session_state['chat_session'])\}**")\
        else:\
             st.caption("\uc0\u49464 \u49496  \u49345 \u53468 : \u52488 \u44592 \u54868  \u54596 \u50836 ")\
\
    # --- \uc0\u47700 \u51064  \u50689 \u50669 : \u45824 \u54868  \u54364 \u49884  \u48143  \u51077 \u47141  ---\
    \
    if st.session_state['chat_session'] is None:\
        st.info("\uc0\u49884 \u51089 \u54616 \u47140 \u47732  Gemini API \u53412 \u47484  \u51077 \u47141 \u54616 \u44256  \u49444 \u51221 \u51012  \u50756 \u47308 \u54644  \u51452 \u49464 \u50836 .")\
        return\
\
    # \uc0\u52488 \u44592  \u47700 \u49884 \u51648  \u49444 \u51221  (\u52395  \u49884 \u51089  \u49884  \u54872 \u50689  \u47700 \u49884 \u51648 )\
    if not st.session_state['messages']:\
        welcome_message = "\uc0\u50504 \u45397 \u54616 \u49464 \u50836 , \u44256 \u44061 \u45784 . \u51613 \u44428 \u49324  \u50545  \u51060 \u50857  \u51473  \u44202 \u51004 \u49888  \u48520 \u54200  \u49324 \u54637 \u50640  \u45824 \u54644  \u44618 \u51008  \u49324 \u44284 \u51032  \u47568 \u50432 \u51012  \u51204 \u54633 \u45768 \u45796 . \u50612 \u46500  \u47928 \u51228 \u47196  \u48520 \u54200 \u51012  \u44202 \u51004 \u49512 \u45716 \u51648  \u47568 \u50432 \u54644  \u51452 \u49884 \u47732 , \u51221 \u51473 \u54616 \u44256  \u49888 \u49549 \u54616 \u44172  \u51217 \u49688 \u47484  \u46020 \u50752 \u46300 \u47532 \u44192 \u49845 \u45768 \u45796 ."\
        st.session_state['messages'].append(\{"role": "assistant", "content": welcome_message\})\
        st.session_state['log_history'].append(\{"Role": "assistant", "Content": welcome_message, "Timestamp": time.time()\})\
\
    # \uc0\u45824 \u54868  \u55176 \u49828 \u53664 \u47532  \u54364 \u49884 \
    for message in st.session_state['messages']:\
        with st.chat_message(message["role"]):\
            st.markdown(message["content"])\
\
    # \uc0\u49324 \u50857 \u51088  \u51077 \u47141  \u52376 \u47532 \
    if prompt := st.chat_input("\uc0\u48520 \u54200  \u49324 \u54637 \u51012  \u51088 \u49464 \u55176  \u47568 \u50432 \u54644  \u51452 \u49464 \u50836 ."):\
        \
        # 1. \uc0\u49324 \u50857 \u51088  \u47700 \u49884 \u51648  \u44592 \u47197  \u48143  \u54868 \u47732  \u54364 \u49884 \
        st.session_state['messages'].append(\{"role": "user", "content": prompt\})\
        st.session_state['log_history'].append(\{"Role": "user", "Content": prompt, "Timestamp": time.time()\})\
        with st.chat_message("user"):\
            st.markdown(prompt)\
\
        # 2. Gemini API \uc0\u54840 \u52636 \
        with st.chat_message("assistant"):\
            with st.spinner("\uc0\u45812 \u45817 \u51088 \u50640 \u44172  \u51204 \u45804 \u54624  \u45236 \u50857 \u51012  \u44160 \u53664 \u54616 \u44256  \u51080 \u49845 \u45768 \u45796 ..."):\
                \
                # API \uc0\u54840 \u52636  \u48143  429 \u50640 \u47084  \u51116 \u49884 \u46020  \u47196 \u51649  \u51201 \u50857 \
                response = call_gemini_with_retry(st.session_state['chat_session'], prompt)\
                \
                if response:\
                    full_response = response.text\
                    st.markdown(full_response)\
                    \
                    # 3. AI \uc0\u51025 \u45813  \u44592 \u47197  \u48143  \u54868 \u47732  \u54364 \u49884 \
                    st.session_state['messages'].append(\{"role": "assistant", "content": full_response\})\
                    st.session_state['log_history'].append(\{"Role": "assistant", "Content": full_response, "Timestamp": time.time()\})\
\
if __name__ == "__main__":\
    # Firebase \uc0\u49324 \u50857  \u49884  \u47196 \u44536  \u47112 \u48296  \u49444 \u51221  (\u54596 \u50836 \u50640  \u46384 \u46972  \u51452 \u49437  \u54644 \u51228 )\
    # import logging\
    # logging.basicConfig(level=logging.DEBUG) \
    \
    main()\
}