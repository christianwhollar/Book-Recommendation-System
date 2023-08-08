from scripts.hybrid_filter_recommender import HybridFilterRecommender
from scripts.content_filter_recommender import ContentFilterRecommender

import pandas as pd
import streamlit as st
import sys
sys.path.append('scripts')
ref_path = 'data/processed/final_dataframe.pkl'
final_dataframe = pd.read_pickle(ref_path)

st.set_page_config(layout='centered')

@st.cache
def get_final_df():
    ref_path = 'data/processed/final_dataframe.pkl'
    final_dataframe = pd.read_pickle(ref_path)
    return final_dataframe

final_dataframe = get_final_df()
hfr = HybridFilterRecommender(final_dataframe)
cf = ContentFilterRecommender(final_dataframe)
isbn_ref = hfr.load_isbn_df()

# Session State
if 'clicks' not in st.session_state:
    st.session_state['clicks'] = {}
    st.session_state.clicks['book_selection_1'] = False
    st.session_state.clicks['book_selection_2'] = False
    st.session_state.clicks['book_selection_3'] = False

    st.session_state.clicks['display_book_2'] = False
    st.session_state.clicks['display_book_3'] = False

    st.session_state.clicks['enter'] = False
    
def click(key):
    st.session_state.clicks[key] = True

def show(key):
    st.session_state.clicks[key] = True

st.title('Book Recommender')

mode = st.select_slider('Select your mode:', options = ('Hybrid Filter', 'Content Filter'))

books = []

col01, col02 = st.columns([9,1])

with col01: 
    option1 = st.selectbox(
        'What is your favorite book?',
        options = isbn_ref['title_gr'].unique(),
        on_change = click,
        args = ['book_selection_1']
    )

    if st.session_state.clicks['book_selection_1']:
        st.write('You selected:', option1)
        books.append(option1)
    
with col02:
    st.button(
        label = '+',
        key = 'button1',
        on_click = click,
        args = ['display_book_2']
    )

col11, col12 = st.columns([9,1])

with col11:

    if st.session_state.clicks['display_book_2']:
        option2 = st.selectbox(
            label = 'What is your second favorite book?',
            options = isbn_ref['title_gr'].unique(),
            on_change = click,
            args = ['book_selection_2'],
            disabled = not st.session_state.clicks['display_book_2']
        )

        if st.session_state.clicks['book_selection_2']:
            st.write('You selected:', option2)
            books.append(option2)

with col12:
    if st.session_state.clicks['display_book_2']:
        st.button(
            label = '+',
            key = 'button2',
            on_click = click,
            args = ['display_book_3']
        )

col21, col22 = st.columns([9,1])

with col21:
    if st.session_state.clicks['display_book_3']:
        option3 = st.selectbox(
            label = 'What is your third favorite book?',
            options = isbn_ref['title_gr'].unique(),
            on_change = click,
            args = ['book_selection_3'],
            disabled = not st.session_state.clicks['display_book_3']
        )

        if st.session_state.clicks['book_selection_3']:
            st.write('You selected:', option3)
            books.append(option3)

with col22:
    if st.session_state.clicks['book_selection_3']:
        st.button(
            label = 'Search',
            key = 'button3',
            on_click = click,
            args = ['enter']
        )

if st.session_state.clicks['enter']:
    if mode == 'Hybrid Filter':
        authors, titles, years = hfr.get_default_books(selected_books = books)
    elif mode == 'Content Filter':
        best_user = hfr.get_best_user(selected_books = books)
        authors, titles, years = cf.get_aty_recs(user = best_user)

    st.markdown('The following books are recommend:')

    for author, title, year in zip(authors,titles,years):
        st.markdown(f'- {title} by {author} published in {int(year)}')

    st.markdown('''
    <style>
    [data-testid="stMarkdownContainer"] ul{
        list-style-position: inside;
    }
    </style>
    ''', unsafe_allow_html=True)