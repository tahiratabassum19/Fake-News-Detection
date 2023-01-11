import streamlit as st
import pickle
import string 
import pandas as pd 
import altair as alt
from nltk.corpus import stopwords
from nltk import word_tokenize
import nltk 
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()


def root_words(string):
    porter = PorterStemmer()
    
    #  sentence into a list of words
    words = word_tokenize(string)
    
    valid_words = []

    for word in words:
        
        root_word = porter.stem(word)
        
        valid_words.append(root_word)
        
    string = ' '.join(valid_words)

    return string 

tfidf = pickle.load(open('Fake-News-Detection/models/vectorizer.pkl','rb'))
rf_model = pickle.load(open('Fake-News-Detection/models/rf_model.pkl','rb'))
mn_model=pickle.load(open('Fake-News-Detection/models/mn.pkl','rb'))
#adab_model=pickle.load(open('Fake-News-Detection/models/adab.pkl','rb'))



def main():
    st.subheader("Home")
    def add_bg_from_url():
        st.markdown(
            f"""
        <style>
        .stApp {{
             background-image: url("https://media.istockphoto.com/id/1278709873/photo/brown-recycled-paper-crumpled-texture-background-cream-old-vintage-page-or-grunge-vignette.jpg?b=1&s=170667a&w=0&k=20&c=NqKmm_gkRwJAqpTbiiqv3TwfWjq9ymOwUDwfG2ck9no=");
             background-attachment: fixed;
             background-size: cover
         }}
        </style>
         """,
        unsafe_allow_html=True
        )

    add_bg_from_url() 
#CSS-HTML Plug in from st.markdoownfor BreakNews
    st.markdown("""
    <style>
    .d {
        font-size:60px !important;
        text-align: center;
        color: black
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="d"> BREAKING NEWS </p>', unsafe_allow_html=True)

    st.markdown("""
    <style>
    .details {
        font-size:20px !important;
        text-align: left;
    
        font-family:Gothic;
        font-weight: bold;
        color: maroon
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="details">Did you know 85% Internet users are tricked by fake news? Are you confident in your ability to detect fake news? Check the authenticity of your news article here ↓ </p>', unsafe_allow_html=True)
    text_input=st.text_input("Enter The Title Below", key = "<uniquevalueofsomesort>")
       

    button = st.button("Predict")
    if button:
        transformed = root_words(text_input)
        vector_input = tfidf.transform([transformed]).toarray()
        result = mn_model.predict(vector_input)
       
        if result[0] == 'Fake':
            st.markdown("""
                <style>
            .fake {
            font-size:30px !important;
            text-align: left;
            font-family:Gothic;
            font-weight: bold;
            color: red

            }
             </style>
            """, unsafe_allow_html=True)

            st.markdown('<p class="fake">FAKE !!!!</p>', unsafe_allow_html=True)

        else:
            st.markdown("""
                <style>
            .real {
            font-size:30px !important;
            text-align: left;
            font-family:Gothic;
            font-weight: bold;
            color: red

            }
             </style>
            """, unsafe_allow_html=True)
            st.markdown('<p class="real">Real!!!!</p>', unsafe_allow_html=True)
            st.balloons()

        st.markdown("""
                <style>
            .pred {
            font-size:25px !important;
            text-align: left;
            font-family:Gothic;
            font-weight: bold;
            color: black

            }
             </style>
            """, unsafe_allow_html=True)
        st.markdown('<p class="pred">Prediction Probabilty</p>', unsafe_allow_html=True)
        
        result_prob= mn_model.predict_proba(vector_input)
        proba_df =pd.DataFrame(result_prob, columns = mn_model.classes_)
        proba_df_clean = proba_df.T.reset_index()
        proba_df_clean.columns = ["Result","probability"]
        fig =alt.Chart(proba_df_clean).mark_bar().encode(x='Result',y='probability',color='Result')
        st.altair_chart(fig,use_container_width=True)

if __name__ == '__main__':
    main()
