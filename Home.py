import streamlit as st
import pickle
import string 
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

tfidf = pickle.load(open('models/vectorizer.pkl','rb'))
rf_model = pickle.load(open('models/rf_model.pkl','rb'))
mn_model=pickle.load(open('models/mn.pkl','rb'))


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



#st.write("Did you know 85% Internet users are tricked by fake news? Are you confident in your ability to detect fake news? Check the authenticity of your news article here ↓ ")
#text_input=st.text_input("Enter The Title Below")

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
#st.write("Did you know 85% Internet users are tricked by fake news? Are you confident in your ability to detect fake news? Check the authenticity of your news article here ↓ ")
    text_input=st.text_input("Enter The Title Below", key = "<uniquevalueofsomesort>")

#st.markdown(f"",text_input)
    button = st.button("Predict")
    # label = {'Fake':0, 'Real':1}
    if button:
        st.text("Original Text:\n{}".format(text_input))
        transformed = root_words(text_input)
        vector_input = tfidf.transform([transformed]).toarray()
        result = mn_model.predict(vector_input)
        # st.write(result)
        if result[0] == 'Fake':
            st.write("Fake!!!!")
        else:
            st.write("Real")

            #st.balloons()
        # final_result = get_key(result, label)
        # st.success("News Categorized as:: {}".format(final_result))
    # to_predict = [result]
    # prediction = model.predict([to_predict])
    # print(prediction_proba)
    # value = prediction["Fake"]
    # if value == "Fake":
    #     pred_output = 'Fake'
    #     pred_proba = prediction_proba[0][0].round(2) * 100
    # else:
    #     pred_output = 'Real'
    #     pred_proba = prediction_proba[0][1].round(2) * 100
    # output_text = '## Predicted a ' + '%' + '**%s chance of %s** \n\n based on the input of %s' % (pred_proba, pred_output, str(to_predict))

if __name__ == '__main__':
    main()
