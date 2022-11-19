import streamlit as st

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
text_input=st.text_input("Enter The Title Below")

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
text_input=st.text_input("Enter The Title Below")

st.markdown(f"",text_input)

#st.button("Check")

