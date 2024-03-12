import streamlit as st
import pickle
import re
import nltk
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
import os

nltk.download('punkt')
nltk.download('stopwords')

# Setting up paths
model_dir = './model/'
clf_path = os.path.join(model_dir, 'clf.pkl')
tfidfd_path = os.path.join(model_dir, 'tfidf.pkl')

# Loading models
if os.path.exists(clf_path) and os.path.exists(tfidfd_path):
    with open(clf_path, 'rb') as f:
        clf = pickle.load(f)
    with open(tfidfd_path, 'rb') as f:
        tfidfd = pickle.load(f)
else:
    st.error("Model files not found. Please make sure the model files are located in the 'model' directory.")
    st.stop()

def clean_resume(resume_text):
    clean_text = re.sub('http\S+\s*', ' ', resume_text)
    clean_text = re.sub('RT|cc', ' ', clean_text)
    clean_text = re.sub('#\S+', '', clean_text)
    clean_text = re.sub('@\S+', '  ', clean_text)
    clean_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', clean_text)
    clean_text = re.sub(r'[^\x00-\x7f]', r' ', clean_text)
    clean_text = re.sub('\s+', ' ', clean_text)
    return clean_text


# Web app
def main():
    st.title("Resume Screening App")
    st.sidebar.header("Settings")

    uploaded_file = st.file_uploader('Upload Resume', type=['txt', 'pdf'])

    if uploaded_file is not None:
        try:
            resume_bytes = uploaded_file.read()
            resume_text = resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            # If UTF-8 decoding fails, try decoding with 'latin-1'
            resume_text = resume_bytes.decode('latin-1')

        cleaned_resume = clean_resume(resume_text)
        input_features = tfidfd.transform([cleaned_resume])
        prediction_id = clf.predict(input_features)[0]

        # Map category ID to category name
        category_mapping = {
            15: "Java Developer",
            23: "Testing",
            8: "DevOps Engineer",
            20: "Python Developer",
            24: "Web Designing",
            12: "HR",
            13: "Hadoop",
            3: "Blockchain",
            10: "ETL Developer",
            18: "Operations Manager",
            6: "Data Science",
            22: "Sales",
            16: "Mechanical Engineer",
            1: "Arts",
            7: "Database",
            11: "Electrical Engineering",
            14: "Health and fitness",
            19: "PMO",
            4: "Business Analyst",
            9: "DotNet Developer",
            2: "Automation Testing",
            17: "Network Security Engineer",
            21: "SAP Developer",
            5: "Civil Engineer",
            0: "Advocate",
        }

        category_name = category_mapping.get(prediction_id, "Unknown")

        st.subheader("Prediction Result:")
        st.success(f"Predicted Category: {category_name}")

        # Display additional information or visualizations
        show_word_cloud = st.sidebar.checkbox("Show Word Cloud", value=False, key="word_cloud_checkbox")

        if show_word_cloud:
            st.subheader("Word Cloud:")
            # Generate and display word cloud
            stopwords_list = set(stopwords.words('english'))
            wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=stopwords_list).generate(cleaned_resume)
            fig, ax = plt.subplots()
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
            
            # Display accuracy
            accuracy = calculate_accuracy()  # Calculate accuracy here
            if accuracy is not None:
                st.subheader("Accuracy:")
                st.info(f"Model Accuracy: {accuracy:.2f}%")
            else:
                st.error("Unable to calculate accuracy.")

# Python main
if __name__ == "__main__":
    main()
