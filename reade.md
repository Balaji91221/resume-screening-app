# Resume Screening App

This is a web application designed to automate the process of screening resumes using Natural Language Processing (NLP) techniques. The app predicts the category of a resume based on its content, allowing recruiters to quickly filter through large numbers of resumes.



## Features

- Upload a resume in .txt or .pdf format.
- Predict the category of the job role based on the resume content.
- Display a word cloud visualization of the most frequent words in the resume.
.
- Easily customizable and deployable.

## Installation

1. Clone the repository:

```
git clone https:/githubcomyour_usernameresume-screening-app.git
```

2. Install the required Python packages:

```
pip install -r requirements.txt
```



## Usage

1. Navigate to the project directory:


```
cd resume-screening-app
```



2. Run the Streamlit app:
```
streamlit run app.py
```


3. Upload a resume using the file uploader in the sidebar.
4. View the prediction result, word cloud, and model accuracy plot.

## Dataset

The model is trained on a dataset of labeled resumes, where each resume is associated with a job category. The dataset used for training can be found in the `data` directory.

## Model Training

The machine learning models used for prediction are trained using the `train_model.py` script. The trained models are saved in the `model` directory.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or create a pull request.

