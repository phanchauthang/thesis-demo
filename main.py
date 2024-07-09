import streamlit as st
from transformers import pipeline

# Load models and define label dictionaries
model_names_visogcn = {
    "emotion": "mecoaoge2/VSMEC",
    "sentiment": "mecoaoge2/VLSP",
    "spam": "mecoaoge2/ViSpam",
    "hate_speech": "mecoaoge2/ViHSD",
    "hate_speech_span": "mecoaoge2/ViHOS"
}

model_names_visobert = {
    "emotion": "mecoaoge2/VSMEC1",
    "sentiment": "mecoaoge2/VLSP1",
    "spam": "mecoaoge2/ViSpam1",
    "hate_speech": "mecoaoge2/ViHSD1",
    "hate_speech_span": "mecoaoge2/ViHOS1"
}

emotion_labels = {'LABEL_6': 'Other', 'LABEL_4': 'Disgust', 'LABEL_5': 'Enjoyment', 'LABEL_3': 'Anger', 'LABEL_2': 'Surprise', 'LABEL_0': 'Sadness', 'LABEL_1': 'Fear'}
sentiment_labels = {'LABEL_0': 'Good', 'LABEL_1': 'Normal', 'LABEL_2': 'Bad'}
spam_labels = {'LABEL_0': 'No Spam', 'LABEL_1': 'Fake Review', 'LABEL_2': 'review on brand only','LABEL_3':'non-review'}
hate_speech_labels = {'LABEL_0': 'Good', 'LABEL_1': 'Offensive', 'LABEL_2': 'Hate Speech'}

# Streamlit app
# Create two columns
col1, col2 = st.columns([1, 2])

# Place the image in the first column
with col1:
    st.image("a1.png", width=130)

# Place the text in the second column
with col2:
    st.markdown("""
    <div style="color: blue; font-size: 24px;">
                Khóa Luận Tốt Nghiệp <br>
        VISOBERT: MÔ HÌNH NGÔN NGỮ CHO TÁC VỤ XỬ LÝ DỮ LIỆU TRUYỀN THÔNG XÃ HỘI TIẾNG VIỆT
    </div>
    """, unsafe_allow_html=True)

st.title("DEMO")
text_input = st.text_area("Enter text:", "")

task = st.selectbox("Chọn một nhiệm vụ:", ["Nhận diện cảm xúc", "Phân tích cảm xúc", "Phát hiện spam", "Phát hiện lời nói căm thù", "Phát hiện khoảng lời nói căm thù"])
model_type = st.selectbox("Chọn một mô hình:", ["VisoBERT", "VisoGCN"])

if model_type == "VisoBERT":
    model_names = model_names_visobert
else:
    model_names = model_names_visogcn
device = 'cpu'
classifier_vihos = pipeline("ner", model=model_names["hate_speech_span"], device=device)
classifiers = {
    "emotion": pipeline("text-classification", model=model_names["emotion"], device=device),
    "sentiment": pipeline("text-classification", model=model_names["sentiment"], device=device),
    "spam": pipeline("text-classification", model=model_names["spam"], device=-1),
    "hate_speech": pipeline("text-classification", model=model_names["hate_speech"], device=device)
}

def classify_text(text, classifier):
    return classifier(text)

def highlight_tokens(text, results):
    highlighted_text = ""
    current_position = 0

    for result in results:
        start = result['start']
        end = result['end']
        entity = result['entity']

        # Append the text before the current token
        highlighted_text += text[current_position:start]

        # Highlight the token based on the entity type
        if 'T' in entity:
            highlighted_text += f'<mark style="background-color: yellow">{text[start:end]}</mark>'
        else:
            highlighted_text += text[start:end]

        current_position = end

    # Append the remaining text
    highlighted_text += text[current_position:]

    return highlighted_text

if st.button("Classify"):
    if text_input:
        if task == "Phát hiện khoảng lời nói căm thù":
            results_ner = classify_text(text_input, classifier_vihos)
            highlighted_text = highlight_tokens(text_input, results_ner)
            st.markdown(highlighted_text, unsafe_allow_html=True)
        else:
            if task == "Nhận diện cảm xúc":
                results = classify_text(text_input, classifiers["emotion"])
                label = emotion_labels[results[0]['label']]
                score = results[0]['score']
            elif task == "Phân tích cảm xúc":
                results = classify_text(text_input, classifiers["sentiment"])
                label = sentiment_labels[results[0]['label']]
                score = results[0]['score']
            elif task == "Phát hiện spam":
                results = classify_text(text_input, classifiers["spam"])
                label = spam_labels[results[0]['label']]
                score = results[0]['score']
            elif task == "Phát hiện lời nói căm thù":
                results = classify_text(text_input, classifiers["hate_speech"])
                label = hate_speech_labels[results[0]['label']]
                score = results[0]['score']

            st.write(f"{task}: {label} (Score: {score:.4f})")
    else:
        st.write("Please enter some text to classify.")
