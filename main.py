import streamlit as st
import transformers
from transformers import pipeline

# Define model names and label dictionaries
model_names = {
    "emotion": "mecoaoge2/VSMEC",
    "sentiment": "mecoaoge2/VLSP",
    "spam": "mecoaoge2/ViSpam",
    "hate_speech": "mecoaoge2/ViHSD",
    "hate_speech_span": "mecoaoge2/ViHOS"
}

emotion_labels = {'LABEL_6': 'Other', 'LABEL_4': 'Disgust', 'LABEL_5': 'Enjoyment', 'LABEL_3': 'Anger', 'LABEL_2': 'Surprise', 'LABEL_0': 'Sadness', 'LABEL_1': 'Fear'}
sentiment_labels = {'LABEL_0': 'Good', 'LABEL_1': 'Normal', 'LABEL_2': 'Bad'}
spam_labels = {'LABEL_0': 'No Spam', 'LABEL_1': 'Fake Review', 'LABEL_2': 'review on brand only', 'LABEL_3': 'non-review'}
hate_speech_labels = {'LABEL_0': 'Good', 'LABEL_1': 'Offensive', 'LABEL_2': 'Hate Speech'}

def classify_text(text, model_name, task, label_dict=None):
    print(123)
    classifier = pipeline(task, model=model_name, device=0)
    
    predictions = classifier(text)
    if label_dict:
        return [{**prediction, 'label': label_dict[prediction['label']]} for prediction in predictions]
    return predictions

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
        VISOBERT: MÔ HÌNH NGÔN NGỮ CHO TÁC VỤ XỬ LÝ DỮ LIỆU TRUYỀN THÔNG XÃ HỘI TIẾNG VIỆT
    </div>
    """, unsafe_allow_html=True)

st.title("DEMO")
text_input = st.text_area("Enter text:", "")

task = st.selectbox("Chọn một nhiệm vụ:", ["Nhận diện cảm xúc", "Phân tích cảm xúc", "Phát hiện spam", "Phát hiện lời nói căm thù", "Phát hiện khoảng lời nói căm thù"])
model_name = st.selectbox("Chọn một mô hình:", ["VisoGCN","VisoBert"])

if st.button("Classify"):
    if text_input:
        if task == "Phát hiện khoảng lời nói căm thù":
            results_ner = classify_text(text_input, model_names["hate_speech_span"], "ner")
            highlighted_text = highlight_tokens(text_input, results_ner)
            st.markdown(highlighted_text, unsafe_allow_html=True)
        else:
            if task == "Nhận diện cảm xúc":
                results = classify_text(text_input, model_names["emotion"], "text-classification", emotion_labels)
            elif task == "Phân tích cảm xúc":
                results = classify_text(text_input, model_names["sentiment"], "text-classification", sentiment_labels)
            elif task == "Phát hiện spam":
                results = classify_text(text_input, model_names["spam"], "text-classification", spam_labels)
            elif task == "Phát hiện lời nói căm thù":
                results = classify_text(text_input, model_names["hate_speech"], "text-classification", hate_speech_labels)

            label = results[0]['label']
            score = results[0]['score']
            st.write(f"{task}: {label} (Score: {score:.4f})")
    else:
        st.write("Please enter some text to classify.")
