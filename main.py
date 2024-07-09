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

emotion_labels = {
    'LABEL_6': 'Cảm xúc khác',
    'LABEL_4': 'Ghê tởm',
    'LABEL_5': 'Thích thú',
    'LABEL_3': 'Tức giận',
    'LABEL_2': 'Ngạc nhiên',
    'LABEL_0': 'Buồn',
    'LABEL_1': 'Sợ hãi'
}

sentiment_labels = {
    'LABEL_0': 'Tốt',
    'LABEL_1': 'Bình thường',
    'LABEL_2': 'Xấu'
}

spam_labels = {
    'LABEL_0': 'Không phải spam',
    'LABEL_1': 'Đánh giá giả mạo',
    'LABEL_2': 'Chỉ đánh giá thương hiệu',
    'LABEL_3': 'Không phải đánh giá'
}

hate_speech_labels = {
    'LABEL_0': 'Bình thường',
    'LABEL_1': 'Xúc phạm',
    'LABEL_2': 'Ngôn từ thù ghét'
}


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
                labels = emotion_labels
            elif task == "Phân tích cảm xúc":
                results = classify_text(text_input, classifiers["sentiment"])
                labels = sentiment_labels
            elif task == "Phát hiện spam":
                results = classify_text(text_input, classifiers["spam"])
                labels = spam_labels
            elif task == "Phát hiện lời nói căm thù":
                results = classify_text(text_input, classifiers["hate_speech"])
                labels = hate_speech_labels

            for result in results:
                label = labels[result['label']]
                score = result['score']
                st.write(f"{label}: (Score: {score:.4f})")
    else:
        st.write("Please enter some text to classify.")
