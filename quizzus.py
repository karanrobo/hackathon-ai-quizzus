"""
# AI Quiz App that Accepts text, images, and pdf to generate quizzes
Please select what type of input you would like:
"""

import streamlit as st
import requests
import pdfplumber

from PIL import Image
import easyocr
import numpy as np
import json
from bs4 import BeautifulSoup

# Initialize variables properly
extracted_text = ""
uploaded_pdf = None
uploaded_image = None
text_input = ""
submitted = None


st.set_page_config(page_title="Quiz Generator", layout="centered")
st.title("🧠 AI Quiz Generator Powered by Hugginface API")



API_URL = "https://router.huggingface.co/novita/v3/openai/chat/completions"


# step 1, validate the api key
# Store API key in session_state
if "hf_token" not in st.session_state:
    st.session_state.hf_token = ""
    st.session_state.api_valid = False

with st.sidebar:
    st.subheader("🔐 Hugginface API Key Required")
    hf_token_input = st.text_input("Enter your Hugginface API Key", type="password")
    
    if hf_token_input and hf_token_input != st.session_state.hf_token:
        st.session_state.hf_token = hf_token_input
        headers = {"Authorization": f"Bearer {hf_token_input}",}

        def query(payload):
            response = requests.post(API_URL, headers=headers, json=payload)
            return response.json()


        # try validating:
        # Try a test request
        try:
            response = query({
            "messages": [
                {
                    "role": "user",
                    "content": "Hello!"
                }
            ],
            "model": "mistralai/mistral-7b-instruct"
            })
            res = response["choices"][0]["message"]["content"]

            if response:
                st.session_state.api_valid = True
                st.success(f"✅ HF token is valid!")
           
        except Exception as e:
            st.session_state.api_valid = False
            st.error(f"❌ Invalid token or connection issue: {e}")
            st.stop()



if not st.session_state.hf_token:
    st.warning("Please enter your Hugginface Token in the sidebar to continue.")
    st.stop()


# -- Step 1: API validation already done before this point --

if not st.session_state.get("api_valid", False):
    st.warning("Please enter a valid API key to continue.")
    st.radio("Select input type", ["Text", "PDF", "Image"], disabled=True)
    st.stop()

mode = st.radio("Select input type", ["Text", "PDF", "Image"])


# # t input that should bw turned into quiz
st.markdown("### 📥 Upload or Enter Content")



# image/pdf input that should be turned into quiz

def extract_pdf_text(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])



def extract_image_text(image_file):
    try:
        # Initialize progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Initialize OCR reader
        status_text.text("🔧 Initializing OCR engine...")
        progress_bar.progress(10)
        
        if 'ocr_reader' not in st.session_state:
            with st.spinner("Loading OCR models..."):
                st.session_state.ocr_reader = easyocr.Reader(['en'])
        
        # Step 2: Load and process image
        status_text.text("📸 Loading image...")
        progress_bar.progress(30)
        
        image = Image.open(image_file)
        image_array = np.array(image)
        
        # Step 3: Detect text regions
        status_text.text("🔍 Detecting text regions...")
        progress_bar.progress(50)
        
        # Step 4: Extract text
        status_text.text("📝 Extracting text...")
        progress_bar.progress(70)
        
        results = st.session_state.ocr_reader.readtext(image_array)
        
        # Step 5: Process results
        status_text.text("✨ Processing results...")
        progress_bar.progress(90)
        
        # Combine all detected text
        extracted_text = ' '.join([result[1] for result in results])
        
        # Step 6: Complete
        status_text.text("✅ Text extraction complete!")
        progress_bar.progress(100)
        
        # Clear progress indicators after a short delay
        import time
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        return extracted_text.strip() if extracted_text else "No text found in image"
        
    except Exception as e:
        # Clear progress indicators on error
        if 'progress_bar' in locals():
            progress_bar.empty()
        if 'status_text' in locals():
            status_text.empty()
        return f"Error processing image: {str(e)}"
# ...existing code...

# Initialize session state for extracted text
if "extracted_text" not in st.session_state:
    st.session_state.extracted_text = ""

# Initialize variables properly
uploaded_pdf = None
uploaded_image = None
text_input = ""

if mode == "Text":
    text_input = st.text_area("Enter your text content", key="text_input") 
elif mode == "PDF":
    uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])
elif mode == "Image":
    uploaded_image = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

# Only show extract button when there's content to process
can_extract = (
    (mode == "Text" and text_input.strip()) or 
    (mode == "PDF" and uploaded_pdf is not None) or 
    (mode == "Image" and uploaded_image is not None)
)

if can_extract and st.button("Extract Text"):
    with st.spinner("Processing..."):
        if mode == "Text":
            st.session_state.extracted_text = text_input
            st.success("✅ Text ready for quiz generation!")
        elif mode == "PDF" and uploaded_pdf:
            try:
                st.session_state.extracted_text = extract_pdf_text(uploaded_pdf)
                st.success("✅ PDF text extracted successfully!")
            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")
                st.session_state.extracted_text = ""
        elif mode == "Image" and uploaded_image:
            result = extract_image_text(uploaded_image)
            if not result.startswith("Error"):
                st.session_state.extracted_text = result
                st.success("✅ Image text extracted successfully!")
            else:
                st.error(result)
                st.session_state.extracted_text = ""

# Display extracted text if available
if st.session_state.extracted_text:
    st.subheader("📄 Extracted Content")
    st.subheader("Want to make changes to the extracted text? Just edit below:")
    
    # Use a unique key for the editable text area
    edited_text = st.text_area(
        "Extracted Text (Editable)", 
        value=st.session_state.extracted_text, 
        height=200, 
        key="editable_extracted_text"
    )
    
    # Update session state with edited text
    st.session_state.extracted_text = edited_text

num_questions = st.number_input(
    "Number of questions", 
    min_value=1, max_value=10, value=3, step=1, format="%d",
    help="Enter the number of questions for the quiz"
)

st.write(f"Number of questions: {num_questions}")


def generate_quiz_from_text(extracted_text):
    prompt = f"""
    Based on the following text, generate {num_questions} multiple choice questions in JSON format.

    Return a list of objects where each object has:
    - "question": the question string
    - "choices": a list of 4 strings (A, B, C, D)
    - "answer": the correct answer string

    Text:
    \"\"\"{extracted_text}\"\"\"
    
    Return only valid JSON format, do not label questions, do not provide any other information.
    """

    payload = {
        "model": "mistralai/mistral-7b-instruct",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 1000,
        "temperature": 0.7
    }

    response = requests.post(
        API_URL,
        headers={"Authorization": f"Bearer {st.session_state.hf_token}"},
        json=payload
    )

    return response

# Initialize session state for quiz content
if "quiz_content" not in st.session_state:
    st.session_state.quiz_content = None



if st.session_state.extracted_text:
    with st.form("quiz_form"):
        st.write("📝 Ready to generate quiz from your content!")
        st.write(f"Content length: {len(st.session_state.extracted_text)} characters")
        st.write(f"Number of Questions: {num_questions}")
        submitted = st.form_submit_button("🎯 Generate Quiz!", help="Generate quiz from this text")
else:
    st.info("👆 Please extract some content first to generate a quiz!")




if submitted and st.session_state.api_valid:
  
    st.session_state.quiz_content = None
    if "user_answers" in st.session_state:
        del st.session_state.user_answers
        
    with st.spinner("🧠 Generating quiz questions..."):
        try:
            response = generate_quiz_from_text(st.session_state.extracted_text)
            
            if response.status_code == 200:
                result = response.json()
                quiz_content_raw = result["choices"][0]["message"]["content"]
                
           
                st.session_state.quiz_content = quiz_content_raw
                
                st.success("🎉 Quiz generated successfully!")
                st.subheader("📋 Generated Quiz")
         
                with st.expander("🔍 View Raw Response", expanded=False):
                    st.code(quiz_content_raw, language="json")
                
            else:
                st.error(f"API Error: {response.status_code} - {response.text}")
                
        except Exception as e:
            st.error(f"Error generating quiz: {str(e)}")




# Display quiz if it exists in session state
if st.session_state.quiz_content:
    try:
        quiz_content = json.loads(st.session_state.quiz_content)
        
        if not isinstance(quiz_content, list):
            st.error("❌ Expected a list of questions, but got a different format.")
            # Don't use st.stop() here, it will stop the entire app
        else:
            st.markdown("---")
            st.title("📝 Quiz Time!")
            
            # Initialize session state for user answers
            if "user_answers" not in st.session_state:
                st.session_state.user_answers = {}

            for i, q in enumerate(quiz_content):
                st.markdown(f"### Q{i+1}) {q['question']}")
                
                # Show options as radio buttons
                answer = st.radio(
                    f"Select your answer for question {i+1}:",
                    q['choices'],
                    key=f"question_{i}",
                    index=None  # No default selection
                )
                
                # Store answer in session state
                if answer:
                    st.session_state.user_answers[i] = answer
            
            # Check if all questions are answered
            total_questions = len(quiz_content)
            answered_questions = len(st.session_state.user_answers)
            
            if answered_questions == total_questions:
                if st.button("📊 Submit Quiz"):
                    score = 0
                    
                    st.markdown("---")
                    st.subheader("🎯 Quiz Results")
                    
                    for i, q in enumerate(quiz_content):
                        user_answer = st.session_state.user_answers.get(i, "No answer")
                        correct_answer = q['answer']
                        is_correct = user_answer == correct_answer
                        
                        if is_correct:
                            score += 1
                            st.success(f"Q{i+1}: ✅ Correct! Your answer: {user_answer}")
                        else:
                            st.error(f"Q{i+1}: ❌ Wrong. Your answer: {user_answer} | Correct: {correct_answer}")
                    
                    # Show final score
                    percentage = (score / total_questions) * 100
                    st.markdown("---")
                    st.metric("Final Score", f"{score}/{total_questions}", f"{percentage:.1f}%")
                    
                    if percentage >= 80:
                        st.balloons()
                        st.success("🎉 Excellent work!")
                    elif percentage >= 60:
                        st.success("👍 Good job!")
                    else:
                        st.info("📚 Keep studying!")
                        
                    # Add button to generate new quiz
                    if st.button("🔄 Generate New Quiz"):
                        st.session_state.quiz_content = None
                        st.session_state.user_answers = {}
                        st.rerun()
            else:
                st.info(f"📝 Please answer all questions ({answered_questions}/{total_questions} completed)")

    except json.JSONDecodeError as e:
        st.error(f"❌ Failed to parse quiz JSON: {e}")
        st.error("The AI response was not in valid JSON format. Try generating again.")
        
        # Add button to clear and try again
        if st.button("🔄 Clear and Try Again"):
            st.session_state.quiz_content = None
            st.rerun()

else:
    if st.session_state.extracted_text:
        st.info("👆 Click 'Generate Quiz' to create questions from your content!")
    else:
        st.info("👆 Please extract some content first to generate a quiz!")

# Replace the existing "Next Steps" section with this:


def generate_summary_from_text(extracted_text):
    prompt = f"""
    Based on the following text, generate a short summary 
    Text:
    \"\"\"{extracted_text}\"\"\"
    """

    payload = {
        "model": "mistralai/mistral-7b-instruct",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 1000,
        "temperature": 0.7
    }

    response = requests.post(
        API_URL,
        headers={"Authorization": f"Bearer {st.session_state.hf_token}"},
        json=payload
    )

    return response


def duckduckgo_search(query, n=3):
    url = f"https://html.duckduckgo.com/html/?q={query}"
    headers = {"User-Agent": "Mozilla/5.0"}
    res = requests.get(url, headers=headers)
    soup = BeautifulSoup(res.text, "html.parser")

    results = []
    for a in soup.find_all("a", class_="result__a", limit=n):
        results.append({
            "title": a.get_text(),
            "link": a['href']
        })
    return results


def generate_search_queries_from_text(extracted_text):
    prompt = f"""
    Based on the following text, generate 3 concise keyword phrases or search queries 
    that a user could use to find further reading or explanations online.
    
    Return only the search queries as a plain list (no numbering or explanations).

    Text:
    \"\"\"{extracted_text}\"\"\"
    """

    payload = {
        "model": "mistralai/mistral-7b-instruct",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 300,
        "temperature": 0.5
    }

    response = requests.post(
        API_URL,
        headers={"Authorization": f"Bearer {st.session_state.hf_token}"},
        json=payload
    )

    if response.status_code == 200:
        text = response.json()['choices'][0]['message']['content']
        # Split into list of keywords/queries
        queries = [line.strip("•- ").strip() for line in text.strip().splitlines() if line.strip()]
        return queries
    else:
        st.error("Failed to generate keywords.")
        return []
    # After generating keywords



st.markdown("---")
st.title("🚀 Next Steps")


st.subheader("📝 Create a Summary")
st.write("Generate an AI-powered summary of your extracted content to better understand key concepts.")

if st.button("✨ Generate Summary", type="primary", use_container_width=True):
    if st.session_state.extracted_text:
        with st.spinner("🧠 Creating summary..."):
            st.success("📋 Summary generated! (Feature coming soon)")
            try:
                response = generate_summary_from_text(st.session_state.extracted_text)
                
                if response.status_code == 200:
                    result = response.json()
                    summary = result["choices"][0]["message"]["content"]
                    
                    st.success("📋 Summary generated successfully!")
                    st.subheader("📄 Summary")
                    st.write(summary)
                    
                    # Store summary in session state
                    st.session_state.summary = summary
                    
                else:
                    st.error(f"Failed to generate summary. Status: {response.status_code}")
                    
            except Exception as e:
                st.error(f"Error generating summary: {str(e)}")
    else:
        st.warning("⚠️ Please extract some content first!")


st.subheader("📚 Further Reading")
st.write("Get personalized recommendations for additional learning resources and related topics.")

if st.button("🔍 Find Resources", type="secondary", use_container_width=True):
    if st.session_state.extracted_text:
        with st.spinner("🔎 Finding resources..."):
            try:
                queries = generate_search_queries_from_text(st.session_state.extracted_text)
               
                if queries:
                    st.success("📖 Learning resources found!")
                    st.subheader("🔍 Recommended Search Topics")
                    for i in range(len(queries)):
                        st.write(queries[i].replace("\"", ""))
                        links = duckduckgo_search(queries[i])
                        
                        for link in links:
                            st.markdown(f"📖 [{link['title']}]({link['link']})")
                        
                        st.markdown("---")
                else:
                    st.warning("Could not generate search topics. Try with different content.")
                    
            except Exception as e:
                st.error(f"Error finding resources: {str(e)}")
    else:
        st.warning("⚠️ Please extract some content first!")

st.markdown("---")


# Pro Tips section using info container
st.markdown("---")
st.info("""
💡 **Pro Tips:**
- 📷 For better OCR results, use clear, high-contrast images
- 📚 Longer texts generate more comprehensive quizzes
- 🔄 Try different question counts to find your ideal quiz length
- ✏️ Edit extracted text to focus on specific topics
""")
