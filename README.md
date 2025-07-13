# AI Quiz Generator 🧠

An intelligent quiz generation app powered by Hugging Face's Mistral model that creates engaging quizzes from text, PDF, and image content. Extract content from multiple sources and generate interactive multiple-choice quizzes automatically.

## Features

- **Multi-Input Support**: Text, PDF, and Image content extraction
- **OCR Technology**: Extract text from images using EasyOCR
- **PDF Processing**: Extract text from PDF documents
- **AI-Powered Quiz Generation**: Uses Hugging Face Mistral model
- **Interactive Quiz Interface**: Take quizzes with real-time scoring
- **Progress Tracking**: Visual progress bars for content extraction
- **Customizable**: Choose number of questions (1-10)
- **Modern UI**: Clean Streamlit interface with emojis and animations

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up Hugging Face API Key

You'll need a Hugging Face API key to use this app. Get one from [Hugging Face Hub](https://huggingface.co/settings/tokens).

1. Create a free account at Hugging Face
2. Go to Settings > Access Tokens
3. Create a new token with "Read" permissions
4. Enter the token in the app sidebar

### 3. Run the App

```bash
streamlit run quizzus.py
```

The app will open in your default web browser, typically at `http://localhost:8501`.

## Usage Guide

### Step 1: Enter API Key
- Open the app and enter your Hugging Face API token in the sidebar
- The app will validate your token automatically

### Step 2: Choose Input Type
Select from three content sources:
- **Text**: Directly type or paste your content
- **PDF**: Upload PDF documents (supports multi-page)
- **Image**: Upload images with text (PNG, JPG, JPEG)

### Step 3: Extract Content
- For **Text**: Type directly in the text area
- For **PDF**: Upload your file and click "Extract Text"
- For **Image**: Upload your image and watch the OCR progress bar

### Step 4: Generate Quiz
1. Choose the number of questions (1-10)
2. Click "Generate Quiz!"
3. Wait for AI processing

### Step 5: Take the Quiz
- Answer all multiple-choice questions
- Submit your answers
- View your score and correct answers
- Generate a new quiz if desired

### Further steps:
 - Make a sumamry based on the input
 - Get further links based on the inputs

## Supported File Formats

- **Images**: PNG, JPG, JPEG
- **Documents**: PDF
- **Text**: Direct input

## Example Use Cases

- **Students**: Create quizzes from lecture notes or textbook pages
- **Teachers**: Generate assessments from educational materials
- **Professionals**: Test knowledge from training documents
- **Researchers**: Quiz yourself on academic papers

## Technical Details

- **Framework**: Streamlit for web interface
- **AI Model**: Hugging Face Mistral-7B-Instruct
- **OCR Engine**: EasyOCR for image text extraction
- **PDF Processing**: pdfplumber for document parsing
- **Image Processing**: PIL (Python Imaging Library)
- **Response Format**: JSON-structured quiz questions

## File Structure

```
hackathon-ai-quizzus/
├── quizzus.py          # Main application file
├── requirements.txt    # Python dependencies
├── README.md          # This file
└── venv/              # Virtual environment (optional)
```

## Dependencies

```
streamlit
requests
pdfplumber
Pillow
easyocr
numpy
```

## Troubleshooting

### Common Issues

**API Key Error**
- Make sure your Hugging Face token is valid
- Check that you have proper permissions
- Verify the token in the sidebar

**OCR Not Working**
- Ensure image is clear and has readable text
- Try different image formats (PNG works best)
- Check image size (not too large)

**PDF Extraction Fails**
- Make sure PDF is not password protected
- Try with text-based PDFs (not scanned images)
- Check file size limitations

**Quiz Generation Issues**
- Ensure extracted text is meaningful and substantial
- Try with shorter content if generation fails
- Check your API rate limits

### Performance Tips

- Use clear, high-contrast images for better OCR results
- Keep text content focused and educational
- Start with 3-5 questions for optimal results
- Ensure stable internet connection for API calls

## Error Handling

The app includes comprehensive error handling:
- Invalid API tokens
- OCR processing failures
- PDF extraction errors
- Network connectivity issues
- Malformed AI responses

## Future Enhancements

- Support for more image formats
- Advanced OCR preprocessing
- Quiz difficulty levels
- Export quiz results
- Multiple language support
- Audio content support

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Verify your API setup
3. Test with sample content
4. Report bugs with detailed descriptions

---

**Note**: This app requires an active internet connection and valid Hugging Face API credentials