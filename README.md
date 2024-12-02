# fakenews_distilbert

## Project Overview

### Context and Problem Statement
In the age of digital media, the rapid spread of misinformation poses a significant challenge to public discourse and decision-making. Fake news can influence opinions, manipulate political elections, and cause social unrest. Given the vast amount of content generated daily, manual verification of news authenticity is impractical.

### Approach
This project employs a machine learning approach to address the challenge of detecting fake news. By leveraging the DistilBERT model, a lighter version of the well-known BERT architecture, we develop a binary classification system capable of distinguishing between 'Fake News' and 'Real News'. DistilBERT offers a balance between performance and resource efficiency, making it suitable for real-time applications.


### Solution Implementation
The model was trained on a comprehensive dataset consisting of labeled news articles. Using natural language processing techniques, the text data was preprocessed and encoded using the tokenizer provided with DistilBERT. The training process involved fine-tuning the DistilBERT model on our dataset, adjusting parameters to optimize accuracy, precision, and recall.

The final model is integrated into an interactive web application using Gradio. This platform allows users to input news text and receive immediate classification results, facilitating easy demonstration of the model’s capabilities in real-world scenarios.

### Impact
The deployment of this fake news detection model can aid journalists, educators, and the general public in quickly verifying news content, thereby enhancing information integrity and reliability in media platforms.

## Analysis

### Impact of the Project
The fake news detection project significantly impacts the fight against misinformation by providing automated tools for verifying the authenticity of news articles. This not only assists journalists and news organizations in maintaining high standards of factual reporting but also educates the public about the prevalence of fake news, encouraging critical thinking towards unverified sources. Additionally, social media companies can use this model to flag potentially false information before it spreads widely, thus preventing the viral spread of misinformation and enhancing information integrity across platforms.

### Revelations and Insights
The project demonstrates the effectiveness of using DistilBERT, a distilled version of the more complex BERT model, which is particularly suitable for real-time applications where resources are limited. This shows that even resource-efficient models can achieve high accuracy, making advanced AI tools more accessible for various applications. The endeavor also highlights the challenges natural language processing faces, such as understanding context, sarcasm, and subtle nuances in language, which are crucial for accurately classifying nuanced or complex texts. Furthermore, the success of the model underscores the importance of high-quality, diverse training datasets in building robust machine learning models, as the quality of data directly influences the model's ability to generalize and perform in real-world scenarios.

### Next Steps
To enhance the project further, several steps are proposed. Expanding the training data to include more diverse sources, including non-English articles, could improve the model’s versatility and accuracy across different languages and cultural contexts. Experimenting with newer and more complex models such as RoBERTa or XLNet might provide performance improvements and potential increases in the accuracy of fake news detection. Partnering with news organizations and social media platforms could allow for the integration of this model into their content management workflows, enabling real-time news verification. Finally, implementing a mechanism to collect user feedback on model predictions could help continually refine and update the model based on real-world usage and evolving styles of news.

## Project Descriptive Card

### Uses
This fake news detection model utilizes the DistilBERT architecture to classify text as 'Fake News' or 'Real News'. It is designed for:
- News agencies to verify the authenticity of the information before publication.
- Educational platforms to provide tools for identifying reliable sources.
- Social media platforms to automatically flag potential misinformation.

### Sources
- **Model**: The model is based on the `distilbert-base-uncased` model from Hugging Face's Transformers library.
- **Training Data**: The model was trained on a dataset comprising various news articles sourced from [[Dataset Source]](https://www.kaggle.com/code/therealsampat/fake-news-detection). The dataset includes labeled articles as 'fake' or 'true'.
- **Code**: The implementation uses Python, with key libraries including PyTorch and Transformers for model training and deployment.

### Permissions
- The model and accompanying software are open-source, available under the MIT License.
- The training data used is publicly available and is distributed under the MIT License with permissions for academic and non-commercial use.

### Code
- **Repository**: The full source code and model files are available in this GitHub repository: [[Repository URL]](https://github.com/tessorastefan/fakenews_distilbert)
- **Key Files**:
  - `FakeNews.ipynb`: Notebook for model training and evaluation.
  - `FakeNewsApp.ipynb`: Notebook to launch an interactive Gradio interface.

### Contact
- For any inquiries or contributions, please open an issue in the GitHub repository or contact Tessora Stefan at tessora.stefan@vanderbilt.edu.

## Resources

### Presentation Recording

### Resource Links
- Paper on Detecting Fake News with Machine Learning: https://arxiv.org/abs/2102.04458
- GitHub Repository for Fake News Detection using Logistic Regression, Decision Tree Classifier, Gradient Boost Classifier, and Random Forest Classifier: https://github.com/kapilsinghnegi/Fake-News-Detection
- YouTube Video on Fake News Detection with BERT Fine-Tuning: https://www.youtube.com/watch?v=LbYF0yMIFaM
- Paper on Detecting Fake News with Python: https://www.sciencedirect.com/science/article/pii/S1877050924006252
- Paper on Detecting Fake News with NLP Algorithms: https://www.researchgate.net/publication/365857538_Detection_of_Fake_News_Using_Machine_Learning_and_Natural_Language_Processing_Algorithms

