import re
import nltk
import logging
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Настройка логгера
logger = logging.getLogger('TextPreprocessing')

def setup_nltk_resources():
    """Настройка ресурсов NLTK с обработкой ошибок"""
    resources = [
        ('tokenizers/punkt', 'punkt'),
        ('tokenizers/punkt_tab', 'punkt_tab'),
        ('corpora/stopwords', 'stopwords'),
        ('corpora/wordnet', 'wordnet')
    ]
    
    for path, package in resources:
        try:
            nltk.data.find(path)
            logger.debug(f"Ресурс {package} уже установлен")
        except LookupError:
            try:
                logger.info(f"Загрузка ресурса {package}...")
                nltk.download(package)
            except Exception as e:
                logger.error(f"Ошибка загрузки {package}: {e}")
                raise

def enhanced_text_preprocessing(text):
    """Улучшенная предобработка текста с логированием"""
    try:
        if not isinstance(text, str):
            logger.warning(f"Получен нестроковый текст: {type(text)}")
            return ""
            
        original_length = len(text)
        
        # Удаление URL
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Удаление спецсимволов
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Токенизация
        tokens = word_tokenize(text.lower())
        filtered_tokens = []
        
        # Инициализация инструментов
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))
        
        # Фильтрация и лемматизация
        for token in tokens:
            if token not in stop_words and len(token) > 2:
                lemma = lemmatizer.lemmatize(token)
                filtered_tokens.append(lemma)
        
        processed_text = ' '.join(filtered_tokens)
        logger.debug(
            f"Предобработка текста\n"
            f"Исходная длина: {original_length}\n"
            f"Обработанная длина: {len(processed_text)}\n"
            f"Токенов: {len(filtered_tokens)}"
        )
        
        return processed_text
        
    except Exception as e:
        logger.error(f"Ошибка предобработки текста: {e}", exc_info=True)
        return ""