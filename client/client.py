"""
Клиент для NLP микросервиса
"""

import requests
import json
import time
import sys
from typing import Dict, Any, Optional

class NLPClient:
    """Клиент для взаимодействия с NLP микросервисом"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Инициализация клиента
        
        Args:
            base_url: Базовый URL сервера
        """
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "NLP-Microservice-Client/1.0"
        })
    
    def check_connection(self) -> bool:
        """
        Проверка подключения к серверу
        
        Returns:
            bool: True если сервер доступен
        """
        try:
            response = self.session.get(f"{self.base_url}/", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def get_server_info(self) -> Dict[str, Any]:
        """
        Получение информации о сервере
        """
        try:
            response = self.session.get(f"{self.base_url}/")
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": f"Connection failed: {str(e)}"}
    
    def get_tfidf(self) -> Dict[str, Any]:
        """
        Получение TF-IDF матрицы
        """
        try:
            response = self.session.post(f"{self.base_url}/tf-idf")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    def bag_of_words(self, text: str) -> Dict[str, Any]:
        """
        Преобразование текста в Bag-of-Words
        
        Args:
            text: Текст для обработки
        """
        try:
            response = self.session.get(
                f"{self.base_url}/bag-of-words",
                params={"text": text}
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    def lsa_analysis(self, n_components: int = 2) -> Dict[str, Any]:
        """
        Латентный семантический анализ
        
        Args:
            n_components: Количество компонент
        """
        try:
            response = self.session.post(
                f"{self.base_url}/lsa",
                params={"n_components": n_components}
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    def tokenize(self, text: str) -> Dict[str, Any]:
        """
        Токенизация текста
        
        Args:
            text: Текст для токенизации
        """
        try:
            response = self.session.post(
                f"{self.base_url}/text_nltk/tokenize",
                data={"text": text}
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    def stem(self, text: str) -> Dict[str, Any]:
        """
        Стемминг текста
        
        Args:
            text: Текст для стемминга
        """
        try:
            response = self.session.post(
                f"{self.base_url}/text_nltk/stem",
                data={"text": text}
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    def lemmatize(self, text: str) -> Dict[str, Any]:
        """
        Лемматизация текста
        
        Args:
            text: Текст для лемматизации
        """
        try:
            response = self.session.post(
                f"{self.base_url}/text_nltk/lemmatize",
                data={"text": text}
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    def pos_tagging(self, text: str) -> Dict[str, Any]:
        """
        Частеречная разметка
        
        Args:
            text: Текст для POS тегирования
        """
        try:
            response = self.session.post(
                f"{self.base_url}/text_nltk/pos",
                data={"text": text}
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    def ner(self, text: str) -> Dict[str, Any]:
        """
        Распознавание именованных сущностей
        
        Args:
            text: Текст для NER
        """
        try:
            response = self.session.post(
                f"{self.base_url}/text_nltk/ner",
                data={"text": text}
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    def run_demo(self):
        """
        Запуск демонстрации всех функций
        """
        print("=" * 60)
        print("NLP MICROSERVICE DEMO")
        print("=" * 60)
        
        # Проверка подключения
        print("\n1. Проверка подключения к серверу...")
        if not self.check_connection():
            print("   ❌ Сервер не доступен!")
            print(f"   Убедитесь, что сервер запущен на {self.base_url}")
            return
        
        server_info = self.get_server_info()
        print(f"   ✅ Сервер доступен")
        print(f"   Версия: {server_info.get('version', 'N/A')}")
        print(f"   Документов в корпусе: {server_info.get('corpus_info', {}).get('documents', 0)}")
        
        # TF-IDF
        print("\n2. TF-IDF матрица...")
        tfidf = self.get_tfidf()
        if "error" not in tfidf:
            shape = tfidf.get("shape", {})
            print(f"   ✅ Размер матрицы: {shape.get('rows', 0)}x{shape.get('cols', 0)}")
        else:
            print(f"   ❌ Ошибка: {tfidf['error']}")
        
        # Bag-of-Words
        print("\n3. Bag-of-Words...")
        test_text = "машинное обучение python программирование"
        bow = self.bag_of_words(test_text)
        if "error" not in bow:
            vector = bow.get("vector", [])
            print(f"   ✅ Текст: '{test_text}'")
            print(f"   Размер вектора: {len(vector)}")
            print(f"   Найдено слов: {len(bow.get('found_words', []))}")
        else:
            print(f"   ❌ Ошибка: {bow['error']}")
        
        # LSA
        print("\n4. LSA анализ...")
        lsa = self.lsa_analysis(2)
        if "error" not in lsa:
            matrix = lsa.get("matrix", [])
            print(f"   ✅ Размер LSA матрицы: {len(matrix)}x{len(matrix[0]) if matrix else 0}")
            print(f"   Объясненная дисперсия: {lsa.get('total_variance', 0):.2%}")
        else:
            print(f"   ❌ Ошибка: {lsa['error']}")
        
        # NLTK функции
        print("\n5. NLTK функции (английский текст)...")
        eng_text = "FastAPI is a modern web framework for building APIs with Python 3.7+"
        
        # Токенизация
        tokens = self.tokenize(eng_text)
        if "error" not in tokens:
            print(f"   ✅ Токенизация: {len(tokens.get('tokens', []))} токенов")
        
        # Стемминг
        stems = self.stem(eng_text)
        if "error" not in stems:
            print(f"   ✅ Стемминг: {len(stems.get('stems', []))} стемм")
        
        # POS тегирование
        pos = self.pos_tagging(eng_text)
        if "error" not in pos:
            print(f"   ✅ POS тегирование: {len(pos.get('pos_tags', []))} тегов")
        
        # Пример на русском
        print("\n6. Пример на русском языке...")
        rus_text = "Московский государственный университет был основан в 1755 году."
        
        # Токенизация русского текста
        rus_tokens = self.tokenize(rus_text)
        if "error" not in rus_tokens:
            print(f"   ✅ Токенизация (русский): {rus_tokens.get('tokens', [])}")
        
        # NER для русского текста
        rus_ner = self.ner(rus_text)
        if "error" not in rus_ner:
            entities = rus_ner.get("entities", [])
            print(f"   ✅ NER найдено сущностей: {len(entities)}")
            for entity in entities:
                print(f"      - {entity['entity']} ({entity['type']})")
        
        print("\n" + "=" * 60)
        print("ДЕМОНСТРАЦИЯ ЗАВЕРШЕНА")
        print("=" * 60)

def interactive_mode(client: NLPClient):
    """
    Интерактивный режим работы с клиентом
    """
    print("\n" + "=" * 60)
    print("ИНТЕРАКТИВНЫЙ РЕЖИМ")
    print("=" * 60)
    
    menu = {
        "1": "TF-IDF матрица",
        "2": "Bag-of-Words",
        "3": "LSA анализ",
        "4": "Токенизация",
        "5": "Стемминг",
        "6": "Лемматизация",
        "7": "POS тегирование",
        "8": "NER",
        "9": "Информация о сервере",
        "0": "Выход"
    }
    
    while True:
        print("\n" + "-" * 40)
        print("МЕНЮ:")
        for key, value in menu.items():
            print(f"  {key} - {value}")
        print("-" * 40)
        
        try:
            choice = input("\nВыберите действие: ").strip()
            
            if choice == "0":
                print("Выход из программы.")
                break
            
            elif choice == "1":
                result = client.get_tfidf()
                if "error" not in result:
                    shape = result.get("shape", {})
                    print(f"\nTF-IDF матрица:")
                    print(f"  Размер: {shape.get('rows', 0)}x{shape.get('cols', 0)}")
                    print(f"  Документов: {len(result.get('documents', []))}")
                    print(f"  Словарь: {len(result.get('vocabulary', []))} слов")
                else:
                    print(f"Ошибка: {result['error']}")
            
            elif choice == "2":
                text = input("Введите текст: ").strip()
                if text:
                    result = client.bag_of_words(text)
                    if "error" not in result:
                        vector = result.get("vector", [])
                        found = result.get("found_words", [])
                        print(f"\nBag-of-Words для: '{text}'")
                        print(f"  Размер вектора: {len(vector)}")
                        print(f"  Найдено слов: {len(found)}")
                        print(f"  Слова: {found}")
                    else:
                        print(f"Ошибка: {result['error']}")
                else:
                    print("Текст не может быть пустым!")
            
            elif choice == "3":
                try:
                    n = input("Количество компонент [2]: ").strip()
                    n = int(n) if n else 2
                    result = client.lsa_analysis(n)
                    if "error" not in result:
                        matrix = result.get("matrix", [])
                        variance = result.get("total_variance", 0)
                        print(f"\nLSA анализ с {n} компонентами:")
                        print(f"  Размер матрицы: {len(matrix)}x{len(matrix[0]) if matrix else 0}")
                        print(f"  Объясненная дисперсия: {variance:.2%}")
                    else:
                        print(f"Ошибка: {result['error']}")
                except ValueError:
                    print("Ошибка: введите число")
            
            elif choice in ["4", "5", "6", "7", "8"]:
                text = input("Введите текст для анализа: ").strip()
                if text:
                    if choice == "4":
                        result = client.tokenize(text)
                        func_name = "Токенизация"
                    elif choice == "5":
                        result = client.stem(text)
                        func_name = "Стемминг"
                    elif choice == "6":
                        result = client.lemmatize(text)
                        func_name = "Лемматизация"
                    elif choice == "7":
                        result = client.pos_tagging(text)
                        func_name = "POS тегирование"
                    elif choice == "8":
                        result = client.ner(text)
                        func_name = "NER"
                    
                    if "error" not in result:
                        print(f"\n{func_name} для: '{text}'")
                        if choice == "4":
                            print(f"  Токены: {result.get('tokens', [])}")
                        elif choice == "5":
                            print(f"  Стеммы: {result.get('stems', [])}")
                        elif choice == "6":
                            print(f"  Леммы: {result.get('lemmas', [])}")
                        elif choice == "7":
                            print(f"  POS теги: {result.get('pos_tags', [])}")
                        elif choice == "8":
                            entities = result.get("entities", [])
                            print(f"  Найдено сущностей: {len(entities)}")
                            for entity in entities:
                                print(f"    - {entity['entity']} ({entity['type']})")
                    else:
                        print(f"Ошибка: {result['error']}")
                else:
                    print("Текст не может быть пустым!")
            
            elif choice == "9":
                result = client.get_server_info()
                print("\nИнформация о сервере:")
                for key, value in result.items():
                    print(f"  {key}: {value}")
            
            else:
                print("Неизвестная команда. Попробуйте снова.")
        
        except KeyboardInterrupt:
            print("\n\nПрограмма прервана пользователем.")
            break
        except Exception as e:
            print(f"Ошибка: {e}")

def main():
    """
    Основная функция клиента
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Клиент для NLP микросервиса")
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="URL сервера (по умолчанию: http://localhost:8000)"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Запустить демонстрацию"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Запустить интерактивный режим"
    )
    
    args = parser.parse_args()
    
    # Создание клиента
    client = NLPClient(base_url=args.url)
    
    # Проверка подключения
    if not client.check_connection():
        print(f"❌ Не удалось подключиться к серверу по адресу: {args.url}")
        print("Убедитесь, что сервер запущен:")
        print("  python -m server.main")
        print("  или")
        print("  uvicorn server.main:app --reload")
        sys.exit(1)
    
    # Запуск режима
    if args.demo:
        client.run_demo()
    elif args.interactive:
        interactive_mode(client)
    else:
        # По умолчанию запускаем демо
        client.run_demo()

if __name__ == "__main__":
    main()