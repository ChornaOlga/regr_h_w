import numpy as np
import pandas as pd
import requests

from bs4 import BeautifulSoup


def exsel_file_parsing(URL, File_name, Data_name):
    """
    Parse an Excel or csv file and extract a specific column of data.

    Parameters:
    - URL (str): The source URL or identifier for the data.
    - File_name (str): The name of the Excel file to be parsed.
    - Data_name (str): The name of the column whose data needs to be extracted.

    Returns:
    - numpy.ndarray: A NumPy array containing the data from the specified column.
    """

    try:
        d = pd.read_excel(File_name)
    except ValueError as ve:
        d = pd.read_csv(File_name)
    model_real = d[Data_name].to_numpy()
    print('Джерело даних: ', URL)
    return model_real


def parser_url_to_pandas(url, Data_name):
    """
        Parse HTML tables from a yahoo URL using pandas and extract a specific column of numerical data.

        Parameters:
        - url (str): The URL of the webpage containing HTML tables to parse.
        - Data_name (str): The name of the column whose numerical data needs to be extracted.

        Returns:
        - numpy.ndarray: A NumPy array containing the numerical data from the specified column.
        """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36'
    }
    response = requests.get(url, headers=headers)
    df_list = pd.read_html(response.text)  # this parses all the tables in webpages to a list
    df = df_list[0]
    df.drop(df.tail(1).index, inplace=True)
    column_names = df.columns.to_list()
    df[Data_name] = df[Data_name].astype(float)
    model_real = df[Data_name].to_numpy()
    print('Джерело даних: ', url)
    return model_real


def bs4_url_to_pandas(url, Data_name):
    """
            Parse HTML tables from a yahoo URL using beautiful soup and extract a specific column of numerical data.

            Parameters:
            - url (str): The URL of the webpage containing HTML tables to parse.
            - Data_name (str): The name of the column whose numerical data needs to be extracted.

            Returns:
            - numpy.ndarray: A NumPy array containing the numerical data from the specified column.
            """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36'
    }
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', attrs={'class': 'W(100%) M(0)'})
    table_rows = table.find_all('tr')
    data = []
    for row in table.find_all('tr'):
        row_data = []
        for cell in row.find_all('td'):
            row_data.append(cell.text.replace(',', ''))
        data.append(row_data)
    df = pd.DataFrame(data)
    df.drop(df.tail(1).index, inplace=True)
    df.drop(df.head(1).index, inplace=True)
    df.columns = ['Date', 'Open', 'High', 'Low', 'Close*', 'Adj Close**', 'Volume']
    df = df[::-1]
    df[Data_name] = df[Data_name].astype(float)
    model_real = df[Data_name].to_numpy()
    print('Джерело даних: ', url)
    return model_real


def yahoo_financial_data_interface():
    """
        Interact with Yahoo Finance to choose and retrieve historical financial data.

        This function prompts the user to choose a financial direction and returns the corresponding
        financial data from Yahoo Finance.

        Options:
        1 - Bitcoin Price History (url: https://finance.yahoo.com/quote/BTC-USD/history/)
        2 - Gold Price History (url: https://finance.yahoo.com/quote/GC%3DF/history)
        3 - Japanese Yen Price History (url: https://finance.yahoo.com/quote/JPY%3DX/history)

        Returns:
        - tuple: A tuple containing the chosen financial direction (text) and the corresponding financial data (model).
    """

    print('Оберіть напрям досліджень:')
    print('1 - Історія ціни Bitcoin url = https://finance.yahoo.com/quote/BTC-USD/history/')
    print('2 - Історія ціни золота url = https://finance.yahoo.com/quote/GC%3DF/history')
    print('3 - Історія ціни японської йени url = https://finance.yahoo.com/quote/JPY%3DX/history')
    mode = int(input('mode:'))

    if mode == 1:
        # ----------------- ПРИКЛАД парсингу_1 сайтів новин метод: GET -------------------------
        text = 'Ціна Bitcoin'
        print('Джерело: url = https://finance.yahoo.com/quote/BTC-USD/history/')
        url = 'https://finance.yahoo.com/quote/BTC-USD/history/'

    if mode == 2:
        # ----------------- ПРИКЛАД парсингу_2 сайтів новин метод: GET -------------------------
        text = 'Ціна золота'
        print('Обрано інформаційне джерело: https://finance.yahoo.com/quote/GC%3DF/history')
        url = 'https://finance.yahoo.com/quote/GC%3DF/history'

    if mode == 3:
        # ----------------- ПРИКЛАД парсингу_2 сайтів новин метод: GET -------------------------
        text = 'Ціна японської йени'
        print('Обрано інформаційне джерело: https://finance.yahoo.com/quote/JPY%3DX/history')
        url = 'https://finance.yahoo.com/quote/JPY%3DX/history'

    model = bs4_url_to_pandas(url, 'Open')

    return text, model


def oshadbank_financial_data_interface():
    """
        Interface for retrieving historical financial data from Oshadbank archives.

        Returns:
        - tuple: A tuple containing text information and the parsed financial data model.
    """

    file_name =''

    print('Аналіз архивів Ощадбанка за:')
    print('1 - лютий 2022 - лютий 2023')
    print('2 - лютий 2023 - лютий 2024')
    file_mode = int(input('mode:'))

    # Data parsing from file
    if file_mode == 1:
        file_name = 'data_storage/Oschadbank (USD).xls'
    if file_mode == 2:
        file_name = 'data_storage/Oschadbank(USD)new.xlsx'

    data_name = ''

    print('Оберіть напрям досліджень:')
    print('1 - Архів купівлі UAH/USD від Ощадбанк')
    print('2 - Архів продажу UAH/USD від Ощадбанк')
    print('3 - Архів курсу НБУ UAH/USD від Ощадбанк')
    data_mode = int(input('mode:'))

    if data_mode == 1:
        data_name = 'Купівля'
    if data_mode == 2:
        data_name = 'Продаж'
    if data_mode == 3:
        if file_mode == 1:
            data_name = 'КурсНбу'
        if file_mode == 2:
            data_name = 'Курс Нбу'

    text = 'Архів ' + data_name + ' UAH/USD від Ощадбанк\n'

    model = exsel_file_parsing('https://www.oschadbank.ua/rates-archive', file_name, data_name)

    return text, model


if __name__ == '__main__':
    url = 'https://finance.yahoo.com/quote/BTC-USD/history/'
    # Model = parser_url_to_pandas(url, 'Open')
    Model = bs4_url_to_pandas(url, 'Open')
    print(Model)
