"""

Виконала: Чорна Ольга
Для верифікації роботи алгоритмів обробки timeseries даних розроблено наступний скрипт.
Від дозволяє застосувати всі кроки з аналізу даних, запропоновані у даному курсі,
до реальних історичних фінансових даних з портала yahoo за останні 5 років з видаленими останніми 100 днями.
Після чього екстраполювати розроблену модель на 100 днів вперед та порівняти з реальними даними за остані 100 днів.
Дозволяє підтвердити або спростувати спроможність створеної екстраполяції.

"""


from HW_Chorna.data_transform.model_evaluate import *
from HW_Chorna.data_extract.data_parse import *


def model_visualise_with_check(data, MNK, check, text=''):
    """
        Visualize the original model and the model estimated using the method of least squares (MNK).

        Parameters:
        - model (array-like): Input array representing the original model.
        - text (str): Additional text for the y-axis label.

        Returns:
        None
    """
    plt.plot(data, label='Model data')
    plt.plot(MNK, label='Least squares method approximation')
    plt.plot(range(len(data), len(data)+100), check, label='Extrapolation check')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title(text)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # extracting the data
    URL = 'https://finance.yahoo.com/'
    File_name = 'data_storage/financial_data_5y_archive/BTC-USD.csv'
    # File_name = 'data_storage/financial_data_5y_archive/GC=F.csv'
    # File_name = 'data_storage/financial_data_5y_archive/JPY=X.csv'
    Data_name = 'Open'
    text = File_name
    # fixing gap for further check
    skipfooter = 100

    df = pd.read_csv(File_name)

    # slicing the data
    df_last_100 = df.tail(skipfooter)
    df.drop(df.tail(skipfooter).index, inplace=True)

    # creating 2 models:
    # model_real - 5-year history without last 100 days,
    # model_real_check_extrapolation - last 100 days real data
    model_real = df[Data_name].to_numpy()
    model_real_check_extrapolation = df_last_100[Data_name].to_numpy()

    # 5-year history real data analysing and extrapolation
    model, extrapolation = timeseries_model_analysis(model_real, text)

    # data and extrapolation for visual check
    model_visualise_with_check(model, extrapolation, model_real_check_extrapolation)
