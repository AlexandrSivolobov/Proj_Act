# 1. Data Collection
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn

# Load the dataset
dataset = pd.read_csv("students.csv")

def show_menu():
    print("\n" + "="*50)
    print("АНАЛИЗ ДАННЫХ СТУДЕНТОВ".center(50))
    print("="*50)
    print("1. Загрузка данных и метаинформация")
    print("2. Обработка пропущенных значений")
    print("3. Визуализация числовых переменных")
    print("4. Визуализация категориальных переменных")
    print("5. Статистический анализ групп")
    print("6. Дополнительный анализ")
    print("7. Экспорт данных для Grafana")
    print("0. Выход")
    print("="*50)
    
    choice = input("Выберите пункт меню (0-7): ")
    return choice

def section1():
    print("\n[1. Загрузка данных и метаинформация]")
    print("\nПервые 5 записей:")
    print(dataset.head(5))
    
    print("\nРазмерность данных:")
    def get_dimensionality(data):        
        print(f"# записей = {data.shape[0]}")
        print(f"# признаков = {data.shape[1]}")
    get_dimensionality(dataset)
    
    print("\nМетаданные:")
    def get_metadata(data):
        metadata = {}
        metadata["числовые"] = data.select_dtypes(include=["float64", "int64"]).columns.tolist()
        metadata["категориальные"] = data.select_dtypes(include=["object"]).columns.tolist()
        print("Числовые признаки:", metadata["числовые"])
        print("Категориальные признаки:", metadata["категориальные"])
        return metadata
    get_metadata(dataset)
    input("\nНажмите Enter для продолжения...")

def section2():
    print("\n[2. Обработка пропущенных значений]")
    global dataset
    
    print("\nВизуализация пропущенных значений:")
    def filter_missing(data):
        sbn.displot(
            data=data.isna().melt(value_name="missing"),
            y="variable",
            hue="missing",
            multiple="fill",
            aspect=1.5
        )
        plt.title("Наличие пропущенных значений")
        plt.show()
        cleaned_data = data.dropna()
        return cleaned_data
    
    dataset = filter_missing(dataset)
    print("\nНовая размерность после обработки пропусков:")
    print(f"# записей = {dataset.shape[0]}")
    print(f"# признаков = {dataset.shape[1]}")
    input("\nНажмите Enter для продолжения...")

def section3():
    print("\n[3. Визуализация числовых переменных]")
    numerical_vars = ['Age', 'Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night', 
                     'Mental_Health_Score', 'Addicted_Score']
    
    while True:
        print("\n" + "-"*50)
        print("АНАЛИЗ ЧИСЛОВЫХ ПЕРЕМЕННЫХ".center(50))
        print("-"*50)
        print("1. Гистограммы распределения")
        print("2. Бивариантные диаграммы")
        print("3. Диаграммы с усами (box plot)")
        print("4. Статистические показатели")
        print("5. Корреляционный анализ")
        print("0. Назад")
        
        sub_choice = input("Выберите пункт (0-5): ")
        
        if sub_choice == "1":
            print("\nГистограммы распределения:")
            def hist_frequencies(data, variables, bins=10):
                ncol_plots = 2
                nrow_plots = (len(variables) + ncol_plots - 1) // ncol_plots
                fig, ax = plt.subplots(nrow_plots, ncol_plots, figsize=(16, 4 * nrow_plots))
                ax = ax.flatten()
                colors = ["skyblue", "lightgreen", "orange", "pink", "purple"]

                for i, col in enumerate(variables):
                    sbn.histplot(data=data[col], ax=ax[i], color=colors[i], bins=bins)
                    plt.xlabel(col)
                    plt.ylabel("Частота")
                    ax[i].set_title(f"Гистограмма для {col}", fontsize=14)
                    
                plt.tight_layout()
                plt.show()
            hist_frequencies(dataset, numerical_vars)
            
        elif sub_choice == "2":
            print("\nБивариантные диаграммы:")
            def bivariate_plots(data, x_var, y_var):
                ncol_plots = 2
                nrow_plots = (len(x_var) + ncol_plots - 1) // ncol_plots
                fig, ax = plt.subplots(nrow_plots, ncol_plots, figsize=(16, 6 * nrow_plots))
                ax = ax.flatten()
                colors = ["skyblue", "lightgreen", "orange", "pink", "purple"]

                for i, col in enumerate(x_var):
                    ax[i] = sbn.barplot(data=data, x=col, y=y_var, ax=ax[i], color=colors[i], errorbar=None)
                    ax[i].bar_label(ax[i].containers[0], fontsize=10)
                    ax[i].set_title(f"{y_var} по {col}", fontsize=14)
                    plt.xlabel(col)
                    plt.ylabel(y_var)
                    
                plt.tight_layout()
                plt.show()
            
            print("Доступные числовые переменные для анализа:", numerical_vars)
            y_var = input("Введите переменную для оси Y (например, 'Addicted_Score'): ")
            bivariate_plots(data=dataset, x_var=["Academic_Level", "Country"], y_var=y_var)
            
        elif sub_choice == "3":
            print("\nДиаграммы с усами (box plot):")
            def get_boxplot(data, x_var):
                ncol_plots = 2
                nrow_plots = (len(x_var) + ncol_plots - 1) // ncol_plots
                fig, ax = plt.subplots(nrow_plots, ncol_plots, figsize=(16, 4 * nrow_plots))
                ax = ax.flatten()
                colors = ["skyblue", "lightgreen", "orange", "pink", "purple"]

                for i, col in enumerate(x_var):
                    sbn.boxplot(data=data, x=col, ax=ax[i], color=colors[i])
                    ax[i].set_title(f"Box plot для {col}", fontsize=14)
                    ax[i].set_xlabel(col)
                    
                plt.tight_layout()
                plt.show()
            get_boxplot(data=dataset, x_var=numerical_vars)
            
        elif sub_choice == "4":
            print("\nСтатистические показатели:")
            print(dataset[numerical_vars].describe().round(2))
            input("\nНажмите Enter для продолжения...")
            
        elif sub_choice == "5":
            print("\nКорреляционный анализ:")
            def plot_correlation(data, cols):
                corr = data[cols].corr()
                plt.figure(figsize=(10, 8))
                sbn.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
                plt.title("Матрица корреляций")
                plt.show()
            plot_correlation(dataset, numerical_vars)
            
        elif sub_choice == "0":
            break
        else:
            print("Неверный выбор, попробуйте снова.")

def section4():
    print("\n[4. Визуализация категориальных переменных]")
    categorical_vars = ['Gender', 'Academic_Level', 'Country', 'Most_Used_Platform', 
                       'Affects_Academic_Performance', 'Relationship_Status', 
                       'Conflicts_Over_Social_Media']
    
    while True:
        print("\n" + "-"*50)
        print("АНАЛИЗ КАТЕГОРИАЛЬНЫХ ПЕРЕМЕННЫХ".center(50))
        print("-"*50)
        print("1. Частотный анализ (таблицы)")
        print("2. Столбчатые диаграммы (абсолютные частоты)")
        print("3. Круговые диаграммы (относительные частоты)")
        print("0. Назад")
        
        sub_choice = input("Выберите пункт (0-3): ")
        
        if sub_choice == "1":
            print("\nЧастотный анализ категориальных переменных:")
            for col in categorical_vars:
                print(f"\n***** {col} ******")
                print(dataset[col].value_counts())
            input("\nНажмите Enter для продолжения...")
            
        elif sub_choice == "2":
            print("\nСтолбчатые диаграммы абсолютных частот:")
            def plot_barchart(data, col):
                results = data[col].value_counts()    
                plt.figure(figsize=(10, 5))
                ax = sbn.barplot(x=results.values.tolist(), y=results.index.tolist(), orient="y")
                ax.bar_label(ax.containers[0], fontsize=10, fmt="%d")
                plt.title(f"Распределение по {col}")
                plt.show()
            
            for col in categorical_vars:
                plot_barchart(dataset, col)
                
        elif sub_choice == "3":
            print("\nКруговые диаграммы относительных частот:")
            def plot_piechart(data, col):
                results = data[col].value_counts()
                total_samples = results.sum()
                rel_freq = results/total_samples
                sbn.set_style("whitegrid")
                plt.figure(figsize=(8, 8))
                plt.pie(rel_freq.values.tolist(), labels=rel_freq.index.tolist(), autopct='%1.1f%%')
                plt.title(f"Относительные частоты для {col}")
                plt.show()
            
            for col in categorical_vars[:5]:  # Показываем первые 5 чтобы избежать перегрузки
                plot_piechart(dataset, col)
                
        elif sub_choice == "0":
            break
        else:
            print("Неверный выбор, попробуйте снова.")

def section5():
    print("\n[5. Статистический анализ групп]")
    
    categorical_vars = ['Gender', 'Academic_Level', 'Country', 'Most_Used_Platform', 
                       'Affects_Academic_Performance']
    
    print("\nДоступные категориальные переменные:")
    for i, var in enumerate(categorical_vars, 1):
        print(f"{i}. {var}")
    
    while True:
        print("\n" + "-"*50)
        print("АНАЛИЗ ГРУПП".center(50))
        print("-"*50)
        print("1. Выбрать переменные для анализа")
        print("2. Абсолютные частоты (столбчатая диаграмма)")
        print("3. Относительные частоты (круговая диаграмма)")
        print("0. Назад")
        
        sub_choice = input("Выберите пункт (0-3): ")
        
        if sub_choice == "1":
            print("\nВыберите две категориальные переменные:")
            for i, var in enumerate(categorical_vars, 1):
                print(f"{i}. {var}")
            
            var1_idx = int(input("Первая переменная (номер): ")) - 1
            var2_idx = int(input("Вторая переменная (номер): ")) - 1
            
            global group_var1, group_var2, group_data
            group_var1 = categorical_vars[var1_idx]
            group_var2 = categorical_vars[var2_idx]
            
            group_data = dataset.groupby([group_var1, group_var2]).size().reset_index(name="samples")
            print(f"\nГруппировка по {group_var1} и {group_var2}:")
            print(group_data)
            
        elif sub_choice == "2":
            if 'group_data' not in globals():
                print("Сначала выберите переменные для анализа!")
                continue
                
            print(f"\nАбсолютные частоты по {group_var1} и {group_var2}:")
            def plot_absfreq_groups(group, g1, g2):
                plt.figure(figsize=(10, 6))
                sbn.barplot(data=group, x=g1, y="samples", hue=g2, palette="pastel")
                plt.xlabel(g1)
                plt.ylabel('Абсолютная частота')
                plt.title(f'Распределение по {g1} и {g2}')
                plt.tight_layout()
                plt.show()
            plot_absfreq_groups(group_data, group_var1, group_var2)
            
        elif sub_choice == "3":
            if 'group_data' not in globals():
                print("Сначала выберите переменные для анализа!")
                continue
                
            print(f"\nОтносительные частоты по {group_var1} и {group_var2}:")
            def plot_relfreq_groups(group, g1, g2):
                labels = group[g1] + " + " + group[g2]
                sizes = group["samples"]
                plt.figure(figsize=(10, 10))
                plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
                plt.axis('equal')
                plt.title(f'Относительные частоты по {g1} и {g2}')
                plt.show()
            plot_relfreq_groups(group_data, group_var1, group_var2)
            
        elif sub_choice == "0":
            break
        else:
            print("Неверный выбор, попробуйте снова.")

def section6():
    print("\n[6. Дополнительный анализ]")
    
    print("\nАнализ взаимосвязи психического здоровья и использования соцсетей:")
    plt.figure(figsize=(10, 6))
    sbn.scatterplot(data=dataset, x='Avg_Daily_Usage_Hours', y='Mental_Health_Score', 
                    hue='Affects_Academic_Performance', style='Gender')
    plt.title('Психическое здоровье vs время в соцсетях')
    plt.xlabel('Часов в соцсетях в день')
    plt.ylabel('Оценка психического здоровья')
    plt.show()
    
    print("\nАнализ зависимости сна от уровня зависимости:")
    sbn.boxplot(data=dataset, x='Addicted_Score', y='Sleep_Hours_Per_Night', hue='Gender')
    plt.title('Продолжительность сна vs уровень зависимости')
    plt.xlabel('Уровень зависимости')
    plt.ylabel('Часов сна за ночь')
    plt.show()

def section7():
    print("\n[7. Экспорт данных для Grafana]")
    
    # Создаем данные для экспорта
    if 'group_data' in globals():
        group_data.to_csv('grafana_group_data.csv', index=False)
        print("Данные группового анализа сохранены в grafana_group_data.csv")
    
    # Экспорт основных данных
    dataset[['Gender', 'Academic_Level', 'Avg_Daily_Usage_Hours', 
             'Mental_Health_Score', 'Addicted_Score']].to_csv('grafana_main_data.csv', index=False)
    print("Основные данные сохранены в grafana_main_data.csv")
    
    # Экспорт корреляционной матрицы
    corr_matrix = dataset[numerical_vars].corr()
    corr_matrix.to_csv('grafana_corr_data.csv')
    print("Корреляционная матрица сохранена в grafana_corr_data.csv")
    
    print("\nИмпортируйте эти CSV-файлы в Grafana для создания панелей мониторинга.")

# Основной цикл программы
while True:
    choice = show_menu()
    
    if choice == "1":
        section1()
    elif choice == "2":
        section2()
    elif choice == "3":
        section3()
    elif choice == "4":
        section4()
    elif choice == "5":
        section5()
    elif choice == "6":
        section6()
    elif choice == "7":
        section7()
    elif choice == "0":
        print("\nЗавершение работы программы...")
        break
    else:
        print("\nНеверный выбор, попробуйте снова.")