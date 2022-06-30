print('{:^72s}'.format("Выберите способ выборки"))
print("Рандомной генерацией - 0")
print("Считываением из файла - 1")
print('{:^72s}'.format("Введите 0 или 1 для выбора"))
v = int(input())

if v == 0:
################################# Генерация данных ############################################
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_blobs
    from sklearn.cluster import KMeans

    print('{:^72s}'.format("Введите число объектов"))
    n_samples = int(input()) #Число объектов

    X, y = make_blobs(n_samples, n_features = 20, centers = 5)
#Вывод плота по сгенерированным данным
    plt.scatter(X[:, 0], X[:, 1])
    plt.title(r'Сгенерированные данные', fontsize=20)
    plt.show()

#Метод логтя для определения оптимального количества кластеров
    crit = []
    for k in range(2, 10):
        kmeans = KMeans(n_clusters = k)
        kmeans.fit(X)
        crit.append(kmeans.inertia_)
    plt.figure(figsize=(6, 6))
    plt.plot(range(2, 10), crit) #Вывод графика (Метод логтя)
    plt.title(r'График <Метод локтя>', fontsize=20)
    plt.show()

#Алгоритм к-средних

    print('{:^72s}'.format("Введите число кластеров, оптимального для данных. Используя график <Метод логктя>"))
    n_clusters = int(input()) #Число кластеров

    kmeans = KMeans(n_clusters, init = 'k-means++', n_init = 10)
    kmeans.fit(X) #Обучение алгоритма
    labels = kmeans.labels_ #Массив с метками кластера
    #Вывод плота с массива с метками кластера
    plt.scatter(X[:, 0], X[:, 1], c = labels)
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c = 'red', s = 50, alpha = 0.9);
    plt.title(r'Кластеры', fontsize=20)
    plt.show()
################################################################################################
elif v == 1:
################################# Считывание из файла ##########################################
    import pandas as pd
    import matplotlib.pyplot as plt
    
    #Считывание файла
    cust_df = pd.read_csv("Cust_Segmentation.csv")
    print(cust_df)
    
    #Удаление столба адрес, т.к. он нам не нужен
    df = cust_df.drop('Address', axis = 1)
    print(df)

    #Создание массива из датасета
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    from sklearn.cluster import KMeans
    
    X = df.values[:, 1:]
    X = np.nan_to_num(X)
    Clus_dataSet = StandardScaler().fit_transform(X)

    #Вывод плота по данным массива соотношение возраста к зп
    plt.scatter(X[:, 0], X[:, 3])
    plt.title(r'Данные', fontsize=20)
    plt.show()

    crit = []
    for k in range(2, 10):
        kmeans = KMeans(n_clusters = k)
        kmeans.fit(X)
        crit.append(kmeans.inertia_)
    plt.figure(figsize=(6, 6))
    plt.plot(range(2, 10), crit) #Вывод графика (Метод логтя)
    plt.title(r'График <Метод локтя>', fontsize=20)
    plt.show()

    #Метод к-средних
    print('{:^72s}'.format("Введите число кластеров"))
    clusterNum = int(input())
    k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 12)
    k_means.fit(X)
    labels = k_means.labels_
    
    #Создание столбца с метками кластера
    df["Cluster"] = labels

    #Группировка данных по меткам кластера и подсчет среднего для каждего столбца
    df.groupby("Cluster").mean()
    print(df.groupby("Cluster").mean())

    #Вывод плота по кластерам (возраст к зп)
    area = np.pi * ( X[:, 1])**2
    plt.scatter(X[:, 0], X[:, 3], s = area, c = labels.astype(np.float), alpha = 0.5)
    centers = kmeans.cluster_centers_
    plt.xlabel("Возраст", fontsize = 16)
    plt.ylabel("Зарплата", fontsize = 16)
    plt.show()
#################################################################################################
else:
    print("Вы указали неправильный номер")