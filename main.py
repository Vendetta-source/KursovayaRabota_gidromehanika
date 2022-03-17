# Библиотеки
from tkinter import *
import numpy as np
import matplotlib.pyplot as plt


# функция для кнопки Выход
def btn_exit():
    exit()


# функция для кнопки Очистка полей
def btn_clear():
    entry_dp_dx.delete(0, END)
    entry_U0.delete(0, END)
    entry_Re.delete(0, END)
    entry_h.delete(0, END)
    label_info.configure(text="")


# Метод Гаусса
def Gauss(Mat_sys, h, U_0):
    Mat_G = np.copy(Mat_sys)

    for i in range(h - 1):
        Mat_G[i + 1] = Mat_G[i + 1] + Mat_G[i] / (-Mat_G[i][i] / Mat_G[i + 1][i])

    for i in range(h):
        Mat_G[i] = -Mat_G[i] / Mat_G[i][i]

    for i in range(h - 2, -1, -1):
        Mat_G[i] = Mat_G[i] + Mat_G[i + 1] * Mat_G[i][i + 1]

    V = -Mat_G.T[-1]
    V = np.insert(V, [0, h], [0, U_0])

    return V


# Метод Гаусса-Зейделя
def Zeidel(a, b, c, d, h, U_0):
    e = 0.01
    Mat_GS_1 = b + c + a
    Mat_GS_2 = d

    V = np.zeros(h)

    acc = 0
    while not acc:
        V_iter = np.copy(V)
        for i in range(h):
            Sum_1 = sum(Mat_GS_1[i][j] * V_iter[j] for j in range(i))
            Sum_2 = sum(Mat_GS_1[i][j] * V[j] for j in range(i + 1, h))
            V_iter[i] = (Mat_GS_2[i] - Sum_1 - Sum_2) / Mat_GS_1[i][i]
        acc = np.sqrt(sum((V_iter[i] - V[i]) ** 2 for i in range(h))) <= e
        V = V_iter

    V = np.insert(V, [0, h], [0, U_0])

    return V


# Метод прогонки
def TDMA(G, dy, h, U_0, L, N, S):
    A = [-2 * G / dy] * (h + 2)
    B = [G / dy] * (h + 2)
    C = [G / dy] * (h + 2)
    D = [S * dy] * (h + 2)

    B[-1] = 0
    C[0] = 0
    D[-1] = S * dy - U_0 * G / dy

    P = [0] * (h + 2)
    P[0] = 0
    P[-1] = 0

    Q = [0] * (h + 2)
    Q[0] = 0
    Q[-1] = U_0

    V = [0] * (h + 2)
    V[-1] = Q[-1]

    for i in range(1, h + 1):
        P[i] = -B[i] / (A[i] + C[i] * P[i - 1])
        Q[i] = (D[i] - C[i] * Q[i - 1]) / (A[i] + C[i] * P[i - 1])

    for i in range(h, -1, -1):
        V[i] = P[i] * V[i + 1] + Q[i]

    # Таблица ТДМА
    ans_TDMA = np.round([N, L, V, A, B, C, D, P, Q], 3).T
    with open("Result.txt", "a", encoding="utf-8") as Output:
        print("\nТДМА", file=Output)
        print("№", "h", "V", "a", "b", "c", "d", "P", "Q", sep="\t", file=Output)
        for i in range(h + 2):
            for j in range(9):
                print(ans_TDMA[i][j], end="\t", file=Output)
            print(file=Output)

    return V


def calculations():
    S = int(entry_dp_dx.get())
    Re = int(entry_Re.get())
    U_0 = int(entry_U0.get())
    h = int(entry_h.get())

    y = 1
    G = 2 / Re
    dy = y / (h - 1)
    h = h - 2

    N = np.arange(1, h + 3)
    L = [dy * i for i in range(h + 2)]

    # Матрица системы
    b = np.diag([G / dy] * (h - 1), 1)
    c = np.diag([G / dy] * (h - 1), -1)
    a = np.diag([-2 * G / dy] * h)
    d = np.array([S * dy] * h).reshape(h, 1)
    d[-1] = S * dy - U_0 * G / dy
    Mat_sys = np.hstack((b + c + a, d))

    with open("Result.txt", "w", encoding="utf-8") as Result:
        print("МАТРИЦА СИСТЕМЫ", file=Result)

    with open("Result.txt", "a", encoding="utf-8") as Result:
        for i in range(h):
            for j in range(h + 1):
                print((np.round(Mat_sys[i][j], 3)), end="\t", file=Result)
            print(file=Result)

    # Результат
    ans = np.round([N, L, Gauss(Mat_sys, h, U_0), Zeidel(a, b, c, d, h, U_0), TDMA(G, dy, h, U_0, L, N, S)], 3).T
    with open("Result.txt", "a", encoding="utf-8") as Result:
        print("\nРЕЗУЛЬТАТЫ", file=Result)
        print("№", "h", "Гаусс", "Зейдель", "ТДМА", sep="\t", file=Result)
        for i in range(h + 2):
            for j in range(5):
                print(ans[i][j], end="\t", file=Result)
            print(file=Result)

    label_info.configure(text="Результаты расчета в файле result.txt и result.jpg\nРасчет окончен!")

    # Эпюры скорости
    plt.plot(ans.T[2], L, ans.T[3], L, ans.T[4], L)
    plt.title("ЭПЮРЫ СКОРОСТИ")
    plt.xlabel("V")
    plt.ylabel("y")
    plt.legend(["Гаусс", "Зейдель", "ТДМА"])
    plt.minorticks_on()
    plt.grid(which="major")
    plt.grid(which="minor")
    plt.xlim(xmin=0)
    plt.ylim(ymin=0)
    plt.savefig("result.jpg", dpi=600)


# Создание и настройки главного окна приложения
window = Tk()
window.title("Плоское ламинарное течение жидкости в канале")
window["bg"] = "gray80"
window.geometry('500x500')  # размеры окна

# Кнопка Выход
button_exit = Button(window, width=10, height=1, text="Выход", bg="red", fg="black", command=btn_exit)
button_exit.place(x=410, y=465)

# Кнопка Очистка полей
button_clear = Button(window, width=15, height=1, text="Очистка полей", bg="white", fg="black", command=btn_clear)
button_clear.place(x=205, y=465)

# Кнопка Расчет
button_calc = Button(window, width=10, height=1, text="Расчет", bg="white", fg="black", command=calculations)
button_calc.place(x=10, y=465)

# Фото
photo = PhotoImage(file="рисунок.png")
label_photo = Label(window, image=photo)
label_photo.place(x=120, y=30)

# Информационная метка
label_info = Label(window, text="", font=("Times New Roman", 14), bg="gray80")
label_info.place(x=10, y=310)

# Первая метка
label_input = Label(window, text="Ввод исходных данных:", font=("Times New Roman", 14), bg="gray80")
label_input.place(x=10, y=160)

# Метки для ввода исходных данных
label_Re = Label(window, text="Re = ", font=("Times New Roman", 13), bg="gray80")
label_Re.place(x=10, y=190)

label_dp_dx = Label(window, text="dP/dx = ", font=("Times New Roman", 13), bg="gray80")
label_dp_dx.place(x=10, y=220)

label_U0 = Label(window, text="U0 = ", font=("Times New Roman", 13), bg="gray80")
label_U0.place(x=10, y=250)

label_h = Label(window, text="h = ", font=("Times New Roman", 13), bg="gray80")
label_h.place(x=10, y=280)

# Поля ввода
entry_dp_dx = Entry(window)
entry_dp_dx.place(x=70, y=220)

entry_Re = Entry(window)
entry_Re.place(x=50, y=190)

entry_U0 = Entry(window)
entry_U0.place(x=50, y=250)

entry_h = Entry(window)
entry_h.place(x=40, y=280)

window.mainloop()
