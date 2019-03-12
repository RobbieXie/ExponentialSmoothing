import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


def exponential_smoothing(alpha, s):
    s2 = np.zeros(s.shape)
    s2[0] = s[0]
    for i in range(1, len(s2)):
        s2[i] = alpha*s[i]+(1-alpha)*s2[i-1]
    return s2


def show_data(new_year, pre_year, data, s_pre_single,s_pre_double, s_pre_triple):
    year, time_id, number = data.T

    # sns.distplot(s_pre_double[2:21]-number)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.figure(figsize=(14, 6), dpi=100)
    plt.ylim(0, 100)
    plt.plot(year, number, color='blue', label="实际值")
    plt.plot(year[1:], s_pre_single[2:],
             color='orange', label="一次平滑预测值")
    plt.plot(year[1:], s_pre_double[2:21],
             color='red', label="二次平滑预测值")
    plt.plot(year[1:], s_pre_triple[2:21],
             color='green', label="三次平滑预测值")
    plt.legend(loc='lower right')
    plt.title('CPU Utilization')
    plt.xlabel('Time')
    plt.ylabel('Utilization')
    plt.xticks(new_year)
    # plt.colors()
    plt.show()


def main():
    alpha = .70
    pre_year = np.array([21])
    data_path = r'data8.txt'
    data = np.loadtxt(data_path)
    year, time_id, number = data.T
    initial_line = np.array([0, 0, number[0]])
    initial_data = np.insert(data, 0, values=initial_line, axis=0)
    initial_year, initial_time_id, initial_number = initial_data.T

    s_single = exponential_smoothing(alpha, initial_number)
    s_double = exponential_smoothing(alpha, s_single)

    s_pre_single = np.zeros(s_single.shape)
    for i in range(1, len(initial_time_id)):
        s_pre_single[i] = alpha*initial_number[i-1]+(1-alpha)*s_single[i-1]

    a_double = 2*s_single-s_double
    b_double = (alpha/(1-alpha))*(s_single-s_double)
    s_pre_double = np.zeros(s_double.shape)
    for i in range(1, len(initial_time_id)):
        s_pre_double[i] = a_double[i-1]+b_double[i-1]
    pre_next_year = a_double[-1]+b_double[-1]*1
    pre_next_two_year = a_double[-1]+b_double[-1]*2
    insert_year = np.array([pre_next_year, pre_next_two_year])
    s_pre_double = np.insert(s_pre_double, len(s_pre_double), values=np.array(
        [pre_next_year]), axis=0)

    s_triple = exponential_smoothing(alpha, s_double)

    a_triple = 3*s_single-3*s_double+s_triple
    b_triple = (alpha/(2*((1-alpha)**2)))*((6-5*alpha)*s_single -
                                           2*((5-4*alpha)*s_double)+(4-3*alpha)*s_triple)
    c_triple = ((alpha**2)/(2*((1-alpha)**2)))*(s_single-2*s_double+s_triple)

    s_pre_triple = np.zeros(s_triple.shape)

    for i in range(1, len(initial_time_id)):
        s_pre_triple[i] = a_triple[i-1]+b_triple[i-1]*1 + c_triple[i-1]*(1**2)

    pre_next_year = a_triple[-1]+b_triple[-1]*1 + c_triple[-1]*(1**2)
    pre_next_two_year = a_triple[-1]+b_triple[-1]*2 + c_triple[-1]*(2**2)
    insert_year = np.array([pre_next_year, pre_next_two_year])
    s_pre_triple = np.insert(s_pre_triple, len(s_pre_triple), values=np.array(
        [pre_next_year]), axis=0)

    new_year = np.insert(year, len(year), values=pre_year, axis=0)
    output_s = np.array([s_single,s_double])
    print(output_s)
    sns.distplot(s_pre_double[1:21]-number)

    output = np.array([new_year, s_pre_single,s_pre_double, s_pre_triple])
    print(output)
    result = check_anomaly_by_d(s_pre_double[1:19] - number[0:18])
    print(result)
    show_data(new_year, pre_year, data, s_pre_single, s_pre_double, s_pre_triple)


def check_anomaly_by_d(arr):
    last = arr[len(arr)-1]
    avg = arr[0:len(arr)-1].mean()
    var = arr[0:len(arr)-1].var()
    std = arr[0:len(arr)-1].std()
    print(last-avg,avg,var,std)
    if abs(last-avg) > 3*std:
        return 1
    else:
        return 0


if __name__ == '__main__':
    main()
