import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# # 显示中文
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
# plt.rcParams['axes.unicode_minus'] = False

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    return df


st.title("📺 Data_Visualization")
uploaded_file = st.file_uploader("Upload an article", type=("csv"))
if uploaded_file:
    # 读取数据
    df = load_data(uploaded_file)
    

    #柱状图1
    def build1(df):
        x = df['薪资'].value_counts()[0:10].keys()
        y = df['薪资'].value_counts()[0:10].values
        fig, ax = plt.subplots(figsize=(40, 20))
        ax.bar(x, y)
        plt.xlabel('salary range')
        plt.ylabel('occurrence number')
        plt.title("Top 10 bar chart of salary distribution")
        return fig

    fig1 = build1(df)
    st.pyplot(fig1)

    # 散点图1
    def sandian1(df):
        x = df['薪资'].value_counts()[0:10].keys()
        y = df['薪资'].value_counts()[0:10].values
        fig, ax = plt.subplots(figsize=(40, 20))
        ax.scatter(x, y, color="red", label=u"salary range data", linewidth=2)
        plt.xlabel('salary range')
        plt.ylabel('occurrence number')
        plt.title("Top 10 bar chart of salary distribution")
        plt.legend()
        return fig

    fig2 = sandian1(df)
    st.pyplot(fig2)

    # 折线图1
    def zhexian1(df):
        x = df['薪资'].value_counts()[0:10].keys()
        y = df['薪资'].value_counts()[0:10].values
        fig, ax = plt.subplots(figsize=(40, 20))
        ax.plot(x, y, marker='o', color='b', label="salary range data")
        plt.xlabel('salary range')
        plt.ylabel('occurrence number')
        plt.title("Top 10 plot chart of salary distribution")
        plt.legend()
        return fig

    fig3 = zhexian1(df)
    st.pyplot(fig3)


    # #柱状图2
    # def build2():
    #     x = df['要求'].value_counts()[0:20].keys()
    #     y = df['要求'].value_counts()[0:20].values
    #     fig, ax = plt.subplots()
    #     ax.bar(x, y)
    #     plt.xlabel('要求')
    #     plt.ylabel('出现次数')
    #     plt.title("要求前20柱状图")
    #     return fig

    # fig4 = build2()
    # st.pyplot(fig4)


    # # 散点图2
    # def sandian2():
    #     x = df['要求'].value_counts()[0:20].keys()
    #     y = df['要求'].value_counts()[0:20].values
    #     fig, ax = plt.subplots()
    #     ax.scatter(x, y, color="red", label=u"要求分布数据", linewidth=2)
    #     plt.xlabel('要求')
    #     plt.ylabel('出现次数')
    #     plt.title("要求前20散点图")
    #     plt.legend()
    #     return fig

    # fig5 = sandian2()
    # st.pyplot(fig5)

    # # 折线图2
    # def zhexian2():
    #     x = df['要求'].value_counts()[0:20].keys()
    #     y = df['要求'].value_counts()[0:20].values
    #     fig, ax = plt.subplots()
    #     ax.plot(x, y, marker='o', color='b', label="要求分布数据")
    #     plt.xlabel('要求')
    #     plt.ylabel('出现次数')
    #     plt.title("要求前20折线图")
    #     plt.legend()
    #     return fig

    # fig6 = zhexian2()
    # st.pyplot(fig6)

    # #柱状图3
    # def build3():
    #     x = df['公司位置'].value_counts()[0:20].keys()
    #     y = df['公司位置'].value_counts()[0:20].values
    #     fig, ax = plt.subplots()
    #     ax.bar(x, y)
    #     plt.xlabel('公司位置')
    #     plt.ylabel('出现次数')
    #     plt.title("公司位置前20柱状图")
    #     return fig

    # fig7 = build3()
    # st.pyplot(fig7)

    # # 散点图3
    # def sandian3():
    #     x = df['公司位置'].value_counts()[0:20].keys()
    #     y = df['公司位置'].value_counts()[0:20].values
    #     fig, ax = plt.subplots()
    #     ax.scatter(x, y, color="red", label=u"公司位置分布数据", linewidth=2)
    #     plt.xlabel('公司位置')
    #     plt.ylabel('出现次数')
    #     plt.title("公司位置前20散点图")
    #     plt.legend()
    #     return fig

    # fig8 = sandian3()
    # st.pyplot(fig8)

    # # 折线图3
    # def zhexian3():
    #     x = df['公司位置'].value_counts()[0:20].keys()
    #     y = df['公司位置'].value_counts()[0:20].values
    #     fig, ax = plt.subplots()
    #     ax.plot(x, y, marker='o', color='b', label="公司位置分布数据")
    #     plt.xlabel('公司位置')
    #     plt.ylabel('出现次数')
    #     plt.title("公司位置前20折线图")
    #     plt.legend()
    #     return fig

    # fig9 = zhexian3()
    # st.pyplot(fig9)


    # #柱状图4
    # def build4():
    #     x = df['企业名称'].value_counts()[0:20].keys()
    #     y = df['企业名称'].value_counts()[0:20].values
    #     fig, ax = plt.subplots()
    #     ax.bar(x, y)
    #     plt.xlabel('企业名称')
    #     plt.ylabel('出现次数')
    #     plt.title("企业名称前20柱状图")
    #     return fig

    # fig10 = build4()
    # st.pyplot(fig10)

    # # 散点图4
    # def sandian4():
    #     x = df['企业名称'].value_counts()[0:20].keys()
    #     y = df['企业名称'].value_counts()[0:20].values
    #     fig, ax = plt.subplots()
    #     ax.scatter(x, y, color="red", label=u"企业名称分布数据", linewidth=2)
    #     plt.xlabel('企业名称')
    #     plt.ylabel('出现次数')
    #     plt.title("企业名称前20散点图")
    #     plt.legend()
    #     return fig

    # fig11 = sandian4()
    # st.pyplot(fig11)

    # # 折线图4
    # def zhexian4():
    #     x = df['企业名称'].value_counts()[0:20].keys()
    #     y = df['企业名称'].value_counts()[0:20].values
    #     fig, ax = plt.subplots()
    #     ax.plot(x, y, marker='o', color='b', label="企业名称分布数据")
    #     plt.xlabel('企业名称')
    #     plt.ylabel('出现次数')
    #     plt.title("企业名称前20折线图")
    #     plt.legend()
    #     return fig

    # fig12 = zhexian4()
    # st.pyplot(fig12)
