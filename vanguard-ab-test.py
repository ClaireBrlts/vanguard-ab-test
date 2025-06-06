import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

demo_path ='Data\df_final_demo.txt'
final_experiment_path = 'Data\df_final_experiment_clients.txt'
final_web_data_1_path = 'Data\df_final_web_data_pt_1.txt'
final_web_data_2_path = 'Data\df_final_web_data_pt_2.txt'

demo_df = pd.read_csv(demo_path)
final_ex_clients_df = pd.read_csv(final_experiment_path)

final_web_1 = pd.read_csv(final_web_data_1_path)
final_web_2 = pd.read_csv(final_web_data_2_path)
final_web_df = pd.concat([final_web_1, final_web_2])

def get_distribution_meassure_numerical(dataframe, column_name):
    """
    Returns mean, median, mode, var, std, range and quantiles for the given column of dataframe
    Also shows a historigram

    Parameters:
        dataframe (pd.DataFrame): The DataFrame containing the data.
        column_name (str): The name of the numerical column to analyze.

    Returns (not implemented):
        mean (int).
        median (int).
        mode (int).
        variation (int).
        standard deviation (int).
        range (int): calculated by the following formula dataframe[column_name].max() - dataframe[column_name].min()
        quantiles (pd.Dataframe).

    """
    mean = dataframe[column_name].mean()
    median = dataframe[column_name].median()
    mode = dataframe[column_name].mode()[0]
    var = dataframe[column_name].var()
    std = dataframe[column_name].std()

    d_min = dataframe[column_name].min()
    d_max = dataframe[column_name].max()
    d_range = d_max - d_min

    quantiles = dataframe[column_name].quantile([0.25, 0.50, 0.75])

print('------------------------')
print(f'distribution meassures for {column_name}')
print('------------------------')
print(f""""
mean is {mean}
median is {median}
mode is {mode}
var is {var}
std is {std}
max is {d_max} and min is {d_min}
range is {d_range}
quantiles are {quantiles}
""")

sns.histplot(dataframe[column_name], kde=True, color='salmon')
plt.show()


def get_bar_plot(column_x, column_y):
    """
    returns bar plot
    """
    sns.barplot(x = column_x,
            y = column_y,
            palette='Set2',
            hue = column_x)

    plt.xticks(rotation=70, fontsize=9)
    plt.show()

def get_frequencies(dataframe, column_name):
    """
    Returns frequency and proportion tables for the given column of dataframe
    Also displays a bar plot for the frequency table.

    Parameters:
        dataframe (pd.DataFrame): The DataFrame containing the data.
        column_name (str): The name of the categorical column to analyze.

    Returns (not implemented):
        frequency_table (pd.Series): Count of each category.
        proportion_table (pd.Series): Proportion of each category.
    """
    frequency_table = dataframe[column_name].value_counts()

    proportion_table = dataframe[column_name].value_counts(normalize=True).apply(lambda x: f"{x:.2%}")

    print(frequency_table)
    print (proportion_table)
    get_bar_plot(column_x= frequency_table.index, column_y=frequency_table.values)

def get_frequencies_null_values(dataframe, column_name):
    """
    Returns frequency and proportion tables for the given column of dataframe
    Also displays a bar plot for the frequency table.

    Parameters:
        dataframe (pd.DataFrame): The DataFrame containing the data.
        column_name (str): The name of the categorical column to analyze.

    Returns (not implemented):
        frequency_table (pd.Series): Count of each category.
        proportion_table (pd.Series): Proportion of each category.
    """
    frequency_table = dataframe[column_name].value_counts(dropna=False)

    proportion_table = dataframe[column_name].value_counts(dropna=False, normalize=True).apply(lambda x: f"{x:.2%}")

    print(frequency_table)
    print (proportion_table)
    get_bar_plot(column_x= frequency_table.index.astype(str), column_y=frequency_table.values)
    
def get_generation():
    """
    This function merges the demographic dataframe with the web dataframe to calculate the generation of each client:
    Extracts year from date_time (from web dataframe)
    Calculates year_born (year - age) -> rounded to 0 decimals
    Calculates generation by binning year_born up until 2018 (data ends in 2017)
    """
    merged_df = pd.merge(left=demo_df, right=final_web_df, how='inner', on='client_id')
    merged_df['date_time'] = pd.to_datetime(merged_df['date_time'])
    merged_df['year'] = merged_df['date_time'].dt.year
        
    gen_df = merged_df[['client_id', 'clnt_age', 'gendr', 'year']]
    gen_df = gen_df.groupby(['client_id', 'gendr']).agg(age = ('clnt_age', 'mean'), year = ('year', 'mean')).reset_index()
    gen_df['year_born'] = round(gen_df['year'] - gen_df['age'], 0)
    bins = [1925, 1945, 1964, 1980, 1996, 2018]
    labels = ['silent generation', 'baby boomers', 'generation x', 'millenial', 'generation z']
    gen_df['generation'] = pd.cut(gen_df['year_born'], bins = bins, labels = labels)
    gen_df = gen_df[['client_id', 'gendr', 'generation']]
    return gen_df

clnt_gen_df = get_generation()
    #There are too many unknown/undefined we will study male and female only
m_mask = categorical_df['gendr'] == 'M'
f_mask = categorical_df['gendr'] == 'F'

masked_categorical_df = categorical_df[m_mask | f_mask]
get_frequencies(masked_categorical_df, 'gendr')

m_mask = clnt_gen_df['gendr'] == 'M'
f_mask = clnt_gen_df['gendr'] == 'F'

masked_clnt_gen_df = clnt_gen_df[m_mask | f_mask]
sns.countplot(data=masked_clnt_gen_df, x='gendr', hue='generation')
plt.show()

for column in numerical_df.columns:
    get_distribution_meassure_numerical(numerical_df, column)

get_frequencies_null_values(final_ex_clients_df, 'Variation')
final_ex_clients_df['Variation'].value_counts(dropna=False)
final_ex_clients_df['Variation'].value_counts(dropna=False, normalize=True).apply(lambda x: f"{x:.2%}")
#removing null values from experiment file
final_ex_clients_df_clean = final_ex_clients_df.dropna()

def get_variation(dataframe):
    dataframe_clean = dataframe['client_id'].isin(final_ex_clients_df_clean['client_id'])
    #adding Variation column to web path file by merging with experiment file. This will enable us to calculate KPIs for both groups. 
    dataframe_clean = pd.merge(final_ex_clients_df_clean, dataframe, on='client_id')
    #defining test and control group
    return dataframe_clean

def get_group(dataframe): 
    test_group = dataframe[dataframe['Variation']=='Test']
    control_group = dataframe[dataframe['Variation']=='Control']

    return test_group, control_group

demo_df_clean = get_variation(demo_df)
clnt_gen_df_clean = get_variation(clnt_gen_df)
final_web_df_clean = get_variation(final_web_df)

def completion_rate(df):
    completed_df = df[df['process_step']=='confirm']

    completion_rate = completed_df['client_id'].nunique() / df['client_id'].nunique()

    return completion_rate

print(completion_rate(get_group(final_web_df_clean)[0]))
print(completion_rate(get_group(final_web_df_clean)[1]))

# Time Spent on Each Step
def time_spent_on_each_step(df):
    df = df.copy()
    df['next_date_time'] = df.groupby('visit_id')['date_time'].shift(1)
    df['duration'] = pd.to_datetime(df['next_date_time'])-pd.to_datetime(df['date_time'])
    time_spent_on_each_step = df['duration'].mean()
    return time_spent_on_each_step

print(time_spent_on_each_step(get_group(final_web_df_clean)[0]))
print(time_spent_on_each_step(get_group(final_web_df_clean)[1]))

def get_errors(step_dataframe):
    """
    This function returns if a step has an error (meaning the previous step is the same as the next)
    It makes sure that the client_id and visit_id are the same (no errors between different clients/sessions)
    
    Parameters:
    step_dataframe (pd.DataFrame): it is important that this df has client_id, visit_id and process_step for this function to work

    Returns:
    df['step_validation] (pd.Series)
    """

    step_to_int = {'start': 0,
                   'step_1': 1,
                   'step_2': 2,
                   'step_3' : 3,
                   'confirm': 4}
    step_dataframe = step_dataframe.copy()
    step_dataframe = step_dataframe.sort_values(by='date_time', ascending=True)
    function_df = step_dataframe[['client_id', 'visit_id', 'process_step']]
    function_df['step_int'] = function_df['process_step'].map(step_to_int)
    function_df['next_expected_step'] = function_df['step_int']+1
    function_df = function_df.reset_index(drop=True)

    condition_client = function_df['client_id'] == function_df['client_id'].shift(-1)
    condition_visit = function_df['visit_id'] == function_df['visit_id'].shift(-1)
    condition_expected_step = function_df['next_expected_step'] != function_df['step_int'].shift(-1)

    function_df['step_validation'] = np.where(condition_client & condition_visit & condition_expected_step, 
                                    'error', 
                                    'ok')

    return function_df['step_validation']

final_web_df['step_validation'] = get_errors(final_web_df_clean)
error_ratio = final_web_df['step_validation'].value_counts()[1]/len(final_web_df['step_validation'])

print( f'error ratio is {round(error_ratio*100,2)}%')