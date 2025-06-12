import pandas as pd
import numpy as np
import scipy.stats as st
import seaborn as sns
import matplotlib.pyplot as plt

demo_path =r'..\Data\Raw\df_final_demo.txt'
final_experiment_path = r'..\Data\Raw\df_final_experiment_clients.txt'
final_web_data_1_path = r'..\Data\Raw\df_final_web_data_pt_1.txt'
final_web_data_2_path = r'..\Data\Raw\df_final_web_data_pt_2.txt'

demo_df = pd.read_csv(demo_path)
final_ex_clients_df = pd.read_csv(final_experiment_path)

final_web_1 = pd.read_csv(final_web_data_1_path)
final_web_2 = pd.read_csv(final_web_data_2_path)
final_web_df = pd.concat([final_web_1, final_web_2])

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

clnt_gen_df = get_generation()
demo_df_clean = get_variation(demo_df)
clnt_gen_df_clean = get_variation(clnt_gen_df)
final_web_df_clean = get_variation(final_web_df)

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

def get_frequencies(dataframe, column_name = ''):
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

    if 'gendr' in dataframe.columns:
        m_mask = dataframe['gendr'] == 'M'
        f_mask = dataframe['gendr'] == 'F'
        column_name = 'gendr'
    
    masked_categorical_df = dataframe[m_mask | f_mask]
    frequency_table = masked_categorical_df[column_name].value_counts()

    proportion_table = masked_categorical_df[column_name].value_counts(normalize=True).apply(lambda x: f"{x:.2%}")
    

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
    if 'gendr' in dataframe.columns:
        m_mask = dataframe['gendr'] == 'M'
        f_mask = dataframe['gendr'] == 'F'
        column_name = 'gendr'
    
    masked_categorical_df = dataframe[m_mask | f_mask]
    frequency_table = masked_categorical_df[column_name].value_counts(dropna=False)

    proportion_table = masked_categorical_df[column_name].value_counts(dropna=False, normalize=True).apply(lambda x: f"{x:.2%}")

    print(frequency_table)
    print (proportion_table)
    get_bar_plot(column_x= frequency_table.index.astype(str), column_y=frequency_table.values)

def get_demographics(dataframe):
   """
   1. Observe which columns are numerical and which are categorical:
   2. Are there any columns that are numerical but can be considered categorical?
   3. Which are continuous and which are ordinal observations
   4. Proceed with the analysis

   Parameters:
   dataframe (pd.DataFrame)

   Returns:
   0. numerical_df (pd.DataFrame)
   1. categorical_df (pd.DataFrame)
   """

   numerical_df = dataframe.select_dtypes('number')
   categorical_df = dataframe.select_dtypes('object')

   #Drop client id as it is not relevant for analysis
   numerical_df.drop('client_id', axis=1, inplace=True)
   display(numerical_df, categorical_df)
   get_frequencies(dataframe)
   for column in numerical_df.columns:
      get_distribution_meassure_numerical(numerical_df, column)

   return numerical_df, categorical_df

get_demographics(demo_df_clean)

get_demographics(get_group(demo_df_clean)[0])
get_demographics(get_group(demo_df_clean)[1])

def get_completion_data(df):
    completed_df = df[df['process_step']=='confirm'].sort_values(by='date_time', ascending=True)

    n_completions = completed_df['visit_id'].nunique() 
    n = df['visit_id'].nunique()

    return n_completions, n

function_df = get_group(final_web_df_clean)[0]
function_df['process_step'].value_counts()

function_df = get_group(final_web_df_clean)[0]

completed_df = function_df[function_df['process_step']=='confirm']
completed_df['visit_id'].nunique() 
function_df['visit_id'].nunique()

# Time Spent on Each Step
def time_spent_on_each_step(df):
    df = df.sort_values(by='date_time', ascending=True).copy()
    df['next_date_time'] = df.groupby('visit_id')['date_time'].shift(-1)
    df['duration'] = pd.to_datetime(df['next_date_time'])-pd.to_datetime(df['date_time'])
    df['duration']
    df.to_csv('duration.csv')
    return df['duration']

# Time Spent on Each Step
def time_spent_on_step(df, step):

    df = df.sort_values(by='date_time', ascending=True).copy()
    df['next_date_time'] = df.groupby('visit_id')['date_time'].shift(-1)
    filered = df[df['process_step'] == step]
    filered['duration'] = pd.to_datetime(filered['next_date_time'])-pd.to_datetime(filered['date_time'])
    df.to_csv('step_duration.csv')
    return filered['duration']

def get_error_data(dataframe):
    """
    This function returns if a step has an error (meaning the previous step is the same as the next)
    It makes sure that the client_id and visit_id are the same (no errors between different clients/sessions)
    
    Parameters:
    step_dataframe (pd.DataFrame): it is important that this df has client_id, visit_id and process_step for this function to work

    Returns:
    error_rate (float)
    """
    step_to_int = {'start': 0,
                'step_1': 1,
                'step_2': 2,
                'step_3' : 3,
                'confirm': 4}
    step_dataframe = dataframe.copy()
    step_dataframe = step_dataframe.sort_values(by=['date_time', 'visit_id', 'client_id'], ascending=True)
    function_df = step_dataframe[['client_id', 'visit_id', 'process_step']].copy()

    function_df['step_int'] = function_df['process_step'].map(step_to_int)
    function_df['next_expected_step'] = function_df['step_int'].apply(lambda x: x+1 if x < 4 else 4 )
    #function_df = function_df.reset_index(drop=True)
    condition_client = function_df['client_id'] == function_df['client_id'].shift(-1)
    condition_visit = function_df['visit_id'] == function_df['visit_id'].shift(-1)
    condition_expected_step = function_df['next_expected_step'] != function_df['step_int'].shift(-1)

    function_df['step_validation'] = np.where(condition_client & condition_visit & condition_expected_step, 
                                    'error', 
                                    'ok')

    filtered = function_df[function_df['step_validation'] == 'error']
    n_errors = filtered['visit_id'].nunique()
    n = function_df['visit_id'].nunique()
    return n_errors, n, function_df

from statsmodels.stats.proportion import proportions_ztest

#H0 test_completion = control_completion
#H1 test_completion != control_completion

test_results = get_completion_data(get_group(final_web_df_clean)[0])
print(f'test: {test_results} rate is {test_results[0]/test_results[1]}')
test_completions = test_results[0]
test_count = test_results[1]

control_results = get_completion_data(get_group(final_web_df_clean)[1])
print(f'control: {control_results} rate is {control_results[0]/control_results[1]}')
control_completions = control_results[0]
control_count = control_results[1]

# Example data
confirms = [test_completions, control_completions]      # number of completions
print(confirms)
users = [test_count, control_count]        # number of users
print(users)

# Run the test
z_stat, p_value = proportions_ztest(confirms, users, alternative='two-sided')

# Output
print(f"Z = {z_stat}, p = {p_value}")

#H0 test_completion >= control_completion
#H1 test_completion < control_completion

test_results = get_completion_data(get_group(final_web_df_clean)[0])
print(f'test: {test_results} rate is {test_results[0]/test_results[1]}')
test_completions = test_results[0]
test_count = test_results[1]

control_results = get_completion_data(get_group(final_web_df_clean)[1])
print(f'control: {control_results} rate is {control_results[0]/control_results[1]}')
control_completions = control_results[0]
control_count = control_results[1]

# Example data
confirms = [test_completions, control_completions]      # number of completions
print(confirms)
users = [test_count, control_count]        # number of users
print(users)

# Run the test
z_stat, p_value = proportions_ztest(confirms, users, alternative='smaller')

# Output
print(f"Z = {z_stat}, p = {p_value}")

#H0 avg time step control = avg time step test
#h1 avg time step control != avg time step test

#significance level: 5%
test = time_spent_on_each_step(get_group(final_web_df_clean)[0]).dropna().dt.seconds
control = time_spent_on_each_step(get_group(final_web_df_clean)[1]).dropna().dt.seconds

#H0: mu_price female <= mu_price male
#H1: mu_price female > mu_price male

alpha = 0.05


st.ttest_ind(control, test, equal_var=False) 

#H0 avg time step control >= avg time step test
#h1 avg time step control < avg time step test

#significance level: 5%
test = time_spent_on_each_step(get_group(final_web_df_clean)[0]).dropna().dt.seconds
control = time_spent_on_each_step(get_group(final_web_df_clean)[1]).dropna().dt.seconds

#H0: mu_price female <= mu_price male
#H1: mu_price female > mu_price male

alpha = 0.05


st.ttest_ind(control, test, equal_var=False, alternative='less') 

#H0 avg time step control = avg time step test
#h1 avg time step control != avg time step test

#significance level: 5%
test = time_spent_on_step(get_group(final_web_df_clean)[0],'start').dropna().dt.seconds
control = time_spent_on_step(get_group(final_web_df_clean)[1], 'start').dropna().dt.seconds

#H0: mu_price female <= mu_price male
#H1: mu_price female > mu_price male

alpha = 0.05


st.ttest_ind(control, test, equal_var=False) 

#H0 avg time step control = avg time step test
#h1 avg time step control != avg time step test

#significance level: 5%
test = time_spent_on_step(get_group(final_web_df_clean)[0],'step_1').dropna().dt.seconds
control = time_spent_on_step(get_group(final_web_df_clean)[1], 'step_1').dropna().dt.seconds

#H0: mu_price female <= mu_price male
#H1: mu_price female > mu_price male

alpha = 0.05


st.ttest_ind(control, test, equal_var=False) 

#H0 avg time step control <= avg time step test
#h1 avg time step control > avg time step test

#significance level: 5%
test = time_spent_on_step(get_group(final_web_df_clean)[0],'step_1').dropna().dt.seconds
control = time_spent_on_step(get_group(final_web_df_clean)[1], 'step_1').dropna().dt.seconds

#H0: mu_price female <= mu_price male
#H1: mu_price female > mu_price male

alpha = 0.05


st.ttest_ind(control, test, equal_var=False, alternative='greater') 

#H0 avg time step control = avg time step test
#h1 avg time step control != avg time step test

#significance level: 5%
test = time_spent_on_step(get_group(final_web_df_clean)[0],'step_2').dropna().dt.seconds
control = time_spent_on_step(get_group(final_web_df_clean)[1], 'step_2').dropna().dt.seconds

#H0: mu_price female <= mu_price male
#H1: mu_price female > mu_price male

alpha = 0.05


st.ttest_ind(control, test, equal_var=False) 

#H0 avg time step control <= avg time step test
#h1 avg time step control > avg time step test

#significance level: 5%
test = time_spent_on_step(get_group(final_web_df_clean)[0],'step_2').dropna().dt.seconds
control = time_spent_on_step(get_group(final_web_df_clean)[1], 'step_2').dropna().dt.seconds

#H0: mu_price female <= mu_price male
#H1: mu_price female > mu_price male

alpha = 0.05


st.ttest_ind(control, test, equal_var=False, alternative='greater') 

#H0 avg time step control = avg time step test
#h1 avg time step control != avg time step test

#significance level: 5%
test = time_spent_on_step(get_group(final_web_df_clean)[0],'step_3').dropna().dt.seconds
control = time_spent_on_step(get_group(final_web_df_clean)[1], 'step_3').dropna().dt.seconds

#H0: mu_price female <= mu_price male
#H1: mu_price female > mu_price male

alpha = 0.05


st.ttest_ind(control, test, equal_var=False) 

#H0 avg time step control <= avg time step test
#h1 avg time step control > avg time step test

#significance level: 5%
test = time_spent_on_step(get_group(final_web_df_clean)[0],'step_3').dropna().dt.seconds
control = time_spent_on_step(get_group(final_web_df_clean)[1], 'step_3').dropna().dt.seconds

#H0: mu_price female <= mu_price male
#H1: mu_price female > mu_price male

alpha = 0.05


st.ttest_ind(control, test, equal_var=False, alternative = 'greater') 

#H0 test_error = control_error
#H1 test_error != control_error

test_results = get_error_data(get_group(final_web_df_clean)[0])
print(f'test: {test_results} rate is {test_results[0]/test_results[1]}')
test_completions = test_results[0]
test_count = test_results[1]

control_results = get_error_data(get_group(final_web_df_clean)[1])
print(f'control: {control_results} rate is {control_results[0]/control_results[1]}')
control_completions = control_results[0]
control_count = control_results[1]

# Example data
confirms = [test_completions, control_completions]      # number of completions
print(confirms)
users = [test_count, control_count]        # number of users
print(users)

# Run the test
z_stat, p_value = proportions_ztest(confirms, users)

# Output
print(f"Z = {z_stat}, p = {p_value}")

#H0 test_error <= control_error
#H1 test_error > control_error

test_results = get_error_data(get_group(final_web_df_clean)[0])
print(f'test: {test_results[0],test_results[1]} rate is {test_results[0]/test_results[1]}')
test_completions = test_results[0]
test_count = test_results[1]

control_results = get_error_data(get_group(final_web_df_clean)[1])
print(f'control: {control_results[0],control_results[1]} rate is {control_results[0]/control_results[1]}')
control_completions = control_results[0]
control_count = control_results[1]

# Example data
confirms = [test_completions, control_completions]      # number of completions
print(confirms)
users = [test_count, control_count]        # number of users
print(users)

# Run the test
z_stat, p_value = proportions_ztest(confirms, users, alternative='larger')

# Output
print(f"Z = {z_stat}, p = {p_value}")