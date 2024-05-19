### Portfolio 4


### Analysis of an Smart Phone  Dataset


This dataset provides a comprehensive collection of information on various smartphones, enabling a detailed analysis of their specifications and pricing. It encompasses a wide range of smartphones, encompassing diverse brands, models, and configurations, making it a valuable resource for researchers, data analysts, and machine learning enthusiasts interested in the smartphone industry.

The data comes from the spanish website PC componentes. The data was collected using Power Automate.

Fields included:

Smartphone Name: The unique identifier or model name of the smartphone.

Brand: Smartphone brand.

Model: Smartphone brand model.

RAM (Random Access Memory): The amount of memory available for multitasking.

Storage: capacity of the smartphone.

Color: Color of the smarthpone.

Free: Yes/No if the smartphone is attached to a cell company contract.

Price: The cost of the smartphone in the respective currency.

By utilizing this dataset, researchers and analysts can explore patterns, trends, and relationships between smartphone specifications and their pricing. It serves as an excellent resource for tasks such as price prediction, market analysis, and comparison of different smartphone configurations. Whether you are interested in identifying the most cost-effective options or understanding the impact of specific hardware components on smartphone
prices, this dataset offers abundant possibilities for in-depth exploration.



```python
your_name = "Thi Thanh Truc Le"
your_student_id= "46837922"
```

# This analysis aims to explore and analyze a dataset containing information about the specifications and prices of various smartphones from different brands. The dataset provides an overview of the smartphone market and can be utilized for various purposes such as price prediction, market analysis, and comparison of different smartphone configurations.

# DATA EXPLORATION



```python
# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

```


```python
# Load the dataset
smartphones_data = pd.read_csv("smartphones.csv")
```


```python
# Display the first few rows of the dataset
print("Sample data:")
print(smartphones_data.head())
```

    Sample data:
                                         Smartphone     Brand           Model  \
    0            Realme C55 8/256GB Sunshower Libre    Realme             C55   
    1      Samsung Galaxy M23 5G 4/128GB Azul Libre   Samsung      Galaxy M23   
    2  Motorola Moto G13 4/128GB Azul Lavanda Libre  Motorola        Moto G13   
    3      Xiaomi Redmi Note 11S 6/128GB Gris Libre    Xiaomi  Redmi Note 11S   
    4       Nothing Phone (2) 12/512GB Blanco Libre   Nothing       Phone (2)   
    
        RAM  Storage   Color Free  Final Price  
    0   8.0    256.0  Yellow  Yes       231.60  
    1   4.0    128.0    Blue  Yes       279.00  
    2   4.0    128.0    Blue  Yes       179.01  
    3   6.0    128.0    Gray  Yes       279.99  
    4  12.0    512.0   White  Yes       799.00  



```python
# Check the dimensions of the dataset
print("Number of rows and columns:", smartphones_data.shape)
```

    Number of rows and columns: (1816, 8)



```python
# Check the data types of each column
print("Data types of columns:")
print(smartphones_data.dtypes)

```

    Data types of columns:
    Smartphone      object
    Brand           object
    Model           object
    RAM            float64
    Storage        float64
    Color           object
    Free            object
    Final Price    float64
    dtype: object



```python
# Summary statistics for numerical variables
print("Summary statistics for numerical variables:")
print(smartphones_data.describe())
```

    Summary statistics for numerical variables:
                  RAM      Storage  Final Price
    count  1333.00000  1791.000000  1816.000000
    mean      5.96099   162.652150   492.175573
    std       2.66807   139.411605   398.606183
    min       1.00000     2.000000    60.460000
    25%       4.00000    64.000000   200.990000
    50%       6.00000   128.000000   349.990000
    75%       8.00000   256.000000   652.717500
    max      12.00000  1000.000000  2271.280000



```python
# Handle missing values
# Check for missing values
missing_values = smartphones_data.isnull().sum()
print("Missing values:")
print(missing_values)
```

    Missing values:
    Smartphone       0
    Brand            0
    Model            0
    RAM            483
    Storage         25
    Color            0
    Free             0
    Final Price      0
    dtype: int64


# Q1 Split Training and Testing Data
# Q1 Split Training and Testing Data
To investigate whether the size of the training/testing data affects the performance of smartphone price prediction models, please randomly split the smartphone dataset into training and testing sets with different sizes:

Case 1: Training data containing 20% of the entire dataset; Testing data containing 80%.

Case 2: Training data containing 80% of the entire dataset; Testing data containing 20%.

Print the shape of the training and testing sets for both cases.



```python
import pandas as pd
from sklearn.model_selection import train_test_split
```


```python
# Load the smartphone dataset
smartphones_df = pd.read_csv("smartphones.csv")

```


```python
# Case 1: Training data containing 20% of the entire dataset; Testing data containing 80%
train_data_case1, test_data_case1 = train_test_split(smartphones_df, test_size=0.8, random_state=42)

```


```python
# Case 2: Training data containing 80% of the entire dataset; Testing data containing 20%
train_data_case2, test_data_case2 = train_test_split(smartphones_df, test_size=0.2, random_state=42)

```


```python
# Print the shape of training and testing sets for both cases
print("Case 1 - Training Data Shape:", train_data_case1.shape)
print("Case 1 - Testing Data Shape:", test_data_case1.shape)
print("Case 2 - Training Data Shape:", train_data_case2.shape)
print("Case 2 - Testing Data Shape:", test_data_case2.shape)
```

    Case 1 - Training Data Shape: (363, 8)
    Case 1 - Testing Data Shape: (1453, 8)
    Case 2 - Training Data Shape: (1452, 8)
    Case 2 - Testing Data Shape: (364, 8)


#### Q2. Descriptive statistics

With the cleaned data in Q1, please provide the data summarization as below:

Q2.1 Total number of unique brands, unique models, and unique color options

Q2.2 What are the descriptive statistics regarding smartphone prices, including the total count, mean, standard deviation, minimum, and maximum values?

Q2.3 What are the descriptive statistics regarding RAM and storage capacities of smartphones, including mean, standard deviation, minimum, and maximum values?

Q2.4 What are the descriptive statistics regarding smartphone prices by brand, including mean, standard deviation, minimum, and maximum values for each brand?




```python
# Load the cleaned dataset
smartphones_df = pd.read_csv("smartphones.csv")
```


```python
# Display the column names of the DataFrame
print("Column Names:")
print(smartphones_df.columns)

```

    Column Names:
    Index(['Smartphone', 'Brand', 'Model', 'RAM', 'Storage', 'Color', 'Free',
           'Final Price'],
          dtype='object')


# Q2.1Total number of unique brands, unique models, and unique color options:


```python
# Q2.1 
# Total number of unique brands
unique_brands = smartphones_df['Brand'].nunique()
```


```python
# Total number of unique models
unique_models = smartphones_df['Model'].nunique()

```


```python
# Total number of unique color options
unique_colors = smartphones_df['Color'].nunique()

```


```python
# Display the results
print("Q2.1 Results:")
print("Total number of unique brands:", unique_brands)
print("Total number of unique models:", unique_models)
print("Total number of unique color options:", unique_colors)

```

    Q2.1 Results:
    Total number of unique brands: 37
    Total number of unique models: 383
    Total number of unique color options: 17


# Q2.2 Descriptive statistics regarding smartphone prices


```python
# Display the column names of the DataFrame
print("Column Names:")
print(smartphones_df.columns)
```

    Column Names:
    Index(['Smartphone', 'Brand', 'Model', 'RAM', 'Storage', 'Color', 'Free',
           'Final Price'],
          dtype='object')



```python
# Calculate descriptive statistics for smartphone prices
price_statistics = smartphones_df['Final Price'].describe()

# Display the results
print("\nQ2.2 Results:")
print(price_statistics)
```

    
    Q2.2 Results:
    count    1816.000000
    mean      492.175573
    std       398.606183
    min        60.460000
    25%       200.990000
    50%       349.990000
    75%       652.717500
    max      2271.280000
    Name: Final Price, dtype: float64


# Q2.3 Descriptive statistics regarding RAM and storage capacities of smartphones



```python
ram_statistics = smartphones_df['RAM'].describe()
storage_statistics = smartphones_df['Storage'].describe()

print("\nQ2.3 Results for RAM:")
print(ram_statistics)
print("\nQ2.3 Results for Storage:")
print(storage_statistics)


```

    
    Q2.3 Results for RAM:
    count    1333.00000
    mean        5.96099
    std         2.66807
    min         1.00000
    25%         4.00000
    50%         6.00000
    75%         8.00000
    max        12.00000
    Name: RAM, dtype: float64
    
    Q2.3 Results for Storage:
    count    1791.000000
    mean      162.652150
    std       139.411605
    min         2.000000
    25%        64.000000
    50%       128.000000
    75%       256.000000
    max      1000.000000
    Name: Storage, dtype: float64


# Q2.4 Descriptive statistics regarding smartphone prices by brand


```python
price_by_brand_statistics = smartphones_df.groupby('Brand')['Final Price'].describe()

print("\nQ2.4 Results for smartphone prices by brand:")
print(price_by_brand_statistics)
```

    
    Q2.4 Results for smartphone prices by brand:
                count        mean         std     min       25%      50%  \
    Brand                                                                  
    Alcatel       7.0  113.842857   55.172923   70.98   78.4850  100.990   
    Apple       292.0  842.396815  496.019972  109.00  428.7575  802.255   
    Asus          3.0  751.573333  289.957283  444.72  616.8600  789.000   
    BQ            1.0  140.760000         NaN  140.76  140.7600  140.760   
    Blackview    27.0  215.399259   96.248166  109.32  148.2650  177.860   
    CAT           6.0  378.208333  200.470267  116.99  219.7575  411.240   
    Crosscall     7.0  497.754286  152.868714  299.00  379.5000  487.090   
    Cubot        34.0  179.586765   49.347719   84.95  150.9975  173.460   
    Doro          3.0  254.410000   64.899944  179.47  235.6750  291.880   
    Fairphone     1.0  634.190000         NaN  634.19  634.1900  634.190   
    Funker        1.0  220.780000         NaN  220.78  220.7800  220.780   
    Gigaset       3.0  252.346667   57.715094  186.65  231.0750  275.500   
    Google        9.0  516.978889  302.231464  187.40  283.8100  425.000   
    Hammer       21.0  242.617143  104.946926  160.51  179.4200  216.530   
    Honor        27.0  395.626667  403.161928  129.00  225.6750  297.700   
    Huawei       57.0  429.191579  236.317000  109.00  239.9800  356.260   
    LG            1.0  570.740000         NaN  570.74  570.7400  570.740   
    Lenovo        1.0  757.180000         NaN  757.18  757.1800  757.180   
    Maxcom        1.0  123.880000         NaN  123.88  123.8800  123.880   
    Microsoft     1.0  552.390000         NaN  552.39  552.3900  552.390   
    Motorola     57.0  399.899123  300.960404  106.00  186.9800  288.000   
    Nokia        13.0  338.124615  377.962994  111.01  133.0000  196.630   
    Nothing       9.0  657.748889  109.048097  499.00  549.0000  699.000   
    OPPO         92.0  415.113370  275.377249  138.99  202.5900  307.500   
    OnePlus      22.0  571.135909  268.252943  219.00  367.9450  542.005   
    POCO         67.0  328.073731  130.505238  119.00  210.0000  327.000   
    Qubo          3.0  104.646667   11.287588   94.04   98.7150  103.390   
    Realme      117.0  311.799487  174.794257  111.60  192.5900  239.000   
    SPC          11.0  102.763636   14.881778   70.68   97.4750  107.250   
    Samsung     458.0  639.754367  431.395786   95.00  279.5000  546.185   
    Sony          2.0  248.230000  175.687751  124.00  186.1150  248.230   
    Swissvoice    1.0  179.990000         NaN  179.99  179.9900  179.990   
    TCL          36.0  188.088889  121.301507   80.63  119.0025  150.990   
    Ulefone      30.0  259.109333  159.457991   93.81  170.4950  199.130   
    Vivo         27.0  330.651111  254.078270  120.08  189.8400  267.990   
    Xiaomi      351.0  326.669744  222.434877   77.98  179.9900  249.990   
    ZTE          17.0  122.491176   37.957928   60.46   99.6600  119.000   
    
                      75%      max  
    Brand                           
    Alcatel      120.4900   226.98  
    Apple       1159.0000  2119.00  
    Asus         905.0000  1021.00  
    BQ           140.7600   140.76  
    Blackview    224.1500   490.24  
    CAT          541.4550   588.99  
    Crosscall    620.0950   699.00  
    Cubot        205.3200   299.00  
    Doro         291.8800   291.88  
    Fairphone    634.1900   634.19  
    Funker       220.7800   220.78  
    Gigaset      285.1950   294.89  
    Google       654.0000  1166.96  
    Hammer       244.7700   539.68  
    Honor        429.7250  2271.28  
    Huawei       620.0000   999.00  
    LG           570.7400   570.74  
    Lenovo       757.1800   757.18  
    Maxcom       123.8800   123.88  
    Microsoft    552.3900   552.39  
    Motorola     491.6400  1199.00  
    Nokia        348.0400  1499.00  
    Nothing      701.5000   799.00  
    OPPO         499.0000  1299.00  
    OnePlus      687.2500  1333.00  
    POCO         414.5000   689.00  
    Qubo         109.9500   116.51  
    Realme       401.6000   852.59  
    SPC          109.9250   129.91  
    Samsung      866.0000  2191.29  
    Sony         310.3450   372.46  
    Swissvoice   179.9900   179.99  
    TCL          201.9900   568.91  
    Ulefone      258.6425   792.19  
    Vivo         309.0000  1152.98  
    Xiaomi       393.9950  1451.00  
    ZTE          149.0000   190.90  


#### Q3. Plotting and Analysis

Q3.1: Is there a correlation between RAM/storage capacity and smartphone ratings? For example, do smartphones with higher RAM or storage capacities tend to receive higher ratings?

Q3.2: How does the color of smartphones affect ratings? Are there any differences in ratings based on the color options available for smartphones?

Q3.3: Does the brand of smartphones influence ratings? Are certain brands associated with higher or lower ratings compared to others?


```python
#Boxplot chart for correlation between gender and rating
ecommerce_dataset.boxplot("rating", by="gender")
plt.xticks(rotation=90) 

```




    (array([1, 2]), [Text(1, 0, 'F'), Text(2, 0, 'M')])




    
![png](output_35_1.png)
    


# Q3.1: Explore the correlation between RAM/storage capacity and smartphone ratings



```python
import matplotlib.pyplot as plt
import seaborn as sns

# Scatter plot for RAM vs. Ratings
plt.figure(figsize=(10, 6))
plt.scatter(smartphones_df['RAM'], smartphones_df['Final Price'], alpha=0.5)
plt.title('RAM vs. Final Price')
plt.xlabel('RAM (GB)')
plt.ylabel('Final Price')
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(smartphones_df['Storage'], smartphones_df['Final Price'], alpha=0.5)
plt.title('Storage vs. Final Price')
plt.xlabel('Storage (GB)')
plt.ylabel('Final Price')
plt.show()


```


    
![png](output_37_0.png)
    



    
![png](output_37_1.png)
    



```python
# Scatter plot for Storage vs. Ratings
plt.figure(figsize=(10, 6))
plt.scatter(smartphones_df['Storage'], smartphones_df['Final Price'], alpha=0.5)
plt.title('Storage vs. Final Price')
plt.xlabel('Storage (GB)')
plt.ylabel('Final Price')
plt.show()


```


    
![png](output_38_0.png)
    


# Q3.2: Investigate how smartphone color affects ratings:


```python
# Box plot for Color vs. Ratings
plt.figure(figsize=(10, 6))
sns.boxplot(x='Color', y='Final Price', data=smartphones_df)
plt.title('Color vs. Final Price')
plt.xlabel('Color')
plt.ylabel('Final Price')
plt.xticks(rotation=45)
plt.show()
```


    
![png](output_40_0.png)
    


# Q3.3: Analyze the influence of smartphone brands on ratings:


```python
# Box plot for Brand vs. Ratings
plt.figure(figsize=(10, 6))
sns.boxplot(x='Brand', y='Final Price', data=smartphones_df)
plt.title('Brand vs. Final Price')
plt.xlabel('Brand')
plt.ylabel('Final Price')
plt.xticks(rotation=45)
plt.show()
```


    
![png](output_42_0.png)
    


### Visualize, Compare and Analyze the Results
Question 4 

1 Visualize, compare, and analyze the relationship between RAM and smartphone prices. Does higher RAM generally correspond to higher prices?

2 Perform insightful analysis on the correlation between storage capacity and smartphone prices. Are smartphones with larger storage capacities typically more expensive?

3 Investigate the impact of color options on smartphone prices through visualization and analysis. Are certain colors associated with higher or lower prices?

4 Compare smartphone prices across different brands and analyze any trends or patterns. Do certain brands tend to offer higher-priced smartphones compared to others?


```python
import seaborn as sns
import matplotlib.pyplot as plt

# Scatter plot for RAM vs. Smartphone Prices
plt.figure(figsize=(10, 6))
sns.scatterplot(x='RAM', y='Final Price', data=smartphones_df)
plt.title('RAM vs. Smartphone Prices')
plt.xlabel('RAM (GB)')
plt.ylabel('Price')
plt.show()

```


    
![png](output_44_0.png)
    


# Q4.2 Perform insightful analysis on the correlation between storage capacity and smartphone prices


```python
# Scatter plot for Storage vs. Smartphone Prices
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Storage', y='Final Price', data=smartphones_df)
plt.title('Storage vs. Smartphone Prices')
plt.xlabel('Storage (GB)')
plt.ylabel('Price')
plt.show()
```


    
![png](output_46_0.png)
    



```python

```

# Investigate the impact of color options on smartphone prices through visualization and analysis:



```python
# Bar plot for Color vs. Average Smartphone Prices
plt.figure(figsize=(10, 6))
sns.barplot(x='Color', y='Final Price', data=smartphones_df)
plt.title('Color vs. Average Smartphone Prices')
plt.xlabel('Color')
plt.ylabel('Average Price')
plt.xticks(rotation=45)
plt.show()
```


    
![png](output_49_0.png)
    


# Compare smartphone prices across different brands and analyze any trends or patterns:



```python
# Box plot for Brand vs. Smartphone Prices
plt.figure(figsize=(12, 8))
sns.boxplot(x='Brand', y='Final Price', data=smartphones_df)
plt.title('Brand vs. Smartphone Prices')
plt.xlabel('Brand')
plt.ylabel('Price')
plt.xticks(rotation=45)
plt.show()
```


    
![png](output_51_0.png)
    



```python
# your code and solutions
#Eliminate reviews that are more helpful than two.
clean_df = ecommerce_dataset.drop(ecommerce_dataset.index[(ecommerce_dataset['helpfulness'] <= 2)])
#After deleting the hopeless reviews, print the length.
print('The length after removing helpfullness reviews is', len(clean_df))
```


    
![png](output_52_0.png)
    



```python
#Sort people based on their IDs and tally how many ratings they have.
user_rating_count = clean_df.groupby(['userId'])['rating'].count().reset_index(name='rating_count')
# Print the number of users after removing outliers by counting the number of their ratings
print('The number of users after removing outliers by counting the number of their ratings is', len(user_rating_count))

```

    The number of users after removing outliers by counting the number of their ratings is 6535



```python
user_rating_count.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userId</th>
      <th>rating_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Make a list to hold the legitimate users; for this to happen, a user must have at least seven reviews.
genuine_users = user_rating_count[user_rating_count['rating_count'] >= 7]['userId'].tolist()
##Print the number of genuine users
print('The number of genuine users is', len(genuine_users))
```

    The number of genuine users is 267



```python
# Pick the genuine users
genuine_df = clean_df[clean_df['userId'].isin(genuine_users)]
# Print the length of genuine user data
print('The length of genuine user data is', len(genuine_df))
```

    The length of genuine user data is 2741



```python

print('The length of genuine user data is', len(genuine_df))
```

    The length of genuine user data is 2741



```python

```
