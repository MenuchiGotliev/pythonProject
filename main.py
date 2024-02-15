from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import seaborn as sns


class SalesData:

    def __init__(self, data):
        self.data = data

    def eliminate_duplicates(self):
        if self.data is None:
            print("Error: No data available")
            return
        df = pd.DataFrame(self.data)
        df.drop_duplicates(inplace=True)
        self.data = df

    # 5
    def calculate_total_sales(self):
        if self.data is None:
            print("Error: No data available")
            return
        total = self.data.groupby('Product')['Total'].sum()
        return total

    # 6
    def calculate_total_sales_per_month(self):
        try:
            temp = self.data
            temp['Date'] = pd.to_datetime(temp['Date'], errors='coerce', dayfirst=True)
            total_sales_per_month = temp.groupby(temp['Date'].dt.to_period('M'))['Total'].sum()
            print("Total sales per month calculated successfully:")
            print(total_sales_per_month)
            return total_sales_per_month
        except Exception as e:
            print(f"<Menuchi {datetime.time(), {datetime.now().now()} } > Error: {e} <Menuchi>")

    # 7
    def identify_best_selling_product(self):
        if self.data is None:
            print("Error: No data available")
            return
        try:
            quantity = self.data.groupby('Product')['Quantity'].sum()
            max_product = quantity.idxmax()
            return max_product
        except Exception as e:
            print(f"<Menuchi {datetime.time(), {datetime.now().now()} } > Error: {e} <Menuchi>")

    # 8
    def identify_month_with_highest_sales(self):
        temp = self.data
        temp['Date'] = pd.to_datetime(temp['Date'], errors='coerce', dayfirst=True)
        total_sales_per_month = temp.groupby(temp['Date'].dt.to_period('M'))['Total'].sum()
        max_month = total_sales_per_month.idxmax()
        return max_month

    # 9
    def analyze_sales_data(self):
        my_dict = {'best_selling_product': self.identify_best_selling_product(),
                   'month_with_highest_sales': self.identify_month_with_highest_sales()}
        return my_dict

    # 10
    def identify_min_selling_product(self):
        if self.data is None:
            print("Data is empty")
            return None
        product_sales = self.data.groupby('Product')['Quantity'].sum()
        min_selling_product = product_sales.idxmin()
        return min_selling_product

    def calculate_avg_sales_all_months(self):
        if self.data is None:
            print("Data is empty")
            return None
        tmp = self.data
        tmp['Date'] = pd.to_datetime(tmp['Date'], errors='coerce', dayfirst=True)
        monthly_sales = tmp.groupby(tmp['Date'].dt.to_period('M'))['Quantity'].sum()
        avg_sales_all_months = monthly_sales.mean()
        return avg_sales_all_months

    def add_analyze_sales_data(self):
        my_dictionary = self.analyze_sales_data()
        min_selling_product = self.identify_min_selling_product()
        my_dictionary['min_selling_product'] = min_selling_product
        avg_sales_all_months = self.calculate_avg_sales_all_months()
        my_dictionary['average_sales_all_months'] = avg_sales_all_months
        return my_dictionary

    # 11
    def calculate_cumulative_sales(self):
        if self.data is None:
            print("Error: No data available")
            return
        data = self.data
        data['Date'] = pd.to_datetime(data['Date'], errors='coerce', dayfirst=True)
        cumulative_quantities = data.groupby(['Product', data['Date'].dt.to_period('M')])['Total'].sum().groupby(
            level=0).cumsum()
        return cumulative_quantities

    # 12
    def add_90_percent_values_column(self):
        if self.data is None:
            print("Error: No data available")
            return

        self.data['90%_Quantity'] = self.data['Quantity'] * 0.9
        return self.data

    # 13
    def bar_chart_category_sum(self):
        if self.data is None:
            print("Data is empty")
            return
        product_quantity = self.data.groupby('Product')['Quantity'].sum()

        product_quantity.plot(kind='bar', figsize=(10, 20), color='skyblue')
        plt.title('Sum of Quantities Sold for Each Product')
        plt.xlabel('Product')
        plt.ylabel('Sum of Quantity')
        plt.xticks(rotation=45)
        plt.show()

    # 14
    def calculate_mean_quantity(self):
        if self.data is None:
            print("Data is empty")
            return
        total_arr = self.data['Total'].values
        mean = np.mean(total_arr)
        median = np.median(total_arr)
        second_max = np.partition(total_arr, -2)[-2]
        return mean, median, second_max

    # 15
    def filter_by_sellings_or(self):
        condition = (self.data['Quantity'] > 5) | (self.data['Quantity'] == 0)
        filtered_data = self.data[condition]
        return filtered_data

    def filter_by_sellings_and(self):
        condition = (self.data['Price'] > 300) & (self.data['Quantity'] < 2)
        filtered_data = self.data[condition]
        return filtered_data

    # 16
    def divide_by_2(self):
        if self.data is None:
            print("Error: No data available")
            return
        self.data['BLACKFRIDAY'] = self.data['Price'].div(2)
        print(self.data)

    # 19
    def categorize_price(self, price):
        if price <= 50:
            return 'Low'
        elif price <= 100:
            return 'Medium'
        else:
            return 'High'

    def categorize_prices(self):
        if self.data is None:
            print("Error: No data available")
            return
        self.data['Price_Category'] = self.data['Price'].apply(self.categorize_price)
        return self.data


    # Task 7
    # 2
    def read_file(self, file_path: str, file_type: str):
        try:
            if file_type == 'csv':
                return self.read_csv(file_path)
            else:
                with open(file_path, 'r') as file:
                    content = file.read()
                return content

        except FileNotFoundError:
            return None

        except Exception as e:
            return None

    # 3
    def generate_random_number(self, product_name):
        if self.data is None:
            print("Data is empty")
            return
        product_data = self.data[self.data['Product'] == product_name]
        if product_data.empty:
            return "the product not found"
        random_number = np.random.randint(product_data['Quantity'].values[0], self.data['Price'].max())
        arr_random = {'low number': product_data['Quantity'].values[0], 'random number': random_number,
                      'high number': self.data['Price'].max()}
        return arr_random

    # 4
    def print_python_version(self):
        print("Python version:", sys.version)

    # 5
    def process_values(*args):
        result_dict = {}
        for value in args:
            if isinstance(value, (int, float)):
                print(value)
            elif isinstance(value, str):
                result_dict[value] = value
        return result_dict

    # 6
    def print_excel_rows(self):
        # הדפסת 3 השורות הראשונות
        print("First 3 rows:")
        print(self.data.head(3))

        # הדפסת 2 השורות האחרונות
        print("\nLast 2 rows:")
        print(self.data.tail(2))

        # בחירת שורה אקראית
        import random
        random_row_index = random.randint(0, len(self.data) - 1)
        random_row = self.data.iloc[random_row_index]
        print(f"\nRandom row #{random_row_index + 1}:")
        print(random_row)

    # 7
    def move_over_table(self):
        for i in self.data:
            if i == 'Total':
                print("Total:")
                print(self.data[i])
            if i == 'Price':
                print("Price:")
                print(self.data[i])
            if i == 'Quantity':
                print("Quantity:")
                print(self.data[i])
    # 4

    # Task 6
    def plot_total_sales_pie_chart(self):
        total_sales = self.calculate_total_sales()
        if total_sales is None:
            return
        plt.figure(figsize=(8, 8))
        plt.pie(total_sales, labels=total_sales.index, autopct='%1.1f%%')
        plt.title('Total Sales by Product')
        plt.show()

    def plot_total_sales_3d(self):
        total_sales_per_month = self.calculate_total_sales_per_month()
        if total_sales_per_month is None:
            return
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        years = [d.year for d in total_sales_per_month.index]
        months = [d.month for d in total_sales_per_month.index]
        xpos = years
        ypos = months
        zpos = [0] * len(total_sales_per_month)  # z-position is zero
        dx = dy = 0.8
        dz = total_sales_per_month.values
        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='b')
        ax.set_xlabel('Year')
        ax.set_ylabel('Month')
        ax.set_zlabel('Total Sales')
        ax.set_title('Total Sales per Month')
        plt.show()

    def plot_avg_sales_all_months_violin(self):
        avg_sales = self.calculate_avg_sales_all_months()
        if avg_sales is None:
            return
        plt.figure(figsize=(10, 6))
        plt.violinplot([avg_sales], vert=False)
        plt.title('Average Sales per Month (Violin Plot)')
        plt.xlabel('Sales')
        plt.ylabel('Month')
        plt.grid(True)
        plt.show()

    def plot_cumulative_sales_bar(self):
        cumulative_sales = self.calculate_cumulative_sales()
        if cumulative_sales is None:
            return
        plt.figure(figsize=(10, 6))
        cumulative_sales.plot(kind='bar', color='skyblue')
        plt.title('Cumulative Sales per Product')
        plt.xlabel('Product')
        plt.ylabel('Cumulative Sales')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.show()

    def plot_divided_by_2_heatmap(self):
        if self.data is None:
            print("Error: No data available")
            return
        plt.figure(figsize=(10, 6))
        plt.imshow([self.data['BLACKFRIDAY']], cmap='hot', aspect='auto')
        plt.colorbar(label='Price divided by 2')
        plt.title('Price divided by 2 Heatmap')
        plt.xlabel('Data points')
        plt.ylabel('BLACKFRIDAY')
        plt.show()

    def plot_divided_by_2_histogram(self):
        if self.data is None:
            print("Error: No data available")
            return
        plt.figure(figsize=(10, 6))
        plt.hist(self.data['BLACKFRIDAY'], bins=20, color='skyblue', edgecolor='black')
        plt.title('Histogram of Price divided by 2 for BLACKFRIDAY')
        plt.xlabel('Price divided by 2')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()


    def plot_price_category_line(self):
        categorized_data = self.categorize_prices()
        price_counts = categorized_data['Price_Category'].value_counts()
        plt.figure(figsize=(10, 6))
        plt.plot(price_counts.index, price_counts.values, marker='o', linestyle='-')
        plt.title('Number of Prices in Each Category')
        plt.xlabel('Price Category')
        plt.ylabel('Count')
        plt.grid(True)
        plt.show()

    # Seaborn

    def plot_total_sales_seaborn(self):
        total_sales = self.calculate_total_sales()
        if total_sales is None:
            return
        plt.figure(figsize=(10, 6))
        sns.barplot(x=total_sales.index, y=total_sales.values, hue=total_sales.index, palette='viridis',
                    legend=False)
        plt.title('Total Sales per Product')
        plt.xlabel('Product')
        plt.ylabel('Total Sales')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.show()

    def plot_total_sales_per_month_seaborn_stripplot(self):
        total_sales_per_month = self.calculate_total_sales_per_month()
        if total_sales_per_month is None:
            return
        plt.figure(figsize=(10, 6))
        sns.stripplot(x=total_sales_per_month.index.strftime('%Y-%m'), y=total_sales_per_month.values)
        plt.title('Total Sales per Month')
        plt.xlabel('Month')
        plt.ylabel('Total Sales')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.show()


    def plot_total_sales_per_month_seaborn_lineplot(self):
        total_sales_per_month = self.calculate_total_sales_per_month()
        if total_sales_per_month is None:
            return
        plt.figure(figsize=(10, 6))
        sns.lineplot(x=total_sales_per_month.index.strftime('%Y-%m'), y=total_sales_per_month.values)
        plt.title('Total Sales per Month')
        plt.xlabel('Month')
        plt.ylabel('Total Sales')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.show()


    def plot_mean_quantity_boxplot(self):
        mean, median, second_max = self.calculate_mean_quantity()
        data_to_plot = {'Mean': mean, 'Median': median, 'Second Max': second_max}
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=[data_to_plot.values()])
        plt.title('Mean, Median, and Second Max Quantity')
        plt.ylabel('Quantity')
        plt.show()

    def plot_filtered_data(self):
        filtered_data_or = self.filter_by_sellings_or()
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=self.data, x='Price', y='Quantity', color='lightblue', label='Original Data')
        sns.scatterplot(data=filtered_data_or, x='Price', y='Quantity', color='green', label='Filtered Data')
        plt.title('Filtered Data')
        plt.xlabel('Price')
        plt.ylabel('Quantity')
        plt.legend()
        plt.show()


    def plot_price_distribution(self):
        self.categorize_prices()
        plt.figure(figsize=(8, 6))
        sns.countplot(x='Price_Category', data=self.data, hue='Price_Category', palette='viridis', legend=False)
        plt.title('Price Distribution by Category')
        plt.xlabel('Price Category')
        plt.ylabel('Count')
        plt.show()


# יצירת מופע ממחלקת SalesData והכנסת נתונים
sales_data = pd.read_csv('C:\\Users\\User\\Downloads\\סוף הקורס\\חומרים לפרויקט\\YafeNof.csv')

# יצירת אובייקט מסוג SalesData
sales_data_instance = SalesData(sales_data)
# קריאה לפונקציה eliminate_duplicates והדפסת התוכן שחוזר ממנה
print("Calling eliminate_duplicates:")
sales_data_instance.eliminate_duplicates()
print(sales_data_instance.data)

# קריאה לפונקציה calculate_total_sales והדפסת התוכן שחוזר ממנה
print("Calling calculate_total_sales:")
total_sales = sales_data_instance.calculate_total_sales()
print(total_sales)

# קריאה לפונקציה calculate_total_sales_per_month והדפסת התוכן שחוזר ממנה
print("Calling calculate_total_sales_per_month:")
total_sales_per_month = sales_data_instance.calculate_total_sales_per_month()
print(total_sales_per_month)

# קריאה לפונקציה identify_best_selling_product והדפסת התוכן שחוזר ממנה
print("Calling identify_best_selling_product:")
best_selling_product = sales_data_instance.identify_best_selling_product()
print(best_selling_product)

# קריאה לפונקציה identify_month_with_highest_sales והדפסת התוכן שחוזר ממנה
print("Calling identify_month_with_highest_sales:")
month_with_highest_sales = sales_data_instance.identify_month_with_highest_sales()
print(month_with_highest_sales)

# קריאה לפונקציה analyze_sales_data והדפסת התוכן שחוזר ממנה
print("Calling analyze_sales_data:")
sales_analysis = sales_data_instance.analyze_sales_data()
print(sales_analysis)

# קריאה לפונקציה identify_min_selling_product והדפסת התוכן שחוזר ממנה
print("Calling identify_min_selling_product:")
min_selling_product = sales_data_instance.identify_min_selling_product()
print(min_selling_product)

# קריאה לפונקציה calculate_avg_sales_all_months והדפסת התוכן שחוזר ממנה
print("Calling calculate_avg_sales_all_months:")
avg_sales_all_months = sales_data_instance.calculate_avg_sales_all_months()
print(avg_sales_all_months)

# קריאה לפונקציה add_analyze_sales_data והדפסת התוכן שחוזר ממנה
print("Calling add_analyze_sales_data:")
analyzed_data = sales_data_instance.add_analyze_sales_data()
print(analyzed_data)

# קריאה לפונקציה calculate_cumulative_sales והדפסת התוכן שחוזר ממנה
print("Calling calculate_cumulative_sales:")
cumulative_sales = sales_data_instance.calculate_cumulative_sales()
print(cumulative_sales)

# קריאה לפונקציה add_90_percent_values_column והדפסת התוכן שחוזר ממנה
print("Calling add_90_percent_values_column:")
updated_data = sales_data_instance.add_90_percent_values_column()
print(updated_data)

# קריאה לפונקציה bar_chart_category_sum והדפסת התוכן שחוזר ממנה
print("Calling bar_chart_category_sum:")
sales_data_instance.bar_chart_category_sum()

# קריאה לפונקציה calculate_mean_quantity והדפסת התוכן שחוזר ממנה
print("Calling calculate_mean_quantity:")
mean_quantity = sales_data_instance.calculate_mean_quantity()
print(mean_quantity)

# קריאה לפונקציה filter_by_sellings_or והדפסת התוכן שחוזר ממנה
print("Calling filter_by_sellings_or:")
filtered_data_or = sales_data_instance.filter_by_sellings_or()
print(filtered_data_or)

# קריאה לפונקציה filter_by_sellings_and והדפסת התוכן שחוזר ממנה
print("Calling filter_by_sellings_and:")
filtered_data_and = sales_data_instance.filter_by_sellings_and()
print(filtered_data_and)

# קריאה לפונקציה divide_by_2 והדפסת התוכן שחוזר ממנה
print("Calling divide_by_2:")
sales_data_instance.divide_by_2()

# קריאה לפונקציה categorize_prices והדפסת התוכן שחוזר ממנה
print("Calling categorize_prices:")
categorized_data = sales_data_instance.categorize_prices()
print(categorized_data)

# קריאה לפונקציה plot_pie_chart והדפסת התוכן שחוזר ממנה
print("Calling plot_pie_chart:")
sales_data_instance.plot_pie_chart(total_sales)

# קריאה לפונקציה plot_3d_graph והדפסת התוכן שחוזר ממנה
print("Calling plot_3d_graph:")
sales_data_instance.plot_3d_graph()

# קריאה לפונקציה read_file והדפסת התוכן שחוזר ממנה
print("Calling read_file:")
file_content = sales_data_instance.read_file('t', 'file_type')
print(file_content)

# קריאה לפונקציה generate_random_number והדפסת התוכן שחוזר ממנה
print("Calling generate_random_number:")
random_number_info = sales_data_instance.generate_random_number('')
print(random_number_info)

# קריאה לפונקציה print_python_version והדפסת התוכן שחוזר ממנה
print("Calling print_python_version:")
sales_data_instance.print_python_version()

# קריאה לפונקציה process_values והדפסת התוכן שחוזר ממנה
print("Calling process_values:")
sales_data_instance.process_values()

# קריאה לפונקציה print_excel_rows והדפסת התוכן שחוזר ממנה
print("Calling print_excel_rows:")
sales_data_instance.print_excel_rows()

# קריאה לפונקציה move_over_table והדפסת התוכן שחוזר ממנה
print("Calling move_over_table:")
sales_data_instance.move_over_table()

print("Task 6")
# Matplotlib
sales_data_instance.plot_total_sales_pie_chart()
sales_data_instance.plot_total_sales_3d()
sales_data_instance.plot_avg_sales_all_months_violin()
sales_data_instance.plot_cumulative_sales_bar()
sales_data_instance.plot_divided_by_2_heatmap()
sales_data_instance.plot_divided_by_2_histogram()
sales_data_instance.plot_price_category_line()
# Seaborn
sales_data_instance.plot_total_sales_seaborn()
sales_data_instance.plot_total_sales_per_month_seaborn_stripplot()
sales_data_instance.plot_total_sales_per_month_seaborn_lineplot()
sales_data_instance.plot_mean_quantity_boxplot()
sales_data_instance.plot_filtered_data()
sales_data_instance.plot_price_distribution()
