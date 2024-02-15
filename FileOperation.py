import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
class FileOperation:

    def read_excel(self, file_path: str):
        try:
            data = pd.read_csv(file_path)
            return data
        except Exception as e:
            error_message = f"<Menuchi {datetime.time(), {datetime.now().now()} } > Error: {e} <Menuchi>"
            self._handle_error(error_message)
            return None

    def save_to_excel(self, data, file_name: str, folder_path: str):
        try:
            file_path = os.path.join(folder_path, file_name)
            data.to_csv(file_path, index=False)
            print(f"Data has been successfully saved to {file_path}")
        except Exception as e:
            error_message = f"<Menuchi {datetime.time(), {datetime.now().now()} } > Error: {e} <Menuchi>"
            self._handle_error(error_message)

if __name__ == "__main__":
    # ניתוב לקובץ Excel
    file_path = r'C:\Users\User\Downloads\סוף הקורס\חומרים לפרויקט\YafeNof.csv'

    # קריאת נתונים מהקובץ Excel
    file_op = FileOperation()
    data = file_op.read_excel(file_path)

    if data is not None:
        # הדפסת הנתונים הנקראים
        print("Data from Excel:")
        print(data)

        # שמירת הנתונים לקובץ Excel חדש
        folder_path = r'C:\Users\User\Downloads\סוף הקורס\חומרים לפרויקט'
        file_op.save_to_excel(data, "new_file.csv", folder_path)


