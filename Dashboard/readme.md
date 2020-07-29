
Given below is the list of instructions to run the dashboard file

---

+ Copy the entire dashboard directory to the working directory including the assets folder.

+ Install the requirements of the dependencies from the text file. 
  ```python
  pip install -r requirements.txt
  ```
+ Generate the geojson files from the `GEOJSONS` directory.
+ Generate the state level data from the `get_model_predictions.py` to create the `formatted_all_data.csv`.
+ Run the `scratch_dashboard.py` file using 
  ```python
  python scratch_dashboard.py
  ```
+ This will open a port on localhost and display the address `http://127.0.0.1:8050/` where you can click and access the dashboard.

+ After making changes, rerun the code and keep accessing the port and it is recommended to close the tab before rerunning the code.

