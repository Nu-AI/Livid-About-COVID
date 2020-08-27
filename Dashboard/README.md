
Given below is the list of instructions to run the dashboard file on your local machine

---

+ Copy the `Dashboard`,`scripts` and `SIRNet` packages into your working directory. ( Preferably clone the Livid-About-Covid repo entirely )

+ Install the requirements of the dependencies from the text file. 
  ```python
  pip install -r requirements.txt
  ```
  *Note - There is a different `requirements.txt` for the Dashboard inside the `Dashboard` folder.)

+ Update the `parameters.py` file for the required configuration.

+ If running for the first time, generate the state data by running the `state_data.py` to generate the 'formatted_all_data.csv'.
```python
    python get_state_data.py
``` 
* This generated data will be used to populate the dashboard and is necessary to be executed before 
    running the dashboard.
    
+ Run the `texas_dashboard.py` file using 
  ```python
  python texas_dashboard.py
  ```
+ This will open a port on localhost and display the address `http://127.0.0.1:8050/` where you can click and access the dashboard.

+ After making changes, rerun the code and keep accessing the port and it is recommended to close the tab before rerunning the code.

